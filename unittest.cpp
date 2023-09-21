#include "Deep/node.h"
#include "Deep/nn.h"
#include <iostream>
#include <Eigen/Dense>
// #define NDEBUG
#include <cassert>
#define m_assert(expr, msg) assert(( (void)(msg), (expr) ))

// Node Shared Pointer
using NSP = std::shared_ptr<Deep::Node>;  

int testNode()
{
    // Test transpose backward
    {
    Eigen::MatrixXd weights(2,3);
    weights << 0, 1, 2,
        3, 4, 0.1;
    NSP wPtr {std::make_shared<Deep::Node>(weights)};
    NSP twPtr { wPtr->transpose()};
    assert(twPtr->descendents() == 2);

    // Backward
    Eigen::MatrixXd someGradient(3,2);
    someGradient << 0.1,0.2,
                    -0.5,-0.4,
                    0, 1;
    twPtr->backward(someGradient);
    assert(wPtr->gradient == someGradient.transpose());
    }
    
    // Test weight [3,4] and data [2,4] matmul
    // y = data * weight.transpose() shape [2,3]
    // B=2, m=3, n=4
    {
    Eigen::MatrixXd weights(3,4);
    Eigen::MatrixXd data(2,4);
    Eigen::MatrixXd trueProduct(2,3);
    weights << 0, 1, -1, 1,
        1, 0, 1, 1,
        0, 2, 0, 1;
    data << 1, 2, 3, 4, 
        5, 6, 7, 8;
    trueProduct << 3, 8, 8,
                7, 20, 20;
    NSP W {std::make_shared<Deep::Node>(weights)};
    NSP X {std::make_shared<Deep::Node>(data)};
    NSP result { X * (W->transpose()) };
    assert(result -> data == trueProduct);
    assert(result -> descendents() == 4);
    assert(result->nextNodes[0] == X);
    assert(result->nextNodes[1]->nextNodes[0] == W);
    // Below checks that ptr of W^T is only stored once in the final node
    // because we didn't store the transpose anywhere else. 
    // This make sure the transpose is stored properly and minimally. 
    assert(result->nextNodes[1].use_count() == 1);
    
    // Backward
    Eigen::MatrixXd someGradient(2,3);
    someGradient << 0, 1, 0.5,
                -0.2, 0.5, -0.3;
    result->backward(someGradient);
    Eigen::MatrixXd xTrueGrad(2,4);
    xTrueGrad << 1,1,1,1.5,
                0.5,-0.8,0.7,0;
    Eigen::MatrixXd wTrueGrad(3,4);
    wTrueGrad << -1.0,-1.2,-1.4,-1.6,
                3.5,5.0,6.5,8.0,
                -1.0,-0.8,-0.6,-0.4;
    /* Due to rounding error, we only compare values 
    up to a tolerance of 1e-6 */
    assert(X->gradient.isApprox(xTrueGrad, 1e-6));
    assert(W->gradient.isApprox(wTrueGrad, 1e-6));
    }

    /* Test ReLU */
    {
    Eigen::MatrixXd x(2,3);
    x << 1,5,0,
        -1,-2,3;
    Eigen::MatrixXd trueResult(2,3);
    trueResult << 1,5,0,
                0,0,3;
    NSP xPtr { std::make_shared<Deep::Node>(x) };
    NSP xRelu { xPtr -> relu() };
    assert(xRelu->descendents() == 2);
    assert(xRelu->data == trueResult);

    // Backward
    Eigen::MatrixXd someGradient(2,3);
    someGradient << 0,0.5,1,
                    -0.5, 0.2, -1;
    xRelu->backward(someGradient);
    Eigen::MatrixXd trueGrad(2,3);
    // Mask by the map below based on xPtr matrix
    // true, true, false,
    // false, false, true;
    trueGrad << 0, 0.5, 0, 
                0, 0, -1;
    assert(xPtr->gradient.isApprox(trueGrad, 1e-6));
    }

    /* Test sum */
    {
    Eigen::MatrixXd y(3,4);
    y << 0,1,1,0,
        1,2,-1,1,
        0.5,0.4,1,2;
    // Below might be unused when NDEBUG is enabled.
    [[maybe_unused]] double trueValue {8.9};

    NSP yPtr { std::make_shared<Deep::Node>(y) };
    NSP ySum { yPtr->sum() };
    assert( ySum->descendents() == 2 );
    assert( ySum->data(0,0) == trueValue );

    // Backward without argument
    ySum->backward(); // use grad=1 by default
    Eigen::MatrixXd trueGrad1(3,4);
    trueGrad1.fill(1);
    assert( yPtr->gradient.isApprox(trueGrad1, 1e-6) );
    // Clear gradient before next backward
    yPtr->zeroGrad();
    // Backward with argument
    ySum->backward(10.5); 
    Eigen::MatrixXd trueGrad2(3,4);
    trueGrad2.fill(10.5);
    assert( yPtr->gradient.isApprox(trueGrad2, 1e-6) );
    }

    /* Test composite functions */
    {
    Eigen::MatrixXd x(3,2);
    x << 1,2,
        1,5,
        -1, 2;
    Eigen::MatrixXd weights1(5,2);
    weights1 << 2,0.4,
                1,-0.3,
                -1,0,
                0.5,0.1,
                1,2;
    Eigen::MatrixXd weights2(4,5);
    weights2 << 0,0,1,2,0.1,
                5,4,0.2,-0.4,-0.3,
                -0.1,-1,-2,3,1,
                0,0,5,2,2;
    NSP xPtr { std::make_shared<Deep::Node>(x) };
    NSP w1Ptr { std::make_shared<Deep::Node>(weights1) };
    NSP w2Ptr { std::make_shared<Deep::Node>(weights2) };
    /* Use the logic of 
    L = sum( (relu(x*W1T))*W2T ) */
    NSP LPtr {
        (
            (
                (xPtr*(w1Ptr->transpose()))
                -> relu()
            ) * 
            (
                w2Ptr -> transpose()
            )
        ) -> sum()
    };
    assert(LPtr->descendents() == 9);
    assert(LPtr->nextNodes[0]->nextNodes[1]->nextNodes[0] == w2Ptr);
    assert(LPtr->nextNodes[0]->nextNodes[0]->nextNodes[0]
        ->nextNodes[1]->nextNodes[0] == w1Ptr);
    assert(LPtr->nextNodes[0]->nextNodes[0]->nextNodes[0]->nextNodes[0] == xPtr);
    assert(LPtr->nextNodes[0]->nextNodes[0]->nextNodes[0]->gradientFunction == Deep::gradFn::matMulBackward);
    assert(LPtr->nextNodes[0]->nextNodes[0]->gradientFunction == Deep::gradFn::reluBackward);
    
    LPtr->backward();
    // std::cout << "X gradient\n" << xPtr->gradient << '\n';
    // std::cout << "W1 gradient\n" << w1Ptr->gradient << '\n';
    // std::cout << "W2 gradient\n" << w2Ptr->gradient << '\n';
    }



    std::cout << "All Nodes and gradFn unittests passed.\n\n";
    return 0;
}

int testFC()
{
    Deep::FullyConnected fcLayer(3, 5);
    Eigen::MatrixXd in(4, 3);
    in << 1,2,3,4,5,6,7,8,9,10,11,12;

    Eigen::MatrixXd output { fcLayer(in) };

    Eigen::MatrixXd endGradient(4, 5);
    endGradient << 1, 0.5, 0.5, 1, 1,
            -1, -0.5, -1, -0.5, -0.2,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1;
            
    // Check gradients are zeros
    assert(fcLayer.viewGradients() == Eigen::MatrixXd::Zero(5,3));
    fcLayer.backward(endGradient);
    Eigen::MatrixXd expectGradient(5,3);
    expectGradient << 14,16,18,
                    15.5,17.5,19.5,
                    13.5,15,16.5,
                    16,18.5,21,
                    17.2,20,22.8;
    assert(fcLayer.viewGradients() == expectGradient);
    fcLayer.zeroGrad();
    assert(fcLayer.viewGradients() == Eigen::MatrixXd::Zero(5,3));
    std::cout << "Fully Connected Layer unittest passed.\n";
    return 0; // Everything works well.
}

int testLayer()
{
    testFC();
    std::cout << "All Layers derived classes unittests passed.\n\n";
    return 0;
}

int main()
{
    testNode();
    // testLayer();

    return 0;
}
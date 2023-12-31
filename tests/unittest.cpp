#include "Deep/node.h"
#include "Deep/base.h"
#include "Deep/utility.h"
#include "Deep/nn.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <Eigen/Dense>
// #define NDEBUG
#include <cassert>
#define m_assert(expr, msg) assert(( (void)(msg), (expr) ))

// Node Shared Pointer
using NSP = std::shared_ptr<Deep::Node>;

// Toy model for testing utility.
class MyReg: public Deep::Model
{
    public:
        MyReg()
        {
            // Add layers
            layers["fc1"] = std::unique_ptr<Deep::Layer>(new Deep::FullyConnected(5,3));
            layers["fc2"] = std::unique_ptr<Deep::Layer>(new Deep::FullyConnected(3,2));
            layers["fc3"] = std::unique_ptr<Deep::Layer>(new Deep::FullyConnected(2,10));
            layers["fc4"] = std::unique_ptr<Deep::Layer>(new Deep::FullyConnected(10,1));
        }
        NSP forward(NSP in) override
        {
            // Forward pass
            NSP x1 { Deep::relu(layers["fc1"]->forward(in)) };
            NSP x2 { Deep::relu(layers["fc2"]->forward(x1)) };
            NSP x3 { Deep::relu(layers["fc3"]->forward(x2)) };
            NSP y { layers["fc4"]->forward(x3) };
            return y;
        }
}; 

int testNode()
{
    // Test Node size 
    {
    Eigen::MatrixXd dummy(5,7);
    dummy.fill(0.5);
    NSP dummyPtr {std::make_shared<Deep::Node>(dummy)};
    std::vector<int> theShape {dummyPtr->shape()};
    assert(theShape[0] == 5);
    assert(theShape[1] == 7);
    assert(dummyPtr->size() == 35);
    }
    // Test transpose backward
    {
    Eigen::MatrixXd weights(2,3);
    weights << 0, 1, 2,
        3, 4, 0.1;
    NSP wPtr {std::make_shared<Deep::Node>(weights, Deep::gradFn::accumulateGrad)};
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
    NSP W {std::make_shared<Deep::Node>(weights, Deep::gradFn::accumulateGrad)};
    NSP X {std::make_shared<Deep::Node>(data, Deep::gradFn::accumulateGrad)};
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
    NSP xPtr { std::make_shared<Deep::Node>(x, Deep::gradFn::accumulateGrad) };
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

    NSP yPtr { std::make_shared<Deep::Node>(y, Deep::gradFn::accumulateGrad) };
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

    /* Test MSE */
    {
    Eigen::MatrixXd y1(2,3);
    y1 << 0, 1, 0, 
        -1, 2, 1;
    Eigen::MatrixXd y2(2,3);    
    y2 << 1, 1, -1, 
        -1, 2, 5;
    NSP y1Ptr {std::make_shared<Deep::Node>(y1, Deep::gradFn::accumulateGrad)};
    NSP y2Ptr {std::make_shared<Deep::Node>(y2, Deep::gradFn::accumulateGrad)};
    NSP LPtr { Deep::MSE(y1Ptr, y2Ptr) };
    LPtr->backward();
    Eigen::MatrixXd trueLeft(2,3);
    trueLeft << -1.0/3, 0.0, 1.0/3,
                0.0, 0.0, -4.0/3;
    assert(LPtr->descendents() == 3);
    assert(y1Ptr->gradient.isApprox(trueLeft, 1e-6));
    assert(y2Ptr->gradient.isApprox(-trueLeft, 1e-6));
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
    NSP w1Ptr { std::make_shared<Deep::Node>(weights1, Deep::gradFn::accumulateGrad) };
    NSP w2Ptr { std::make_shared<Deep::Node>(weights2, Deep::gradFn::accumulateGrad) };
    /* Use the logic of 
    L = sum( (relu(x*W1T))*W2T ) */
    NSP LPtr {
        (
            (
                (xPtr*(w1Ptr->transpose()))
                -> relu()
            ) * 
            (
                transpose(w2Ptr)
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
    }

    /* Test many-to-one relation
    Can be achieved by using y = mu(W1 * x) + W2 * x. */
    {
    Eigen::MatrixXd x(2,2);
    x << 1,2,-1,1;
    Eigen::MatrixXd w1(3,2);
    w1 << 0.1,0,1.5,2,1,-1;
    Eigen::MatrixXd w2(3,2);
    w2 << -1,-0.9,-0.7,1,0,0;
    
    NSP xPtr { std::make_shared<Deep::Node>(x) };
    NSP w1Ptr { std::make_shared<Deep::Node>(w1, Deep::gradFn::accumulateGrad) };
    NSP w2Ptr { std::make_shared<Deep::Node>(w2, Deep::gradFn::accumulateGrad) };
    NSP LPtr { Deep::sum(Deep::relu(w1Ptr * xPtr) + (w2Ptr * xPtr)) };
    assert(LPtr->descendents() == 8); // Should be 8 w.r.t. uniqueness.
    LPtr -> backward();
    }

    /* Test visualization 1 */
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
    NSP w1Ptr { std::make_shared<Deep::Node>(weights1, Deep::gradFn::accumulateGrad) };
    NSP w2Ptr { std::make_shared<Deep::Node>(weights2, Deep::gradFn::accumulateGrad) };
    /* Use the logic of 
    L = sum( (relu(x*W1T))*W2T ) */
    NSP LPtr {
        (
            (
                (xPtr*(w1Ptr->transpose()))
                -> relu()
            ) * 
            (
                transpose(w2Ptr)
            )
        ) -> sum()
    };
    LPtr->visualizeGraph("unittest1.svg");
    }

    /* Test visualization 2 */
    {
    MyReg model {};
    Eigen::MatrixXd x(4,5);
    x.fill(0.5);
    Eigen::MatrixXd label(4,1);
    label.fill(1.5);
    NSP xPtr {std::make_shared<Deep::Node>(x)};
    NSP yPtr {model.forward(xPtr)};
    NSP LPtr {Deep::MSE(yPtr, label)};
    /* Use the logic of 
    L = sum( (relu(x*W1T))*W2T ) */
    LPtr->visualizeGraph("unittest2.svg");
    }

    std::cout << "All Nodes and gradFn unittests passed.\n\n";
    return 0;
}

int testFC()
{
    // Test FullyConnected layer without bias
    {
    Deep::FullyConnected fcLayer(3, 5, false);
    Eigen::MatrixXd in(4, 3);
    in << 1,2,3,4,5,6,7,8,9,10,11,12;
    NSP inPtr {std::make_shared<Deep::Node>(in)};
    NSP outPtr { fcLayer.forward(inPtr) };

    Eigen::MatrixXd endGradient(4, 5);
    endGradient << 1, 0.5, 0.5, 1, 1,
            -1, -0.5, -1, -0.5, -0.2,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1;
            
    outPtr->backward(endGradient);
    Eigen::MatrixXd expectGradient(5,3);
    expectGradient << 14,16,18,
                    15.5,17.5,19.5,
                    13.5,15,16.5,
                    16,18.5,21,
                    17.2,20,22.8;
    assert(outPtr->nextNodes[1]->nextNodes[0]->gradient == expectGradient);

    // Use layer's parameters, no bias layer should only have one element.
    assert(fcLayer.params().size() == 1);
    NSP wPtr { fcLayer.params()[0] };
    assert(wPtr->gradient.isApprox(expectGradient, 1e-6));
    }

    // Test FullyConnected layer with bias
    {
    Deep::FullyConnected fcLayer(3, 5);
    Eigen::MatrixXd in(4, 3);
    in << 1,2,3,4,5,6,7,8,9,10,11,12;
    NSP inPtr {std::make_shared<Deep::Node>(in)};
    NSP outPtr { fcLayer.forward(inPtr) };

    Eigen::MatrixXd endGradient(4, 5);
    endGradient << 1, 0.5, 0.5, 1, 1,
                -1, -0.5, -1, -0.5, -0.2,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1;
            
    outPtr->backward(endGradient);
    Eigen::MatrixXd expectWeightGradient(5,3);
    expectWeightGradient << 14,16,18,
                    15.5,17.5,19.5,
                    13.5,15,16.5,
                    16,18.5,21,
                    17.2,20,22.8;
    Eigen::MatrixXd expectBiasGradient(5,1);
    expectBiasGradient << 2, 2, 1.5, 2.5, 2.8;
    assert(outPtr->nextNodes[0]->gradient == expectBiasGradient);
    assert(outPtr->nextNodes[2]->nextNodes[0]->gradient == expectWeightGradient);

    // Use layer's parameters, no bias layer should only have one element.
    std::vector<NSP> wbList { fcLayer.params() };
    assert(wbList.size() == 2);
    NSP wPtr { wbList[0] };
    NSP bPtr { wbList[1] };
    assert(wPtr->gradient.isApprox(expectWeightGradient, 1e-6));
    assert(bPtr->gradient.isApprox(expectBiasGradient, 1e-6));
    }

    // Test composite fcLayer 
    {
    Eigen::MatrixXd x(4,3);
    x << -10,-9,1,
        1, 2, 3,
        2, -1, -3,
        4, 3, 0;
    NSP xPtr {std::make_shared<Deep::Node>(x)};
    Deep::FullyConnected fc1(3, 5);
    Deep::FullyConnected fc2(5, 2);
    NSP LPtr {
        Deep::sum(
            fc2.forward(Deep::relu(fc1.forward(xPtr)))
        )
    };
    assert(LPtr->descendents() == 11);
    }

    std::cout << "Fully Connected Layer unittest passed.\n";
    return 0; // Everything works well.
}

int testModelUtility()
{
    // Test save and load
    MyReg model {};
    std::string modelPath {"./models/cpp-model.json"};
    std::vector<NSP> prevWeights { model.layers["fc1"]->params() };
    model.saveStateDict(modelPath);
    MyReg copyModel {};
    copyModel.loadStateDict(modelPath);
    
    std::vector<NSP> newWeights { copyModel.layers["fc1"]->params() };
    assert(prevWeights[0]->data.isApprox(newWeights[0]->data, 1e-6));
    std::cout << "Model Utilities unittest passed.\n";
    return 0;
}

int testLayer()
{
    testModelUtility();
    testFC();

    std::cout << "All Layers derived classes unittests passed.\n\n";
    return 0;
}

int main()
{
    testNode();
    testLayer();

    return 0;
}
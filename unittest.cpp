#include "Deep/nn.h"
#include "Deep/node.h"
#include <iostream>
#include <Eigen/Dense>
#include <cassert>
#define m_assert(expr, msg) assert(( (void)(msg), (expr) ))

int testAccumulateGrad()
{
    // Testing template=MatrixXd
    Eigen::MatrixXd mat(3, 2);
    mat << 1,2,3,4,5,6;
    Deep::AccumulateGrad acc1(mat);
    // Verify data() method only returns a copy.
    Eigen::MatrixXd cp { acc1.data() };
    cp(0,1) = 10;
    assert(acc1.data()(0,1) == 2);

    // Verify gradient() method only returns a copy.
    Eigen::MatrixXd grad { acc1.gradient() };
    assert(grad == Eigen::MatrixXd::Zero(3,2));

    // Test backward
    Eigen::MatrixXd dummyGradient(3, 2);
    dummyGradient << 0.5, 1, 0.1, 2, -0.1, -0.3;
    acc1.backward(dummyGradient);
    // Single backward
    assert(acc1.gradient() == dummyGradient);
    acc1.backward(dummyGradient);
    // Double backward
    assert(acc1.gradient() == dummyGradient + dummyGradient); 
    
    // Test zeroGrad
    acc1.zeroGrad();
    assert(acc1.gradient() == Eigen::MatrixXd::Zero(3,2));

    // Check that other template could not work
    Eigen::VectorXd vec {Eigen::VectorXd::Zero(3)};
    bool catchIA { false };
    try 
    {
        // Trying to inintialize a node using vector.
        Deep::AccumulateGrad<Eigen::VectorXd> acc2(vec);
    }
    catch (const std::invalid_argument& ia) 
    {
        catchIA = true;
    }
    m_assert(catchIA, "Seems like AccumulateGrad with VectorXd template is implemented, consider remove this test case.");
    std::cout << "AccumulateGrad unittest passed.\n";

    return 0;
}

int testNode()
{
    /* Test all Node derived classes */
    testAccumulateGrad();

    std::cout << "All Node derived classes unittest passed.\n\n";
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
    std::cout << "All Layers derived classes unittest passed.\n\n";
    return 0;
}

int main()
{
    testNode();
    testLayer();

    return 0;
}
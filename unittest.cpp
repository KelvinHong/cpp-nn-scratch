#include "Deep/nn.h"
#include "Deep/node.h"
#include <iostream>
#include <Eigen/Dense>
#include <cassert>
#define m_assert(expr, msg) assert(( (void)(msg), (expr) ))


int testNode()
{
    // Test transpose and double transpose backward
    {
    Eigen::MatrixXd weights(2,3);
    weights << 0, 1, 2,
        3, 4, 0.1;
    Deep::Node wNode(weights);
    Deep::Node tWeights { wNode.transpose() };
    assert(tWeights.descendents() == 2);
    Deep::Node ttWeights { tWeights.transpose() };
    assert(ttWeights.descendents() == 3);
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
    Deep::Node weightsNode(weights);
    Deep::Node dataNode(data);
    Deep::Node tWeightsNode {weightsNode.transpose()};
    Deep::Node result { dataNode * tWeightsNode };
    // This line won't work because rvalue is destroy after initialization.
    // Deep::Node result { dataNode * weightsNode.transpose() };
    assert(result.data == trueProduct);
    int nodes { result.descendents(0, true) };
    assert(nodes == 4);
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
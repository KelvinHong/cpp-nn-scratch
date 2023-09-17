#include "Deep/nn.h"
#include "Deep/node.h"
#include <iostream>
#include <Eigen/Dense>

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
            
    std::cout << "Weights are (randomly initialized) \n" << fcLayer.viewWeights() << '\n';

    std::cout << "Gradients are (before backward)\n" << fcLayer.viewGradients() << '\n';
    fcLayer.backward(endGradient);
    std::cout << "Gradients are (after backward)\n" << fcLayer.viewGradients() << '\n';
    fcLayer.zeroGrad();
    std::cout << "Gradients are (after zeroGrad)\n" << fcLayer.viewGradients() << '\n';
    return 0; // Everything works well.
}

int main()
{
    // testFC();
    Eigen::MatrixXd x { Eigen::MatrixXd::Zero(2,3) };
    Deep::AccumulateGrad<Eigen::MatrixXd> node1(x);

    return 0;
}
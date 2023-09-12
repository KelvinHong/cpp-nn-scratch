#include "Deep/nn.h"
#include <iostream>
#include <Eigen/Dense>

int main()
{
    Deep::FullyConnected fcLayer(3, 5, true);
    Eigen::VectorXd in(3);
    in << 1,5,10;

    Eigen::MatrixXd output { fcLayer(in) };

    Eigen::VectorXd endGradient(5);
    endGradient << 1, 0.5, 0.2, 1, 1;
    fcLayer.backward(endGradient);
    fcLayer.backward(endGradient);
    return 0;
}
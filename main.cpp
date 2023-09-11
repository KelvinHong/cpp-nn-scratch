#include "Deep/nn.h"
#include <iostream>
#include <Eigen/Dense>

int main()
{
    Deep::FullyConnected fcLayer(3, 5);
    Eigen::MatrixXd batchedInput(4, 3);
    batchedInput << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

    Eigen::MatrixXd output { fcLayer(batchedInput) };

    return 0;
}
#ifndef FUNCTIONAL_H
#define FUNCTIONAL_H
#include "node.h"
#include <Eigen/Dense>

using T = Eigen::MatrixXd;

namespace Deep::F
{
Deep::Node relu(Deep::Node& a);

Deep::Node sum(Deep::Node& a);
}

#endif
/* Functions such as Batchnorm, Maxpool, so on, are defined here. 
*/
#ifndef UTILITY_H
#define UTILITY_H
#include "node.h"
#include <Eigen/Dense>


namespace Deep
{

/* Overload Matrix Multiplication */
std::shared_ptr<Node> operator*(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
/* Overload Matrix Addition */
std::shared_ptr<Node> operator+(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
/* Overload Matrix Subtraction */
std::shared_ptr<Node> operator-(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
/* Overload ReLU */
std::shared_ptr<Node> relu(std::shared_ptr<Node> a);
/* Overload Sum */
std::shared_ptr<Node> sum(std::shared_ptr<Node> a);
/* Affine transformation b + x * W. */
std::shared_ptr<Node> affine(std::shared_ptr<Node> b, std::shared_ptr<Node> x, std::shared_ptr<Node> W);
}

#endif
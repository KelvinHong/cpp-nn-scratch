#include "node.h"
#include <Eigen/Dense>

using T = Eigen::MatrixXd;

namespace Deep::F
{
Deep::Node relu(Deep::Node& a)
{
    Eigen::MatrixXd x { a.data };
    x = x.unaryExpr([](double num){
        return (num>0) ? num : 0.0;
    });
    Deep::Node result(x, false, 
        std::vector<Deep::Node*>{&a}, 
        Deep::gradFn::reluBackward);
    return result;
}

Deep::Node sum(Deep::Node& a)
{
    Eigen::MatrixXd scalar(1,1);
    scalar << a.data.sum(); 
    Deep::Node result(scalar, false, 
        std::vector<Deep::Node*>{&a}, 
        Deep::gradFn::sumBackward);
    return result;

}
}
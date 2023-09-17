#ifndef NODE_H
#define NODE_H
#include "base.h"
#include <Eigen/Dense>
#include <vector>

namespace Deep
{

// template <typename Datatype>
class Node
{
};

/* Class for leaf node, mainly for storing gradient. 
Datatype expected to be Eigen::MatrixXd, Eigen::VectorXd.*/
template <typename Datatype>
class AccumulateGrad: public Node
{
    private:
        Datatype& data_; // Store reference to data
        Datatype gradient_; // But store gradient wholely.
    public:
        /* Constructor: with user-provided data and zero gradient. */
        AccumulateGrad(Datatype& userData): data_(userData), gradient_(Datatype{}) {};
        /* data_ getter */
        Datatype data();
        /* gradient_ getter */
        Datatype gradient();
        /* Set gradient to zero */
        void zeroGrad();
};
// AccumulateGrad<Eigen::MatrixXd>;
// AccumulateGrad<Eigen::VectorXd>;


}

#endif
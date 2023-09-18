#include "base.h"
#include "node.h"
#include <Eigen/Dense>
#include <vector>

template <typename Datatype>
Datatype Deep::AccumulateGrad<Datatype>::data()
{
    // Should return a copy of data_.
    return data_;
}

template <typename Datatype>
Datatype Deep::AccumulateGrad<Datatype>::gradient()
{
    return gradient_;
}

template <typename Datatype>
void Deep::AccumulateGrad<Datatype>::zeroGrad()
{
    if (std::is_same<Datatype, Eigen::MatrixXd>::value)
    {
        /* Assumes data_ will be valid nonempty matrix, which
        is the case if the user didn't manually modify it.*/
        gradient_ = Eigen::MatrixXd::Zero(data_.rows(), data_.cols());
    }
    else
    {
        throw std::invalid_argument("Please implement zeroGrad for your datatype."); 
    }
}

template <typename Datatype>
void Deep::AccumulateGrad<Datatype>::backward(const Datatype& grad)
{
    if (std::is_same<Datatype, Eigen::MatrixXd>::value)
    {
        /* Assumes userData will be valid nonempty matrix, which
        is the case if the user didn't manually modify it.*/
        assert(gradient_.rows() == grad.rows());
        assert(gradient_.cols() == grad.cols());
        gradient_ += grad;
    }
    else
    {
        throw std::invalid_argument("Please implement zeroGrad for your datatype."); 
    }
}

/* Although the default template is Eigen::MatrixXd, we still
have to explicitly tell the compiler to compile this
instance. 
Default template only provides semantic convenience
ex: no need to put template when declaring instances. */
template class Deep::AccumulateGrad<Eigen::MatrixXd>;
// template class Deep::AccumulateGrad<Eigen::VectorXd>;

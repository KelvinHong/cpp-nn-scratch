#ifndef NODE_H
#define NODE_H
#include "base.h"
#include <Eigen/Dense>
#include <vector>
#include <iostream>

namespace Deep
{

// template <typename Datatype>
class Node
{
    private: 
        bool isLeaf;
    protected: 
        Node(bool isleaf): isLeaf(isleaf) {};
    template<class Datatype> friend class AccumulateGrad;
};

/* Class for leaf node, mainly for storing gradient. 
Datatype expected to be Eigen::MatrixXd, Eigen::VectorXd.*/
template <typename Datatype=Eigen::MatrixXd>
class AccumulateGrad: public Node
{
    private:
        Datatype gradient_;
    public:
        /* Since we're storing a reference, no use of setting it private. 
        Modify data_ manually will lead to undefined behavior, please only 
        use provided member functions.*/
        Datatype& data_;
        /* Constructor: with user-provided data and zero gradient. */
        AccumulateGrad(Datatype& userData): Node(true), gradient_(Datatype{}), data_(userData) {
            if (std::is_same<Datatype, Eigen::MatrixXd>::value)
            {
                gradient_ = Eigen::MatrixXd::Zero(userData.rows(), userData.cols());
            }
            else
            {
                throw std::invalid_argument("Please implement constructor for your datatype."); 
            }
            
        };
        /* data_ getter */
        Datatype data();
        /* gradient_ getter */
        Datatype gradient();
        /* Set gradient to zero, shape like data_. */
        void zeroGrad();
        /* Backward: Accumulate Gradient to node */
        void backward(const Datatype& grad);
};


}

#endif
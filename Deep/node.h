#ifndef NODE_H
#define NODE_H
#include "base.h"
#include <Eigen/Dense>
#include <vector>
#include <iostream>

namespace Deep
{

template <typename Datatype=Eigen::MatrixXd>
class Node
{
    private: 
        bool isLeaf;
    protected: 
        Node(bool isleaf): isLeaf(isleaf) {};
    public: 
        virtual void backward([[maybe_unused]] const Datatype& grad)
        {
            throw std::invalid_argument("Please implement backward for your derived Node class."); 
        };
        // virtual Datatype transpose()
        // {
        //     throw std::invalid_argument("Please implement transpose for your derived Node class."); 
        // }
        virtual ~Node() = default;
        
    template<typename T> friend class AccumulateGrad;
};

/* Class for leaf node, mainly for storing gradient. 
Datatype expected to be Eigen::MatrixXd, Eigen::VectorXd.*/
template <typename Datatype=Eigen::MatrixXd>
class AccumulateGrad: public Node<Datatype>
{
    private:
        Datatype gradient_;
    public:
        /* Since we're storing a reference, no use of setting it private. 
        Modify data_ manually will lead to undefined behavior, please only 
        use provided member functions.*/
        Datatype& data_;
        /* Constructor: with user-provided data and zero gradient. */
        AccumulateGrad(Datatype& userData): 
                Node<Datatype>(true), gradient_(Datatype{}), 
                data_(userData) {
            if (std::is_same<Datatype, Eigen::MatrixXd>::value)
            {
                gradient_ = Eigen::MatrixXd::Zero(userData.rows(), userData.cols());
            }
            else
            {
                throw std::invalid_argument("Please implement constructor for your datatype."); 
            }
        };
        /* Rule of three */
        // ~AccumulateGrad(){} // dtor
        // AccumulateGrad(const AccumulateGrad& other): // copy constructor
        //     Node<Datatype>(other.isLeaf), gradient_(other.gradient_),
        //     data_(other.data_) {}
        // /* Following copy and swap idiom v
        // https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom */
        // friend void swap(AccumulateGrad& first, AccumulateGrad& second)
        // {
        //     std::swap(first.gradient_, second.gradient_);
        //     std::swap(first.data_, second.data_);
        // }
        // AccumulateGrad& operator=(AccumulateGrad other)
        // {
        //     swap(*this, other);
        //     return *this;
        // }
        /* data_ getter */
        Datatype data(){ return data_;}
        /* gradient_ getter */
        Datatype gradient(){ return gradient_; }
        /* Set gradient to zero, shape like data_. */
        void zeroGrad()
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
        };
        /* Backward: Accumulate Gradient to node */
        void backward(const Datatype& grad) override 
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
        };
        /* Utility: transpose, does not return a f */
        // Datatype transpose()
        // {
        //     return gradient_.transpose();
        // }
};

/* This is intended to be the implementation of TBackward0 in PyTorch.
Unofficially I think this is TupleBackward, since I
will take weights (W) and input (x) and use it to calculate
the gradient of weights.
But to make it more specific and readable, I 
will call it WxBackward.
 */
template <typename Datatype=Eigen::MatrixXd>
class WxBackward: public Node<Datatype>
{
    public:
        /* Next node, should be AccumulateGrad */
        Node<Datatype>* next; 
        /* No need to store a reference to weights. 
        Can be obtained from next->data_
        But it is not needed for gradient calculation. */
        /* Store a reference to input data. */
        Datatype& data;
        /* Constructor: This node contains no gradient information. 
        Also not a leaf.*/
        WxBackward(Datatype& x, Node<Datatype>* nextNode = nullptr):
            Node<Datatype>(false), next(nextNode), data(x) {}
        /* Backward */
        void backward(const Datatype& grad) override 
        {
            // Verify shape of data
            assert(grad.rows() == data.rows());
            /* No need to verify gradient shape match, as it 
            will be handled in the next node. */
            Datatype gradient { grad.transpose() * data };

            /* Propagate to the next node.
            If this throws an error, polymorphism fail.
            Need to verify this: TODO*/
            next -> backward(gradient);
        }

};


}

#endif
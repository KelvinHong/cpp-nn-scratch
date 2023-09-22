#include "base.h"
#include "node.h"
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <map>

using T = Eigen::MatrixXd;

namespace Deep
{

/* This is kind of a bad design, I know,
hopefully I can find a more elegant way if this
project become bigger :) */
std::ostream& operator<<(std::ostream& out, const gradFn gf){
    switch(gf)
    {
        case none: 
            out << "none";
            break;
        case accumulateGrad: 
            out << "accumulateGrad";
            break;
        case transposeBackward: 
            out << "transposeBackward";
            break;
        case matMulBackward: 
            out << "matMulBackward";
            break;
        case reluBackward: 
            out << "reluBackward";
            break;
        case sumBackward: 
            out << "sumBackward";
            break;
        default:
            throw std::invalid_argument("This case has not been recorded yet.");

    }
    
    
    return out;
}

Node::Node(T x, bool isleaf, std::vector<std::shared_ptr<Node>> nextnodes, gradFn gradfn):
    data(x), gradient(T{}), isLeaf(isleaf), nextNodes(nextnodes), gradientFunction(gradfn)
{
    // zero-initialize gradient
    gradient = T::Zero(x.rows(), x.cols());
    // if isleaf is true, verify gradfn is accumulateGrad. 
    // else throw and error.
    if (isleaf && (gradientFunction != Deep::gradFn::accumulateGrad))
    {
        throw std::invalid_argument(
            "For a leaf node, gradient function must be Deep::gradFn::accumulateGrad."
        );
    }
    // if isleaf is true, there shouldn't be any next nodes.
    if (isleaf && (nextNodes.size() != 0))
    {
        throw std::invalid_argument(
            "A leaf node shouldn't have next nodes."
        );
    }
    
};

void Node::zeroGrad()
{
    gradient.fill(0.0);
}

/* Overloading operators */
std::shared_ptr<Node> Node::transpose()
{
    std::shared_ptr<Node> nodePtr(
        std::make_shared<Node>(
            this -> data.transpose(),
            false,
            std::vector<std::shared_ptr<Node>> {shared_from_this()},
            gradFn::transposeBackward
        )
    );
    
    return nodePtr;
}

std::shared_ptr<Node> Node::relu()
{
    Eigen::MatrixXd x { this->data };
    x = x.unaryExpr([](double num){
        return (num>0) ? num : 0.0;
    });
    std::shared_ptr<Node> nodePtr(
        std::make_shared<Node>(
            x,
            false,
            std::vector<std::shared_ptr<Node>> {shared_from_this()},
            gradFn::reluBackward
        )
    );
    return nodePtr;
}

std::shared_ptr<Node> Node::sum()
{
    Eigen::MatrixXd summation(1,1);
    summation << this -> data.sum();
    std::shared_ptr<Node> nodePtr(
        std::make_shared<Node>(
            summation,
            false,
            std::vector<std::shared_ptr<Node>> {shared_from_this()},
            gradFn::sumBackward
        )
    );
    return nodePtr;
}

void Node::backward(T fromGradient)
{
    /* Refuse if gradfn is none. */
    if (this->gradientFunction == gradFn::none)
        return ;
    /* gradient should have the same shape as data */
    assert((this->data.rows() == fromGradient.rows()) 
        && "Gradient doesn't have the same number of rows as data.");
    assert((this->data.cols() == fromGradient.cols()) 
        && "Gradient doesn't have the same number of cols as data.");
    
    // First solve accumulate Grad, which is a leaf node.
    if (this->gradientFunction == gradFn::accumulateGrad)
    {
        gradient += fromGradient;
        return;
    }
    // For other, get the shape of first next node.
    switch (this->gradientFunction)
    { 
        case gradFn::transposeBackward:
        {
            this->nextNodes[0]->backward(fromGradient.transpose());
            break;
        }
        case gradFn::matMulBackward:
        {
            T data1 { this->nextNodes[0]->data };
            T data2 { this->nextNodes[1]->data };
            this->nextNodes[0]->backward(fromGradient * data2.transpose());
            this->nextNodes[1]->backward(data1.transpose() * fromGradient);
            break;
        }
        case gradFn::reluBackward:
        {
            T mask { this->nextNodes[0]->data };
            mask = mask.unaryExpr([](double num){
                return static_cast<double>(num>0);
            });
            this->nextNodes[0]->backward(fromGradient.cwiseProduct(mask));
            break;
        }
        case gradFn::sumBackward:
        {
            T toGradient{ this->nextNodes[0]->data };
            /* sumBackward node must contains a scalar [1,1] matrix. */
            toGradient.fill(fromGradient(0,0));
            this->nextNodes[0]->backward(toGradient);
            break;
        } /* temporary toGradient destroyed here, so it is not stored.
        I should think of a way for user to retain gradients. TODO */ 
            
        default:
            std::cout << "The backward function for " 
                << this->gradientFunction 
                << " is not implemented yet.\n";
            throw std::invalid_argument(
                "The gradient function is not implemented yet."
                " See the last error message above."
            );
    }

}

void Node::backward(double fromGradient)
{
    assert(this->data.size() == 1 && "Backward without parameter should be used "
        "on a node with a scalar data (i.e., 1x1 matrix.)");
    Eigen::MatrixXd dummy(1,1);
    dummy << fromGradient;
    this->backward(dummy);
}

void Node::backward()
{
    /* Backward without parameter can only be used with a 1x1 data. */
    this->backward(1.0);
}

std::shared_ptr<Node> operator*(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
{
    assert(a->data.cols() == b->data.rows());
    std::shared_ptr<Node> matmulPtr(
        std::make_shared<Node>(
            a->data * b->data, 
            false, 
            std::vector<std::shared_ptr<Node>> {a, b}, 
            Deep::gradFn::matMulBackward
        )
    );
    
    return matmulPtr;
}

std::ostream& operator<< (std::ostream &out, const std::shared_ptr<Node>& node)
{
    out << node->data;
    return out;    
}

int Node::descendents(int level, bool verbose)
{   
    if (level == 0 && verbose)
    {
        std::cout << "Showing backward graph for node at " << this << '\n';
        std::cout << "Numbers in parentheses shows how many pointers are "
            "referencing the node.\n";
    }
        
        
    int ret { 1 };
    if (verbose)
        std::cout << std::string(level * 4, '=') 
            << level << ": " 
            << this->gradientFunction 
            /* minus use count by 1 because 
            this function creates a temporary copy of this. */
            << " (" << shared_from_this().use_count() - 1
            << ")\n";
    
    if (this->nextNodes.size() == 0){ return ret; }

    for (const std::shared_ptr<Node>& nodePtr: this->nextNodes)
    {
        ret += nodePtr->descendents(level+1, verbose);
    }
    return ret;
}

int Node::descendents(bool verbose)
{
    return Node::descendents(0, verbose);
}

}

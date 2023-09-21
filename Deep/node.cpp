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

/* Overloading operators */
std::unique_ptr<Node> Node::transpose()
{
    std::unique_ptr<Node> nodePtr(
        std::make_unique<Node>(
            shared_from_this() -> data.transpose(),
            false,
            std::vector<std::shared_ptr<Node>> {shared_from_this()},
            gradFn::transposeBackward
        )
    );
    
    return nodePtr;
}

std::unique_ptr<Node> Node::relu()
{
    Eigen::MatrixXd x { shared_from_this()->data };
    x = x.unaryExpr([](double num){
        return (num>0) ? num : 0.0;
    });
    std::unique_ptr<Node> nodePtr(
        std::make_unique<Node>(
            x,
            false,
            std::vector<std::shared_ptr<Node>> {shared_from_this()},
            gradFn::reluBackward
        )
    );
    return nodePtr;
}

std::unique_ptr<Node> Node::sum()
{
    Eigen::MatrixXd summation(1,1);
    summation << shared_from_this() -> data.sum();
    std::unique_ptr<Node> nodePtr(
        std::make_unique<Node>(
            summation,
            false,
            std::vector<std::shared_ptr<Node>> {shared_from_this()},
            gradFn::sumBackward
        )
    );
    return nodePtr;
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

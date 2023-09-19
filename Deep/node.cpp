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

Node::Node(T x, bool isleaf, std::vector<Node*> nextnodes, gradFn gradfn):
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
Node Node::transpose()
{
    // Create a new node that points to this.
    Node tNode(this->data.transpose(), false, std::vector<Node*> {this}, gradFn::transposeBackward);
    return tNode;
}

Node Node::operator*(Node& other)
{
    assert(this->data.cols() == other.data.rows());
    Node matmulNode(this->data * other.data, false, 
        std::vector<Node*> {this, &other}, Deep::gradFn::matMulBackward);
    
    return matmulNode;
}

std::ostream& operator<< (std::ostream &out, const Node& node)
{
    out << node.data;
    return out;    
}

int Node::descendents(int level, bool verbose)
{   
    int ret { 1 };
    if (verbose)
        std::cout << std::string(level * 4, '=') << 
            this->gradientFunction << '\n';
    if (this->nextNodes.size() == 0){ return ret; }

    for (Node* nodePtr: this->nextNodes)
    {
        ret += nodePtr->descendents(level+1, verbose);
    }
    return ret;
}

}

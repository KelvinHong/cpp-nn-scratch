#ifndef NODE_H
#define NODE_H
#include "base.h"
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <memory>

using T = Eigen::MatrixXd;
namespace Deep
{

enum gradFn {
    none, // none shouldn't be used in this stage where we assume every node requires gradient. 
    accumulateGrad,
    transposeBackward,
    matMulBackward,
    reluBackward,
    sumBackward,
};

class Node : public std::enable_shared_from_this<Node>
{
    public: 
        T data;
        T gradient;
        bool isLeaf;
        std::vector<std::shared_ptr<Node>> nextNodes;
        gradFn gradientFunction;
        /* Constructor. 
        Gradient must be zero-initialized, same shape as data.
        isLeaf should be used based on situation, 
        if isLeaf is true, gradFn will be accumulateGrad. 
        gradientFunction used to govern backward() behavior.
        */
        Node(T x, bool isleaf = true, 
            std::vector<std::shared_ptr<Node>> nextnodes = std::vector<std::shared_ptr<Node>>{}, 
            gradFn gradfn = gradFn::accumulateGrad);

        /* Overload transpose */
        std::unique_ptr<Node> transpose();
        /* ReLU */
        std::unique_ptr<Node> relu();
        /* Sum */
        std::unique_ptr<Node> sum();
        

        /* Backward */
        /* Backward for intermediate nodes. */
        void backward(T fromGradient);
        /* Backward for Loss, they typically uses default gradient of 1. */
        void backward();

        /* Custom destructor: Destruct self and every next nodes 
        except accumulateGrad. */
        // ~Node();

        /* Overload << */
        friend std::ostream& operator<< (std::ostream &out, const std::shared_ptr<Node>& nodePtr);
        /* Show descendents (only their gradfn),
        return the total number of nodes */
        int descendents(int level = 0, bool verbose = false);
        int descendents(bool verbose);
};

/* Overload Matrix Multiplication */
std::shared_ptr<Node> operator*(std::shared_ptr<Node> a, std::shared_ptr<Node> b);

}

#endif
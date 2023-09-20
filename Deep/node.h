#ifndef NODE_H
#define NODE_H
#include "base.h"
#include <Eigen/Dense>
#include <vector>
#include <iostream>

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

class Node
{
    public: 
        T data;
        T gradient;
        bool isLeaf;
        std::vector<Node*> nextNodes;
        gradFn gradientFunction;
        /* Constructor. 
        Gradient must be zero-initialized, same shape as data.
        isLeaf should be used based on situation, 
        if isLeaf is true, gradFn will be accumulateGrad. 
        gradientFunction used to govern backward() behavior.
        */
        Node(T x, bool isleaf = true, std::vector<Node*> nextnodes = std::vector<Node*>{}, 
            gradFn gradfn = gradFn::accumulateGrad);

        /* Overloading common mathematical operators: */

        /* Overload transpose */
        Node transpose();
        /* Overload Matrix Multiplication */
        Node operator*(Node& other);
        
        

        /* Backward */
        /* Backward for Loss, they typically uses default gradient of 1. */
        void backward(); 
        /* Backward for intermediate nodes. */
        void backward(T fromGradient);

        /* Custom destructor: Destruct self and every next nodes 
        except accumulateGrad. */
        // ~Node();

        /* Overload << */
        friend std::ostream& operator<< (std::ostream &out, const Node* node);
        /* Show descendents (only their gradfn),
        return the total number of nodes */
        int descendents(int level = 0, bool verbose = false);
        int descendents(bool verbose);
};



}

#endif
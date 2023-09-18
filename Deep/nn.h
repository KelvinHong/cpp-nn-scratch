#ifndef NN_H
#define NN_H
#include "base.h"
#include <Eigen/Dense>
#include <random>

/* The code need some re-sturcture:
Backward should be a responsibility of Node(s)
not Layer(s). 
We should let Node reference Layer's parameters
then backward on Node will affect Layer's 
parameters. */
namespace Deep
{
extern std::mt19937 gen;


/* FullyConnected layer will not be using 
tensor, as 2D-matrices are sufficient for 
all operations. */ 
class FullyConnected: public Layer
{
    private: 
        Eigen::MatrixXd weights;
        Eigen::MatrixXd gradients;
        Eigen::MatrixXd cacheInput;
    public:
        int in_c;
        int out_c;
        /* Constructor determines the weights dimension, 
        then initialize weights */ 
        FullyConnected(const int& in_channel, const int& out_channel);
        /* Batched Forward call (overload) */
        Eigen::MatrixXd operator()(const Eigen::MatrixXd& in); 
        /* Calculate batched backward gradients */
        void backward(const Eigen::MatrixXd& endGradient);
        /* Zero the gradient */
        void zeroGrad();
        /* Handy function to inspect the weights */
        const decltype(weights)& viewWeights();
        /* Handy function to inspect the gradients */
        const decltype(gradients)& viewGradients();
};
}

#endif
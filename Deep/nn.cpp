#include "nn.h"
#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <assert.h>

std::mt19937 Deep::gen(std::random_device{}());


Deep::FullyConnected::FullyConnected(const int& in_channel, const int& out_channel, const bool& use_grad):
    weights(Eigen::MatrixXd()), gradients(Eigen::MatrixXd()), cacheInput(Eigen::VectorXd()),
    in_c(in_channel), out_c(out_channel), requires_grad(use_grad)
{
    // Check in_c and out_c are positive;
    if (in_channel <= 0 || out_channel <= 0)
        throw std::invalid_argument("Input channel and output channel must both be positive. ");

    requires_grad = use_grad;
    // Random generator learned from 
    // https://stackoverflow.com/questions/21292881/matrixxfrandom-always-returning-same-matrices
    double xavierGap { std::pow(in_c, -0.5) };
    std::uniform_real_distribution<double> weightDis(-xavierGap, xavierGap);

    // Initialize weights.
    weights = Eigen::MatrixXd::NullaryExpr(out_c,in_c,
        [&](){return weightDis(Deep::gen);}
    );
    // Zero-initialize gradients.
    gradients = Eigen::MatrixXd::Zero(out_c, in_c);
    // Zero-initialize cacheInput
    cacheInput = Eigen::VectorXd::Zero(in_c);
}


Eigen::MatrixXd Deep::FullyConnected::operator()(const Eigen::VectorXd& in)
{
    /*This is the single pass forward function of linear layer, mimicking 
    PyTorch's Layer.__call__() signature, we overload the 
    parenthesis operator. 

    input (in) assumed to be of shape [in_c]
    output will be of shape [out_c].

    This is experimental as mainstream usage are probably using batched input. 
    This is just for toy testing. 
    */  
    assert(in.size() == in_c);
    /* This creates a copy as mentioned in documentation. 
    https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html */
    cacheInput = in; 
    Eigen::VectorXd out {weights * in};

    return out;
}

Eigen::MatrixXd Deep::FullyConnected::operator()(const Eigen::MatrixXd& in)
{
    /*This is the batched forward function of linear layer, mimicking 
    PyTorch's Layer.__call__() signature, we overload the 
    parenthesis operator. 

    input (in) assumed to be of shape [B, in_c]
    output will be of shape [B, out_c].
    */  
    assert(in.cols() == in_c);

    Eigen::MatrixXd out {in * weights.transpose()};

    return out;
}

// This is a useless prototype
void Deep::FullyConnected::backward(const Eigen::VectorXd& endGradient)
{
    // endGradient is the gradient coming from later layer. 
    // Check shape agreement.
    assert(endGradient.size() == weights.rows());
    
    // Increment the gradient instead of assign
    // In case user want to intentionally backward multiple times.    
    gradients += (endGradient * cacheInput.transpose());
}

void Deep::FullyConnected::zeroGrad()
{   
    // Need to check this works properly
    // i.e., it setZero inplace rather than copy.
    gradients.setZero();
}

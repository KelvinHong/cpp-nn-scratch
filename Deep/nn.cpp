#include "nn.h"
#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <assert.h>

std::mt19937 Deep::gen(std::random_device{}());


Deep::FullyConnected::FullyConnected(const int& in_channel, const int& out_channel, const bool& use_grad):
    weights(Eigen::MatrixXd()), gradients(Eigen::MatrixXd()), cacheInput(Eigen::MatrixXd()),
    in_c(in_channel), out_c(out_channel), requires_grad(use_grad)
{
    // Check in_c and out_c are positive;
    if (in_channel <= 0 || out_channel <= 0)
        throw std::invalid_argument("Input channel and output channel must both be positive. ");

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

    /* Only remember input in cache if requiring gradient calculation */
    if (requires_grad)
    {
        cacheInput = in;
    }
        
    Eigen::MatrixXd out {in * weights.transpose()};

    return out;
}

void Deep::FullyConnected::backward(const Eigen::MatrixXd& endGradient)
{
    // endGradient is the gradient coming from later layer. 
    // Check shape agreement.
    /*Assume endGradient.shape == [B, m],
    weights.shape == [m, n],
    cacheInput.shape == [B, n].
    */
   assert(requires_grad);
    assert(endGradient.cols() == weights.rows());
    assert(endGradient.rows() == cacheInput.rows());
    // Increment the gradient instead of assign
    // In case user want to intentionally backward multiple times.    
    gradients += (endGradient.transpose() * cacheInput);
    // std::cout << "Input is\n" << cacheInput << '\n';
    // std::cout << "Supplied gradient is\n" << endGradient << '\n';
    // std::cout << "weights are\n" << weights << '\n';
    // std::cout << "gradients are\n" << gradients << '\n';
}

void Deep::FullyConnected::zeroGrad()
{   
    gradients.setZero();
}

const decltype(Deep::FullyConnected::weights)& Deep::FullyConnected::viewWeights() 
{ 
    return weights;
}

const decltype(Deep::FullyConnected::gradients)& Deep::FullyConnected::viewGradients() 
{ 
    return gradients;
}

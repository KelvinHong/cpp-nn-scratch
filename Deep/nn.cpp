#include "nn.h"
#include <Eigen/Dense>
#include <random>
#include <iostream>

std::mt19937 Deep::gen(std::random_device{}());

Deep::FullyConnected::FullyConnected(const int& in_channel, const int& out_channel, const bool& use_grad = true):
    weights(Eigen::MatrixXd()), gradients(Eigen::MatrixXd()), 
    in_c(in_channel), out_c(out_channel), requires_grad(use_grad)
{
    // Check in_c and out_c are positive;
    if (in_c <= 0 || out_c <= 0)
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

Eigen::MatrixXd Deep::FullyConnected::operator()(const Eigen::MatrixXd& in) const
{
    /*This is the forward function of linear layer, mimicking 
    PyTorch's Layer.__call__() signature, we overload the 
    parenthesis operator. 
    
    input (in) assumed to be of shape [B, in_c]
    output will be of shape [B, out_c].

    Will need to update this function in the future for gradient calculation.
    */  
    Eigen::MatrixXd out {in * weights.transpose()};

    return out;
}


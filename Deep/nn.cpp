#include "base.h"
#include "node.h"
#include "utility.h"
#include "nn.h"
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <cassert>
#include <random>


Deep::FullyConnected::FullyConnected(int in_channel, int out_channel, bool use_bias, bool requires_grad):
    useBias(use_bias), requiresGrad(requires_grad), in_c(in_channel), out_c(out_channel),
    weights(nullptr), biases(nullptr)
{
    gradFn mode { requiresGrad ? gradFn::accumulateGrad : gradFn::none };
    // Check in_c and out_c are positive;
    if (in_channel <= 0 || out_channel <= 0)
        throw std::invalid_argument("Input channel and output channel must both be positive. ");

    // Random generator learned from 
    // https://stackoverflow.com/questions/21292881/matrixxfrandom-always-returning-same-matrices
    double xavierGap { std::pow(in_c, -0.5) };
    std::uniform_real_distribution<double> weightDis(-xavierGap, xavierGap);

    // Initialize weights.
    Eigen::MatrixXd weightsData = Eigen::MatrixXd::NullaryExpr(out_c,in_c,
        [&](){return weightDis(Deep::gen);}
    );
    weights = std::make_shared<Deep::Node>(weightsData, mode);
    // If use bias, zero initialize bias.
    if (useBias)
    {
        Eigen::MatrixXd biasesData = Eigen::MatrixXd::Zero(out_c, 1);
        biases = std::make_shared<Deep::Node>(biasesData, mode);
    }
}



NSP Deep::FullyConnected::forward(NSP in)
{
    /*This is the batched forward function of linear layer, mimicking 
    PyTorch's Layer.__call__() signature, we overload the 
    parenthesis operator. 

    input (in) assumed to be of shape [B, in_c]
    output will be of shape [B, out_c].
    */  
    assert(in->data.cols() == in_c && "Input data dimension doesn't match FC Layer's in_c.");

    NSP out;
    if (useBias)
    {
        out = Deep::affine(biases, in, weights->transpose());
    }
    else
    {
        out = in * (weights->transpose());
    }

    return out;
}

std::vector<NSP> Deep::FullyConnected::params()
{
    std::vector<NSP> ret {weights};
    if (useBias) ret.push_back(biases);
    return ret;
}

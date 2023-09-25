#include "optimizer.h"
#include <memory>
#include <cassert>

// Node Shared Pointer
using NSP = std::shared_ptr<Deep::Node>;

namespace Deep::Optim
{
Optimizer::Optimizer(std::vector<NSP> parameters):
    params(parameters)
{
}

void Optimizer::step()
{
    throw std::invalid_argument("Please implement gradient step for your optimizer.");
}

SGD::SGD(std::vector<NSP> parameters, double learningRate, double momentumValue):
    Optimizer(parameters), lr(learningRate), momentum(momentumValue)
{
    assert((learningRate > 0) && "Learning rate should be a positive decimal number.");
    assert((momentumValue >= 0) && (momentumValue <= 1) && "Momentum should be a decimal number within 0 to 1.");
}

void SGD::step()
{
    
}

}
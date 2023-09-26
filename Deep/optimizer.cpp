#include "base.h"
#include "optimizer.h"
#include <memory>
#include <cassert>
#include <vector>



namespace Deep::Optim
{
Optimizer::Optimizer(std::vector<std::pair<std::string, NSP>> namedParams):
    namedParameters(namedParams)
{
}

Optimizer::~Optimizer(){}

void Optimizer::step()
{
    throw std::invalid_argument("Please implement gradient step for your optimizer.");
}

SGD::SGD(std::vector<std::pair<std::string, NSP>> namedParams, double learningRate, double momentumValue):
    Optimizer(namedParams), lr(learningRate), momentum(momentumValue), 
    prevParameters(std::unordered_map<std::string, MAT>{})
{
    assert((namedParams.size() > 0) && "There are no trainable paramaters, you probably do not need an optimizer.");
    assert((learningRate > 0) && "Learning rate should be a positive decimal number.");
    assert((momentumValue >= 0) && (momentumValue <= 1) && "Momentum should be a decimal number within 0 to 1.");
}

/* This function can be optimized, as it now uses two linear passes.
It is actually possible to use single linear pass but it will result in 
code duplication. Need to measure to see if need to optimize it. */
void SGD::step()
{
    // Update gradient with momentum
    if (prevParameters.size() == 0)
    {
        // First time stepping
        for (std::pair<std::string, NSP> namedParam: namedParameters)
        {
            prevParameters[namedParam.first] = namedParam.second->gradient;
        }
    }
    else
    {
        // Second time stepping and later
        for (std::pair<std::string, NSP> namedParam: namedParameters)
        {
            prevParameters[namedParam.first] = momentum * prevParameters[namedParam.first]
                + namedParam.second->gradient;
        }        
    }
    // Update parameters
    for (std::pair<std::string, NSP> namedParam: namedParameters)
    {
        namedParam.second->data -= lr * prevParameters[namedParam.first];
    }        
}

}
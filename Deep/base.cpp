#include "base.h"
#include "node.h"
#include <vector>
#include <unordered_map>
#include <memory>
// Node Shared Pointer
using NSP = std::shared_ptr<Deep::Node>;

std::vector<NSP> Deep::Layer::params()
{
    throw std::invalid_argument("Please implement parameters in derived classes.");
}

NSP Deep::Layer::forward([[maybe_unused]] NSP in)
{
    throw std::invalid_argument("Please implement forward pass in derived classes.");
}

Deep::Model::Model(): layers(std::unordered_map<std::string, std::unique_ptr<Layer>> {}) 
{

}

std::vector<NSP> Deep::Model::parameters()
{
    std::vector<NSP> ret {};
    for (auto it=layers.begin(); it!=layers.end(); ++it)
    {
        std::vector<NSP> pars { (*it).second->params() };
        ret.insert(ret.end(),pars.begin(),pars.end());
    }
    return ret;
}

NSP Deep::Model::forward([[maybe_unused]] NSP in)
{
    throw std::invalid_argument("Please implement forward pass in derived classes.");
}
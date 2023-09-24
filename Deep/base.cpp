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
    // return std::vector<NSP>{};
}

Deep::Model::Model(): layers(std::unordered_map<std::string, Layer> {}) 
{

}

std::vector<NSP> Deep::Model::parameters()
{
    std::vector<NSP> ret {};
    for (auto it=layers.begin(); it!=layers.end(); ++it)
    {
        std::vector<NSP> pars { (*it).second.params() };
        ret.insert(ret.end(),pars.begin(),pars.end());
    }
    return ret;
}

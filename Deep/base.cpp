#include "base.h"
#include "node.h"
#include <vector>
#include <random>
#include <unordered_map>
#include <memory>
// Node Shared Pointer
using NSP = std::shared_ptr<Deep::Node>;

// Using random_device is not stable, might have to change this later.
std::mt19937 Deep::gen(std::random_device{}());

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

std::vector<std::pair<std::string, NSP>> Deep::Model::namedParameters()
{
    std::vector<std::pair<std::string, NSP>> ret {};
    for (auto it=layers.begin(); it!=layers.end(); ++it)
    {
        std::vector<NSP> pars { (*it).second->params() };
        std::string layerName { (*it).first };
        for (int paramNum = 1; paramNum <= static_cast<int>(pars.size()); paramNum++)
        {
            ret.push_back(std::pair<std::string, NSP> {
                layerName + '.' + std::to_string(paramNum), 
                pars[paramNum-1]
            });
        }
    }
    return ret;
}

void Deep::Model::showParametersInfo()
{
    std::cout << "Printing Model Parameters Information:\n";
    std::vector<std::pair<std::string, NSP>> fullParams {namedParameters()};
    int paramsCount { 0 };
    for (auto it=fullParams.begin(); it!=fullParams.end(); ++it)
    {
        std::cout << (*it).first << ": Data shape (" 
                << (*it).second->data.rows() << ", "
                << (*it).second->data.cols() << ");\n";
        paramsCount += static_cast<int>((*it).second->data.size());
    }
    std::cout << "Total: " << paramsCount << " parameters.\n";
}

std::vector<NSP> Deep::Model::parameters()
{
    std::vector<std::pair<std::string, NSP>> fullParams {namedParameters()};
    std::vector<NSP> ret {};
    for (auto it=fullParams.begin(); it!=fullParams.end(); ++it)
    {
        ret.push_back((*it).second);
    }
    return ret;
}

NSP Deep::Model::forward([[maybe_unused]] NSP in)
{
    throw std::invalid_argument("Please implement forward pass in derived classes.");
}
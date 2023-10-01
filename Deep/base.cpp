#include "base.h"
#include "node.h"
#include <sys/stat.h>
#include <vector>
#include <random>
#include <unordered_map>
#include <memory>
#include <fstream>
#include <cassert>
// Node Shared Pointer
using NSP = std::shared_ptr<Deep::Node>;

inline bool fileExist (const std::string& name) {
    struct stat buffer;   
    return (stat (name.c_str(), &buffer) == 0); 
}

namespace Deep
{
// Using random_device is not stable, might have to change this later.
std::mt19937 gen(std::random_device{}());

std::vector<NSP> Layer::params()
{
    throw std::invalid_argument("Please implement parameters in derived classes.");
}

NSP Layer::forward([[maybe_unused]] NSP in)
{
    throw std::invalid_argument("Please implement forward pass in derived classes.");
}

Model::Model(): layers(std::unordered_map<std::string, std::unique_ptr<Layer>> {}) 
{

}

std::vector<std::pair<std::string, NSP>> Model::namedParameters()
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

void Model::showParametersInfo()
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

std::vector<NSP> Model::parameters()
{
    std::vector<std::pair<std::string, NSP>> fullParams {namedParameters()};
    std::vector<NSP> ret {};
    for (auto it=fullParams.begin(); it!=fullParams.end(); ++it)
    {
        ret.push_back((*it).second);
    }
    return ret;
}

void Model::saveStateDict(std::string modelPath)
{
    std::vector<std::pair<std::string, NSP>> NP {namedParameters()};
    json modelStateDict {};
    for (auto p: NP)
    {
        modelStateDict[p.first] = p.second->data;
    }

    std::ofstream file(modelPath);
    file << modelStateDict;
}

void Model::loadStateDict(std::string modelPath)
{
    if (!fileExist(modelPath))
    {
        throw std::invalid_argument("The model path provided doesn't exist. ");
    }
    std::ifstream file(modelPath);
    json object = json::parse(file);
    // std::cout << object << '\n';

    std::vector<std::pair<std::string, NSP>> NP {namedParameters()};
    /* Using shared pointer feature, we load the eigen matrix into the named parameters. 
    Throws an error if one of the model's key cannot be found from the json file. */
    for (auto pair: NP)
    {
        if (!object.contains(pair.first))
        {   
            std::string message {"Error occurred during loading state dict into the model. Key not found from file: "};
            message += pair.first;
            throw std::invalid_argument(message);
        }
        // Load the Eigen Matrix into this
        T toBeLoaded {object.at(pair.first).template get<T>()};
        // Make sure two sides got the same shape
        assert(pair.second->data.rows() == toBeLoaded.rows() );
        assert(pair.second->data.cols() == toBeLoaded.cols() );
        // Assign it
        pair.second->data = toBeLoaded;
    }
    
}

NSP Model::forward([[maybe_unused]] NSP in)
{
    throw std::invalid_argument("Please implement forward pass in derived classes.");
}
}


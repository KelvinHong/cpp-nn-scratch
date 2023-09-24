#ifndef BASE_H
#define BASE_H
#include "node.h"
#include <vector>
#include <unordered_map>
#include <memory>

// Node Shared Pointer
using NSP = std::shared_ptr<Deep::Node>;

namespace Deep
{
/* All layer types should inherited from Layer class. 
Layer is the base class of all layers that have parameters.
Ex: Fully Connected Layer should be derived from Layer,
but MaxPooling should not be. 
We do not put this in a single header file because we 
anticipate its usage in many different codes, such as 
nn.cpp, cnn.cpp, rnn.cpp, and so on.*/
class Layer 
{
    public:
        virtual std::vector<NSP> params();
        virtual NSP forward(NSP in);
        virtual ~Layer() = default;
};


/* This is the base class for model
that consists of multiple layers. */
class Model 
{
    public:
        std::unordered_map<std::string, std::unique_ptr<Layer>> layers;
        Model();
        std::vector<NSP> parameters();
        virtual NSP forward(NSP in);
        virtual ~Model() = default;
};

}


#endif
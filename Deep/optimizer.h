#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "base.h"
#include <vector>

// Node Shared Pointer
using NSP = std::shared_ptr<Deep::Node>;

namespace Deep::Optim
{
class Optimizer
{
    public: 
        std::vector<NSP> params;
        /* Constructor */
        Optimizer(std::vector<NSP> parameters);
        /* Optimizer Step */
        virtual void step();
};

class SGD: public Optimizer
{
    public:
        const double lr;
        const double momentum;
        SGD(std::vector<NSP> parameters, double learningRate=1e-5, double momentumValue=0.9);
        void step() override;
};

}

#endif
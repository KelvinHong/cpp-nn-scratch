#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "base.h"
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>

// Node Shared Pointer
using NSP = std::shared_ptr<Deep::Node>;
using MAT = Eigen::MatrixXd;

namespace Deep::Optim
{
class Optimizer
{
    public: 
        std::vector<std::pair<std::string, NSP>> namedParameters;
        /* Constructor */
        Optimizer(std::vector<std::pair<std::string, NSP>> namedParams);
        /* Optimizer Step */
        virtual void step();
        virtual ~Optimizer();
};

class SGD: public Optimizer
{
    public:
        const double lr;
        const double momentum;
        /* Not storing previous params as NSP to make memory more efficient. */
        std::unordered_map<std::string, MAT> prevParameters;
        SGD(std::vector<std::pair<std::string, NSP>> namedParams, double learningRate=1e-5, double momentumValue=0.9);
        void step() override;
        // ~SGD() override = default;
};

}

#endif
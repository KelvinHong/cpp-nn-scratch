#ifndef NN_H
#define NN_H
#include "base.h"
#include "node.h"
#include "utility.h"
#include <Eigen/Dense>

// Node Shared Pointer
using NSP = std::shared_ptr<Deep::Node>;
namespace Deep
{


/* FullyConnected layer will not be using 
tensor, as 2D-matrices are sufficient for 
all operations. */ 
class FullyConnected: public Layer
{
    private: 
        bool useBias;
        bool requiresGrad;
        int in_c;
        int out_c;
    public:
        NSP weights;
        NSP biases;
        /* Constructor determines the weights dimension, 
        then initialize weights */ 
        FullyConnected(int in_channel, int out_channel, bool use_bias = true, bool requires_grad = true);
        /* Batched Forward call (overload) */
        NSP forward(NSP in); 
        /* Get pointers to the weights */
        std::vector<NSP> params() override;
};
}

#endif
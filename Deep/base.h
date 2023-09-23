#ifndef BASE_H
#define BASE_H

namespace Deep
{
/* All layer types should inherited from Layer class. 
Layer is the base class of all layers that have parameters.
Ex: Fully Connected Layer should be derived from Layer,
but MaxPooling should not be. 
We do not put this in a single header file because we 
anticipate its usage in many different codes, such as 
nn.cpp, cnn.cpp, rnn.cpp, and so on.*/
class Layer {};

/* BatchNorm Layer (1D) */ 
// class BatchNorm1D: public Layer
// {
//     private: 
//         bool useBias;
//         bool requiresGrad;
//         int in_c;
//         int out_c;
//     public:
//         NSP weights;
//         NSP biases;
// };
}

#endif
#ifndef NN_H
#define NN_H
#include <Eigen/Dense>
#include <random>

namespace Deep
{
extern std::mt19937 gen;

/* FullyConnected layer will not be using 
tensor, as 2D-matrices are sufficient for 
all operations. */ 
class FullyConnected
{
    private: 
        Eigen::MatrixXd weights;
        Eigen::MatrixXd gradients;
    public:
        int in_c;
        int out_c;
        bool requires_grad;
        /* Constructor determines the weights dimension, 
        then initialize weights */ 
        FullyConnected(const int& in_channel, const int& out_channel, const bool& use_grad = true);
        /* Forward call */
        Eigen::MatrixXd operator()(const Eigen::MatrixXd& in) const; 
};
}


#endif
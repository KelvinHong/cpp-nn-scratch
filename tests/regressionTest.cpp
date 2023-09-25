// Write your test on regression task here.
#include "../Deep/utility.h"
#include "../Deep/base.h"
#include "../Deep/nn.h"
#include <Eigen/Dense>
#include <memory>

#define PTR_LAYER(DERIVED) std::unique_ptr<Deep::Layer>(new DERIVED)

// Node Shared Pointer
using NSP = std::shared_ptr<Deep::Node>;

class MyReg: public Deep::Model
{
    public:
        MyReg()
        {
            // Add layers
            layers["fc1"] = PTR_LAYER(Deep::FullyConnected(11,64));
            layers["fc2"] = PTR_LAYER(Deep::FullyConnected(64,64));
            layers["fc3"] = PTR_LAYER(Deep::FullyConnected(64,32));
            layers["fc4"] = PTR_LAYER(Deep::FullyConnected(32,1));
        }
        NSP forward(NSP in) override
        {
            // Forward pass
            NSP x1 { Deep::relu(layers["fc1"]->forward(in)) };
            NSP x2 { Deep::relu(layers["fc2"]->forward(x1)) };
            NSP x3 { Deep::relu(layers["fc3"]->forward(x2)) };
            NSP y { layers["fc4"]->forward(x3) };
            return y;
        }

};

int main()
{   
    MyReg model {};
    Eigen::MatrixXd data(3,11);
    data.row(0).fill(0.5);
    data.row(1).fill(-0.5);
    data.row(2).fill(1.0);
    Eigen::MatrixXd label(3, 1);
    label << 5, 8, 4;
    NSP labelPtr { std::make_shared<Deep::Node>(label) }; // No gradient required for label.
    NSP xPtr { std::make_shared<Deep::Node>(data, Deep::gradFn::accumulateGrad) };
    NSP yPtr {model.forward(xPtr)};
    NSP LPtr {Deep::MSE(yPtr, labelPtr)};
    int count { LPtr->descendents() };
    std::cout << "This tree has " << count << " nodes.\n";


    return 0;
}
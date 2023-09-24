// Write your test on regression task here.
#include "../Deep/base.h"
#include "../Deep/nn.h"
#include <Eigen/Dense>
#include <type_traits>


class MyReg: public Deep::Model
{
    public:
        MyReg()
        {
            // Add layers
            layers["fc1"] = Deep::FullyConnected(11, 64);
            layers["fc2"] = Deep::FullyConnected(64, 64);
            layers["fc3"] = Deep::FullyConnected(64, 32);
            layers["fc4"] = Deep::FullyConnected(32, 1);
        }
};

int main()
{   
    MyReg model {};
    for (auto param: model.parameters())
    {
        std::cout << param << '\n';
    }
    return 0;
}
#include "Deep/master.h"
#include <Eigen/Dense>
#include <iostream>
#include <memory>  

// Node Shared Pointer
using NSP = std::shared_ptr<Deep::Node>;

class StandardModel: public Deep::Model
{
public:
    StandardModel()
    {
        // Add layers
        layers["fc1"] = std::unique_ptr<Deep::Layer>(new Deep::FullyConnected(11,64));
        layers["fc2"] = std::unique_ptr<Deep::Layer>(new Deep::FullyConnected(64,64));
        layers["fc3"] = std::unique_ptr<Deep::Layer>(new Deep::FullyConnected(64,32));
        layers["fc4"] = std::unique_ptr<Deep::Layer>(new Deep::FullyConnected(32,1));
    }
    NSP forward(NSP in) override
    {
        // in->data has shape [B, 11]
        // Forward pass
        NSP x1 { Deep::relu(layers["fc1"]->forward(in)) };
        NSP x2 { Deep::relu(layers["fc2"]->forward(x1)) };
        NSP x3 { Deep::relu(layers["fc3"]->forward(x2)) };
        NSP y { layers["fc4"]->forward(x3) };
        return y; // y->data has shape [B, 1]
    }
};

void demonstration1()
{
    /* Demonstrate Model usage as like PyTorch model building,
    The model has a mostly linear backward graph: 
    All children only has at most one parent. */
    // Initialize input batch, batchsize = 5, no. of features = 11.
    Eigen::MatrixXd inp(5,11);
    // Initialize input node without tracking gradient on it.
    // Same as NSP xPtr {std::make_shared<Deep::Node>(inp, Deep::gradFn::none)}; 
    NSP xPtr {std::make_shared<Deep::Node>(inp)}; 

    // Initialize model
    StandardModel model {};
    // Get Prediction with shape [5, 1] since batch size is 5.
    NSP yPtr {model.forward(xPtr)};
    // Create a dummy ground truth
    Eigen::MatrixXd gt(5,1);
    gt << 1,0,1,2,1;
    // Aggregate the MSE loss into another pointer.
    NSP LPtr {Deep::MSE(yPtr, gt)};

    // View backward graph on CMD
    int numOfDescendents { LPtr->descendents(true) };
    std::cout << "The Directed Acyclic Graph (DAG) of this model has " << numOfDescendents << " nodes.\n";
    // Generate backward graph in SVG format.
    std::string svgPath { "generate1.svg" };
    LPtr->visualizeGraph(svgPath);
    std::cout << "View generated graph of this model in " << svgPath << ".\n";
}

void demonstration2()
{
    /* Demonstrate non-trivial backward graph relation.
    Some children have more than one parent. */
    // Create a dummy input node, 
    Eigen::MatrixXd x(2,3);
    x.fill(1.5);
    // Create weight
    Eigen::MatrixXd w(3,3);
    w.fill(0.5);
    // Input not tracking gradient
    NSP xPtr {std::make_shared<Deep::Node>(x)};
    // Tracking weight's gradient 
    NSP wPtr {std::make_shared<Deep::Node>(w, Deep::gradFn::accumulateGrad)};
    NSP LPtr {Deep::sum(Deep::relu(xPtr*wPtr) + xPtr)};
    int numOfDescendents { LPtr->descendents(true) };
    std::cout << "The Directed Acyclic Graph (DAG) has " << numOfDescendents << " nodes.\n";
    // Generate backward graph in SVG format.
    std::string svgPath { "generate2.svg" };
    LPtr->visualizeGraph(svgPath);
    std::cout << "View generated graph in " << svgPath << ".\n";
}

int main()
{
    demonstration1();
    demonstration2();
    return 0;
}

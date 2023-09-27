// Write your test on regression task here.
#include "../Deep/utility.h"
#include "../Deep/base.h"
#include "../Deep/nn.h"
#include "../Deep/optimizer.h"
#include "rapidcsv.h"
#include <Eigen/Dense>
#include <sstream>
#include <fstream>
#include <memory>
#include <numeric>
#include <algorithm>

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

std::vector<std::string> split(std::string s, char sep = ',')
{
    std::stringstream ss( s );
    std::vector<std::string> result {};

    while( ss.good() )
    {
        std::string substr;
        getline( ss, substr, sep );
        result.push_back( substr );
    }
    return result;
}   

std::vector<Eigen::MatrixXd> loadData(std::string filepath)
{
    // Count number of rows
    int rowCount { 0 };
    std::ifstream file;
    file.open(filepath);
    std::string line;
    getline(file, line); // ignore header
    while (getline(file, line))
        ++rowCount;
    file.close();
    
    Eigen::MatrixXd feature(rowCount, 11);
    Eigen::MatrixXd label(rowCount, 1);
    file.open(filepath);
    getline(file, line); // ignore header
    for (int i=0; i<rowCount; ++i)
    {
        getline(file, line);
        std::vector<std::string> tokens {split(line)};
        for (int j=1; j<12; ++j)
        {
            feature(i, j-1) = std::stod(tokens[j]);
        }
        label(i, 0) = std::stod(tokens[12]);
    }
    file.close();

    return {feature, label};
}

Eigen::MatrixXd postProcess(Eigen::MatrixXd rawPrediction)
{
    // Convert double outputs into integer output.
    // But technically it is still double. 
    return rawPrediction.unaryExpr([](const double& x){
        return std::round(x);
    });
}

Eigen::MatrixXd postProcess(NSP rawPrediction)
{
    // Convert double outputs into integer output.
    // But technically it is still double. 
    return postProcess(rawPrediction->data);
}

class DataLoader1D 
{
    public:
        std::vector<Eigen::MatrixXd> allData;
        const int batchSize;
        const bool shuffle; 
        const int length;
        int counter;
        std::vector<int> perm;
        DataLoader1D(std::vector<Eigen::MatrixXd> dataset, int batchsize = 1, bool shuf = true):
            allData(dataset), batchSize(batchsize), shuffle(shuf), 
            length(static_cast<int>(dataset[0].rows())), counter(0),
            perm(std::vector<int>(length))
        {
            std::iota(perm.begin(), perm.end(), 0);
            std::shuffle(perm.begin(), perm.end(), Deep::gen);
        };

        std::vector<Eigen::MatrixXd> nextBatch()
        {
            int initial = counter*batchSize;
            std::vector<int> indices(
                perm.begin()+initial, 
                perm.begin()+std::min(initial+batchSize, length)
            );
            ++counter;

            // Perform indexing            
            std::vector<Eigen::MatrixXd> ret {};
            for (Eigen::MatrixXd thisData: allData)
            {
                ret.push_back(thisData(indices, Eigen::all));
            }
            return ret;  
        }

        bool hasNext()
        {
            return (counter * batchSize) < length;
        }

        void reInitialize()
        {
            // Refresh the dataloader into its initial state, ready to resample again.
            std::shuffle(perm.begin(), perm.end(), Deep::gen);
            counter = 0;
        }
        
};

int testForward()
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
    Deep::Optim::SGD optimizer(model.namedParameters());
    assert(count == 22);

    Eigen::MatrixXd prediction { postProcess(yPtr) };
    return 0;
}

int train(std::vector<Eigen::MatrixXd> dataset)
{
    int epochs { 50 };
    
    MyReg model {};
    Deep::Optim::SGD optimizer(model.namedParameters());
    DataLoader1D dl(dataset, 64, true);

    for (int epoch = 1; epoch <= epochs; ++epoch)
    {
        std::cout << "Training on epoch " << epoch << "...\n";
        double running_loss {0.0};
        while (dl.hasNext())
        {
            /* TODO implement zeroGrad for optimizer. */
            std::vector<Eigen::MatrixXd> thisBatch {dl.nextBatch()};
            Eigen::MatrixXd trainFeature { thisBatch[0] };
            Eigen::MatrixXd trainLabel { thisBatch[1] };
            NSP trainNode { std::make_shared<Deep::Node>(trainFeature) };
            NSP yPtr { model.forward(trainNode) };
            NSP LPtr { Deep::MSE(yPtr, trainLabel) } ;
            LPtr->backward();


        }
    }
    return 0;
}

int main()
{   
    testForward();

    std::vector<Eigen::MatrixXd> dataset{loadData("./datasets/winequality/winequality-white-train.csv")};
    train(dataset);
    return 0;
}
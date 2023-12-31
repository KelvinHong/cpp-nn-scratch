#include "node.h"
#include <Eigen/Dense>

namespace Deep
{

std::shared_ptr<Node> operator*(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
{
    assert(a->data.cols() == b->data.rows());
    std::shared_ptr<Node> matmulPtr(
        std::make_shared<Node>(
            a->data * b->data, 
            false, 
            std::vector<std::shared_ptr<Node>> {a, b}, 
            Deep::gradFn::matMulBackward
        )
    );
    
    return matmulPtr;
}

std::shared_ptr<Node> operator+(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
{
    assert(a->data.rows() == b->data.rows());
    assert(a->data.cols() == b->data.cols());
    std::shared_ptr<Node> addPtr(
        std::make_shared<Node>(
            a->data + b->data, 
            false, 
            std::vector<std::shared_ptr<Node>> {a, b}, 
            Deep::gradFn::addBackward
        )
    );
    return addPtr;
}

std::shared_ptr<Node> operator-(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
{
    assert(a->data.rows() == b->data.rows());
    assert(a->data.cols() == b->data.cols());
    std::shared_ptr<Node> subPtr(
        std::make_shared<Node>(
            a->data - b->data, 
            false, 
            std::vector<std::shared_ptr<Node>> {a, b}, 
            Deep::gradFn::subtractBackward
        )
    );
    return subPtr;
}

std::shared_ptr<Node> transpose(std::shared_ptr<Node> a)
{
    return a->transpose();
}

std::shared_ptr<Node> relu(std::shared_ptr<Node> a)
{
    return a->relu();
}
std::shared_ptr<Node> sum(std::shared_ptr<Node> a)
{
    return a->sum();
}

std::shared_ptr<Node> affine(std::shared_ptr<Node> b, std::shared_ptr<Node> x, std::shared_ptr<Node> W)
{
    /* Some routine dimensional checks */
    assert(x->data.cols() == W->data.rows());
    assert(W->data.cols() == b->data.rows());
    assert(b->data.cols() == 1);

    Eigen::MatrixXd newData {x->data * W->data};
    newData.rowwise() += Eigen::RowVectorXd(b->data.col(0));
    std::shared_ptr<Node> retPtr {std::make_shared<Node>(
        newData,
        false,
        std::vector<std::shared_ptr<Node>> {b,x,W},
        Deep::gradFn::addMmBackward
    )};
    return retPtr;
}

std::shared_ptr<Node> MSE(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
{
    assert(a->data.rows() == b->data.rows());
    assert(a->data.cols() == b->data.cols());
    Eigen::MatrixXd diffSquare { a->data - b->data };
    diffSquare = diffSquare.unaryExpr([](double num){
        return num * num;
    });
    Eigen::MatrixXd result(1,1);
    result << diffSquare.mean();
    std::shared_ptr<Node> retPtr {std::make_shared<Node>(
        result,
        false,
        std::vector<std::shared_ptr<Node>> {a,b},
        Deep::gradFn::mseBackward
    )};
    return retPtr;
}

std::shared_ptr<Node> MSE(std::shared_ptr<Node> a, Eigen::MatrixXd b)
{
    std::shared_ptr<Node> bPtr {std::make_shared<Node>(b, Deep::gradFn::none)};
    return MSE(a, bPtr);
}

}
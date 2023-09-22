#include <Eigen/Dense>
#include <iostream>
#include <memory>  //declarations of unique_ptr
// #include <string>

int main()
{
    Eigen::MatrixXd x(2,2);
    x << 1,2,3,5;
    std::cout << x.colwise().sum() << '\n';
    std::cout << x.colwise().sum().rows() << '\n';
    std::cout << x.colwise().sum().cols() << '\n';
    // Eigen::MatrixXd w(2,2);
    // w << 0,1,1,5;
    // Eigen::MatrixXd b(2,1);
    // b << -10,10;
    // Eigen::MatrixXd data = x*w;
    // data.rowwise() += Eigen::RowVectorXd(b.col(0));
    // std::cout << data << '\n';
    return 0;
}

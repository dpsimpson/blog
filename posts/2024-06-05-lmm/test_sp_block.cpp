#include <iostream>
#include "sp_block.hpp"
#include "Eigen/SparseCore"
#include "Eigen/Dense"


int main() {
    std::cout << "-----------matrix test---------" << std::endl;
    double values[] = { 1., 2., 3., 4. };
    int inner[] = { 4, 3, 2, 1 }; // nonzero row indices
    int outer[] = { 0, 1, 2, 3, 4, 4 }; // start index per column + 1 for last col
    Eigen::SparseMatrix<double> A = Eigen::SparseMatrix<double>::Map(
        5 /*rows*/, 5 /*cols*/, 4 /*nonzeros*/, outer, inner, values);
    
    std::cout << Eigen::MatrixXd(A) << std::endl << std::endl;

    Eigen::Matrix<double, 2, 5> B;
    B << 1.,2.,3.,4.,5.,1.,2.,3.,4.,5.;
    std::cout << B << std::endl << std::endl;
    Eigen::Matrix<double, 2, 2> C;
    C << 1.,1.,1.,1.;
    std::cout << C << std::endl << std::endl;
    
    std::cout << "   -------ans-------" << std::endl;
    Eigen::SparseMatrix<double> D = 
        stan::math::Block_sparse_lower<decltype(A), decltype(B),decltype(C)>(A, B, C)();
    std::cout << Eigen::MatrixXd(D) << std::endl << std::endl;

    std::cout << "-----------to_ref test---------" << std::endl;
    Eigen::SparseMatrix<double> E = A * A;
    std::cout << Eigen::MatrixXd(A) << std::endl << std::endl;
    std::cout << "   -------ans-------" << std::endl;
    Eigen::SparseMatrix<double> F = 
        stan::math::Block_sparse_lower<decltype(E), decltype(B),decltype(C)>(A, B, C)();
    std::cout << Eigen::MatrixXd(F) << std::endl << std::endl;

}


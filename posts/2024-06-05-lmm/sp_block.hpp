#include <stan/math/prim/meta/is_eigen_sparse_base.hpp>
#include <stan/math/prim/meta/is_eigen.hpp>
#include <stan/math/prim/meta/is_stan_scalar.hpp>
#include <stan/math/prim/meta/base_type.hpp>
#include <stan/math/prim/err/check_size_match.hpp>
#include <stan/math/prim/fun/to_ref.hpp>
#include <Eigen/SparseCore>

namespace stan {
namespace math {

typedef Eigen::SparseMatrix<double>::StorageIndex StorageIndex;

// The require_ statements are defined in the first #include
template<typename SpMat, typename EigMat1, typename EigMat2, 
require_eigen_sparse_base_t<SpMat>* = nullptr,
require_all_eigen_t<EigMat1, EigMat2>* = nullptr,
require_all_stan_scalar_t<base_type_t<SpMat>,
                          base_type_t<EigMat1>,
                          base_type_t<EigMat2>>* = nullptr>  
class Block_sparse_lower {
    /* 
    A RAII functor class because Jesus hates memory leaks
    Make this encapsulate the whole thing.
    You may be asking why I'm using arrays and pointers
    like I'm writing in C, and the answer is 
    "that's the interface to Map". The dream of the 
    C-90 is alive and well in the eigen code base.

    Anyway, `operator ()` returns a sparseMatrixMap
    */
    using T = typename base_type<SpMat>::type;
   
    StorageIndex* m_outer;
    StorageIndex* m_inner;
    T* m_val;
    StorageIndex m_cols;
    StorageIndex m_nnz;

    public:

    Block_sparse_lower(const SpMat top_left, const EigMat1 bottom_left, const EigMat2 bottom_right) 
    {
        // only eval once
        const auto& tl_ref = to_ref(top_left);
        const auto& bl_ref = to_ref(bottom_left);
        const auto& br_ref = to_ref(bottom_right);

        // Get sizes.
        // NB tmp_nnz is an upper bound. Will only be correct if `top_left` is lower 
        // triangular. We will compute the real value on the fly.
        const StorageIndex ncols_tl = tl_ref.cols();
        const StorageIndex ncols_br = br_ref.cols();
        const StorageIndex tmp_nnz = (tl_ref.nonZeros() + ncols_tl * ncols_br 
                                        + (ncols_br + 1) * ncols_br / 2);

        // check sizes
        check_size_match("Block_sparse_lower", "Columns of ", "top_left ", tl_ref.cols(), "Columns of ", "Bottom Left", bl_ref.cols());
        check_size_match("Block_sparse_lower", "Rows of ", "bottom-left ", bl_ref.rows(), "Rows of ", "Bottom-right", br_ref.rows());
        
        // Allocate!
        m_cols = ncols_tl + ncols_br;

        m_outer = new StorageIndex[m_cols + 1];
        m_outer[0] = *top_left.outerIndexPtr();
        m_inner = new StorageIndex[tmp_nnz];
        m_val = new T[tmp_nnz];
        
        double* p_val = m_val;
        StorageIndex* p_inner = m_inner;
        StorageIndex out_nnz = 0;
        
        for (StorageIndex j = 0; j < ncols_tl; ++j) {
            StorageIndex col_cnt = 0;
            for (typename SpMat::InnerIterator it(tl_ref, j); it; ++it) {
                if (it.row() < j) continue; // lower triangle only
                *p_val++ = it.value();
                *p_inner++ = it.row();
                ++out_nnz;
                ++col_cnt;
            }

            for (StorageIndex i = 0; i < ncols_br; ++i) {
                *p_val++ = bl_ref.coeff(i, j);
                *p_inner++ = ncols_tl + i;
                ++out_nnz;
                ++col_cnt;
            }
        
            m_outer[j+1] = m_outer[j] + col_cnt;
        }
        
        for (StorageIndex j = 0; j < ncols_br; ++j) {
            // only need lower triangle
            for (StorageIndex i = j; i < ncols_br; ++i) {
                *p_val++ = br_ref.coeff(i,j);
                *p_inner++ = ncols_tl + i;
                ++out_nnz;
            }
            m_outer[ncols_tl+j+1] = m_outer[ncols_tl + j] + ncols_br - j;
        }
        m_nnz = out_nnz;
    } // constructor

    ~Block_sparse_lower() {
        delete[] m_outer;
        delete[] m_inner;
        delete[] m_val;
    } // destructor

    Eigen::SparseMatrix<T> operator () () {
        return typename Eigen::SparseMatrix<T>::Map(
            m_cols, 
            m_cols,
            m_nnz,
            m_outer,
            m_inner,
            m_val
        );   
    } //operator ()
}; // Block_sparse_lower
} // namespace math
} // namespace stan
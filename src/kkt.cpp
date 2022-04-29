#include "kkt.h"

using namespace Eigen;
using SparseMatrixdRowMajor = SparseMatrix<double,RowMajor>;

void mfem::kkt_lhs(const SparseMatrixd& M, const SparseMatrixdRowMajor& Jw,
    double ih2, std::vector<Triplet<double>>& trips) {

  trips.clear();

  // Mass matrix terms
  for (int k=0; k < M.outerSize(); ++k) {
    for (SparseMatrixd::InnerIterator it(M,k); it; ++it) {
      trips.push_back(Triplet<double>(it.row(),it.col(),ih2*it.value()));
    }
  }

  int offset = M.rows(); // offset for off diagonal blocks

  // Jacobian off-diagonal entries
  for (int k=0; k < Jw.outerSize(); ++k) {
    for (SparseMatrixdRowMajor::InnerIterator it(Jw,k); it; ++it) {
      trips.push_back(Triplet<double>(offset+it.row(),it.col(),it.value()));
      trips.push_back(Triplet<double>(it.col(),offset+it.row(),it.value()));
    }
  }
}

void mfem::diagonal_compliance(const Eigen::VectorXd& vols, double mu, int offset,
    std::vector<Eigen::Triplet<double>>& trips) {
  
  // Diagonal compliance block
  int N = vols.size();
  for (int i = 0; i < N; ++i) {

    double H = -vols(i)/mu;
    for (int j = 0; j < 9; ++j) {
      trips.push_back(Eigen::Triplet<double>(offset+9*i+j,offset+9*i+j, H));
    }
  }
}


void mfem::init_compliance_blocks(int N, int offset,
    std::vector<Eigen::Triplet<double>>& trips) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < 9; ++j) {
      for (int k = 0; k < 9; ++k) {
        trips.push_back(Triplet<double>(offset+9*i+j,offset+9*i+k, 0.));
      }
    }
  }
}

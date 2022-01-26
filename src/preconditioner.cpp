#include "preconditioner.h"

using namespace Eigen;

void diag_compliance(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
    const Eigen::VectorXd& vols, double alpha,
    std::vector<Eigen::Triplet<double>>& trips) {

  int offset = V.size();
  for (int i = 0; i < T.rows(); ++i) {
    double He = -vols(i)/alpha;
    for (int j = 0; j < 9; ++j) {
      trips.push_back(Eigen::Triplet<double>(offset+9*i+j,offset+9*i+j, He));
    }
  }
}

#pragma once

#include "optimizers/mixed_optimizer.h"
#include "sparse_utils.h"

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif

#include <Eigen/SVD>
#include <Eigen/QR>

//setup the preconditioner
namespace Eigen {


 template<typename Scalar>
 class CorotatedPreconditioner 
  {
     using Scalar_ = Scalar;
     typedef typename NumTraits<Scalar>::Real RealScalar;
  
     public:

       CorotatedPreconditioner() {
         mat_val_ = -1.0;
       }

       CorotatedPreconditioner(const CorotatedPreconditioner<Scalar> &cr) : dS_(cr.dS_) {

        nd_ = cr.nd_;
        ne_ = cr.ne_;
        mat_val_ = cr.mat_val_;
        P_ = cr.P_;
        // TODO memory leak?
        preconditioner_ = new Eigen::SimplicialLDLT<Eigen::SparseMatrix<double, Eigen::RowMajor>>();
        preconditioner_->compute(P_);
        rtilde_ = cr.rtilde_;
        ztilde_ = cr.ztilde_;
        z_ = cr.z_;
      
      }

      inline double mat_val() { return mat_val_; }

       template<typename MatrixType>
       explicit CorotatedPreconditioner(double mat_val, int ndofs, int ne, const MatrixType& P, std::vector<Eigen::Matrix<Scalar, 9,6>> &dS)  : dS_(&dS) {
         P_ = P;
         nd_ = ndofs;
         ne_ = ne;
         mat_val_ = mat_val;
         preconditioner_ = new Eigen::SimplicialLDLT<Eigen::SparseMatrix<double, Eigen::RowMajor>>();
         preconditioner_->compute(P_);
       }
    
       template<typename MatrixType>
       CorotatedPreconditioner& analyzePattern(const MatrixType& ) { return *this; }
    
       template<typename MatrixType>
       CorotatedPreconditioner& factorize(const MatrixType& ) { return *this; }
    
       template<typename MatrixType>
       CorotatedPreconditioner& compute(const MatrixType& ) { return *this; }
    
      // dX^T A dX = b
// dX*dX^T A = dX*b
//  (dX x) = A^{-1}(dX*dX^T)^{-1} *dX
//  x = (dX'*dX)^{-1}*dX*A^{-1}(dX*dX^T)^{-1} *dX*b
// dX = USV' (some S's are 0)
//   (dX*dX^T)^{-1} = (V*S^-2*V')
//  (dX'*dX)^{-1} = (U*S^-2*U')

       template<typename Rhs>
       inline const Rhs& solve(const Rhs& b) const { 

            rtilde_.resize(nd_+9*ne_);
            ztilde_.resize(nd_+9*ne_);
            z_.resize(nd_+6*ne_);
            
            rtilde_.segment(0, nd_) = b.segment(0, nd_);
            
            #pragma omp parallel for
            for (int i = 0; i < dS_->size(); ++i) {

              //Eigen::MatrixXd A = (*dS_)[i].transpose();
              //Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> svd(A);
              /*Eigen::Matrix<double, 6,1> svs = svd.singularValues();
              svs[0] = (std::abs(svs[0]) > 1e-8 ? 1./(svs[0]*svs[0]) : 0);
              svs[1] = (std::abs(svs[1]) > 1e-8 ? 1./(svs[1]*svs[1]) : 0);
              svs[2] = (std::abs(svs[2]) > 1e-8 ? 1./(svs[2]*svs[2]) : 0);
              svs[3] = (std::abs(svs[3]) > 1e-8 ? 1./(svs[3]*svs[3]) : 0);
              svs[4] = (std::abs(svs[4]) > 1e-8 ? 1./(svs[4]*svs[4]) : 0);
              svs[5] = (std::abs(svs[5]) > 1e-8 ? 1./(svs[5]*svs[5]) : 0);


              Eigen::Matrix9d dSi = svd.matrixU()*svs.asDiagonal()*svd.matrixU().transpose();

              std::cout<<"U: \n"<<svd.matrixU()<<"\n\n";

              std::cout<<"dSi: \n"<<dSi<<"\n\n";*/

              //rtilde_.template segment<9>(nd_ + 9*i) = svd.solve(b.template segment<6>(nd_ + 6*i));

              rtilde_.template segment<9>(nd_ + 9*i) = (*dS_)[i]*(b.template segment<6>(nd_ + 6*i));
            }

            ztilde_ = preconditioner_->solve(rtilde_);
            z_.segment(0, nd_) = ztilde_.segment(0, nd_);
            
            #pragma omp parallel for
            for (int i = 0; i < dS_->size(); ++i) {
              
              //Eigen::MatrixXd A = (*dS_)[i];
              //Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> svd(A);
              /*Eigen::Matrix<double, 6,1> svs = svd.singularValues();
              svs[0] = (std::abs(svs[0]) > 1e-8 ? 1./(svs[0]*svs[0]) : 0);
              svs[1] = (std::abs(svs[1]) > 1e-8 ? 1./(svs[1]*svs[1]) : 0);
              svs[2] = (std::abs(svs[2]) > 1e-8 ? 1./(svs[2]*svs[2]) : 0);
              svs[3] = (std::abs(svs[3]) > 1e-8 ? 1./(svs[3]*svs[3]) : 0);
              svs[4] = (std::abs(svs[4]) > 1e-8 ? 1./(svs[4]*svs[4]) : 0);
              svs[5] = (std::abs(svs[5]) > 1e-8 ? 1./(svs[5]*svs[5]) : 0);

              Eigen::Matrix6d dSi = svd.matrixV().transpose()*svs.asDiagonal()*svd.matrixV();*/

              z_.template segment<6>(nd_ + 6*i) = (*dS_)[i].transpose()*(ztilde_.segment<9>(nd_ + 9*i));
            }

            return z_;
       }
    
       ComputationInfo info() { return Success; }
    
    protected:
      unsigned int nd_, ne_;
      SimplicialLDLT<Eigen::SparseMatrix<double, Eigen::RowMajor>> *preconditioner_;
      Eigen::SparseMatrix<double, Eigen::RowMajor> P_;
      mutable VectorXd rtilde_;
      mutable VectorXd ztilde_;
      mutable VectorXd z_;
      std::vector<Matrix<Scalar, 9,6>> *dS_;
      double mat_val_;

     }; 
  } 

namespace mfem {

  // Mixed FEM Sequential Quadratic Program
  class MixedSQPOptimizer : public MixedOptimizer {
  public:
    
    MixedSQPOptimizer(std::shared_ptr<Mesh> mesh,
        std::shared_ptr<SimConfig> config) : MixedOptimizer(mesh, config) {}

    static std::string name() {
      return "SQP";
    }

    void reset() override;
  
  public:

    virtual double energy(const Eigen::VectorXd& x, const Eigen::VectorXd& s,
        const Eigen::VectorXd& la) override;

    virtual void gradient(Eigen::VectorXd& g, const Eigen::VectorXd& x,
        const Eigen::VectorXd& s, const Eigen::VectorXd& la) override;

    virtual void build_lhs() override;

    virtual void build_rhs() override;

    virtual void update_system() override;

    virtual void substep(int step, double& decrement) override;

    virtual void setup_preconditioner();

    Eigen::SparseMatrixd W_;
    Eigen::SparseMatrixd G_;
    Eigen::SparseMatrixd C_;
    Eigen::SparseMatrixd Gx_;
    Eigen::SparseMatrixd Gx0_;
    Eigen::SparseMatrixd Gs_;
    Eigen::SparseMatrix<double, Eigen::RowMajor> PJ_; // integrated (weighted) jacobian
    Eigen::SparseMatrix<double, Eigen::RowMajor> PM_;
    Eigen::SparseMatrix<double, Eigen::RowMajor> Jw_; // integrated (weighted) jacobian

    // CG temp variables
    Eigen::VectorXd tmp_r_;
    Eigen::VectorXd tmp_z_;
    Eigen::VectorXd tmp_zm1_;
    Eigen::VectorXd tmp_p_;
    Eigen::VectorXd tmp_Ap_;

    std::vector<Eigen::Matrix3f> U_;
    std::vector<Eigen::Matrix3f> V_;
    std::vector<Eigen::Vector3f> sigma_;
    std::vector<Eigen::MatrixXd> Jloc_;
    std::shared_ptr<Assembler<double,3>> assembler_;
    std::shared_ptr<VecAssembler<double,3>> vec_assembler_;

    // // Solve used for preconditioner
    // #if defined(SIM_USE_CHOLMOD)
    // Eigen::CholmodSimplicialLDLT<Eigen::SparseMatrixd> solver_;
    // #else
    // Eigen::SimplicialLLT<Eigen::SparseMatrixd> solver_;
    // #endif
    Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::RowMajor>> solver_;

    Eigen::CorotatedPreconditioner<double> preconditioner_;
    //Eigen::SimplicialLDLT<Eigen::SparseMatrixd> preconditioner_;
    //Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper, Eigen::CorotatedPreconditioner<double>> cg;
  };
}

#pragma once

#include "optimizers/optimizer.h"

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

        preconditioner_ = new Eigen::SimplicialLDLT<SparseMatrixd>();
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
         preconditioner_ = new Eigen::SimplicialLDLT<SparseMatrixd>();
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
      SimplicialLDLT<SparseMatrixd> *preconditioner_;
      SparseMatrixd P_;
      mutable VectorXd rtilde_;
      mutable VectorXd ztilde_;
      mutable VectorXd z_;
      std::vector<Matrix<Scalar, 9,6>> *dS_;
      double mat_val_;

     }; 
  } 

namespace mfem {

  // Mixed FEM Sequential Quadratic Program
  class MixedSQPOptimizer : public Optimizer {
  public:
    
    MixedSQPOptimizer(std::shared_ptr<SimObject> object,
        std::shared_ptr<SimConfig> config) : Optimizer(object, config) {}

    void reset() override;
    void step() override;
  
  public:

    // Evaluated augmented lagrangian energy
    virtual double energy(const Eigen::VectorXd& x, const Eigen::VectorXd& s,
        const Eigen::VectorXd& la);

    // Build system left hand side
    virtual void build_lhs();

    // Build linear system right hand side
    virtual void build_rhs();

    // For a new set of positions, update the rotations and their
    // derivatives
    virtual void update_rotations();

    // Update gradients, LHS, RHS for a new configuration
    virtual void update_system();

    // Linesearch over positions
    // x  - initial positions. Output of linesearch updates this variable
    // dx - direction we perform linesearch on
    virtual bool linesearch_x(Eigen::VectorXd& x, const Eigen::VectorXd& dx);
    virtual bool linesearch_s(Eigen::VectorXd& s, const Eigen::VectorXd& ds);
    virtual bool linesearch(Eigen::VectorXd& x, const Eigen::VectorXd& dx,
        Eigen::VectorXd& s, const Eigen::VectorXd& ds);

    virtual void setup_preconditioner();

    // Simulation substep for this object
    // init_guess - whether to initialize guess with a prefactor solve
    // decrement  - newton decrement norm
    virtual void substep(bool init_guess, double& decrement);

    // At the end of the timestep, update position, velocity variables,
    // and reset lambda & kappa.
    virtual void update_configuration();

    // Configuration vectors & body forces
    Eigen::VectorXd q_;
    Eigen::VectorXd x_;        // current positions
    Eigen::VectorXd vt_;        // current velocities
    Eigen::VectorXd x0_;        // previous positions
    Eigen::VectorXd dx_;        // current update
    Eigen::VectorXd f_ext_;     // per-node external forces
    Eigen::VectorXd la_;        // lambdas
    Eigen::VectorXd ds_;        // deformation updates
    Eigen::VectorXd s_;         // deformation variables
    Eigen::VectorXd b_;         // coordinates projected out
    Eigen::VectorXd vols_;      // per element volume
    Eigen::VectorXd rhs_;       // linear system right hand side
    Eigen::SparseMatrixd lhs_;  // linear system left hand side

    std::vector<Eigen::Matrix3d> R_;  // Per-element rotations
    std::vector<Eigen::Vector6d> S_;    // Per-element deformation
    std::vector<Eigen::Matrix6d> H_; // Elemental hessians w.r.t dS
    std::vector<Eigen::Matrix6d> Hinv_; // Elemental hessians w.r.t dS
    std::vector<Eigen::Vector6d> g_;    // Elemental gradients w.r.t dS
    std::vector<Eigen::Matrix<double,9,6>> dS_;

    Eigen::SparseMatrixd M_;        // mass matrix
    Eigen::SparseMatrixd P_;        // pinning constraint (for vertices)
    SparseMatrixdRowMajor J_;       // jacobian
    SparseMatrixdRowMajor Jw_;      // integrated (weighted) jacobian
    Eigen::SparseMatrixd J2_;
    Eigen::SparseMatrixd J_tilde_;
    SparseMatrixdRowMajor Ws_;      // integrated (weighted) jacobian
    Eigen::SparseMatrixd W_;
    Eigen::SparseMatrixd G_;
    Eigen::SparseMatrixd D_;
    Eigen::SparseMatrixd L_;
    Eigen::SparseMatrixd C_;
    Eigen::SparseMatrixd Gx_;
    Eigen::SparseMatrixd Gx0_;
    Eigen::SparseMatrixd Gs_;
    Eigen::SparseMatrixd Hx_;
    Eigen::SparseMatrixd MinvG_;
    Eigen::SparseMatrixd Minv_;
    Eigen::VectorXd gx_;
    Eigen::VectorXd gs_;

    // CG temp variables
    Eigen::VectorXd tmp_r_;
    Eigen::VectorXd tmp_z_;
    Eigen::VectorXd tmp_p_;
    Eigen::VectorXd tmp_Ap_;

    // // Solve used for preconditioner
    // #if defined(SIM_USE_CHOLMOD)
    // Eigen::CholmodSimplicialLDLT<Eigen::SparseMatrixd> solver_;
    // #else
    // Eigen::SimplicialLLT<Eigen::SparseMatrixd> solver_;
    // #endif
    Eigen::SparseLU<Eigen::SparseMatrixd> solver_;

    Eigen::CorotatedPreconditioner<double> preconditioner_;
    //Eigen::SimplicialLDLT<Eigen::SparseMatrixd> preconditioner_;
    //Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper, Eigen::CorotatedPreconditioner<double>> cg;

    int nelem_;     // number of elements
    double E_prev_; // energy from last result of linesearch
    
    Eigen::VectorXi pinnedV_;
  };
}

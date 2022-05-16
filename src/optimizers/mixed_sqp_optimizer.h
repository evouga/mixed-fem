#pragma once

#include "optimizers/optimizer.h"

#if defined(SIM_USE_CHOLMOD)
#include <Eigen/CholmodSupport>
#endif

#include <Eigen/SVD>
#include <Eigen/QR>
#include <unsupported/Eigen/IterativeSolvers>

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
       explicit CorotatedPreconditioner(double mat_val, int ndofs, int ne, const MatrixType& P, std::vector<Eigen::Matrix<Scalar, 3,3>> &dS)  : dS_(&dS) {
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
       inline const Eigen::VectorXd & solve(const Rhs& b) const { 

            rtilde_.resize(nd_+9*ne_);
            ztilde_.resize(nd_+9*ne_);
            z_.resize(nd_+6*ne_);
            
            rtilde_.segment(0, nd_) = b.segment(0, nd_);
            
            #pragma omp parallel for
            for (int i = 0; i < dS_->size(); ++i) {

              //Eigen::MatrixXd A = (*dS_)[i].transpose();
              //Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> svd(A);
            

              //rtilde_.template segment<9>(nd_ + 9*i) = svd.solve(b.template segment<6>(nd_ + 6*i));

              //rtilde_.template segment<9>(nd_ + 9*i) = (*dS_)[i]*(b.template segment<6>(nd_ + 6*i));

              Eigen::Vector6d rl = b.template segment<6>(nd_ + 6*i);
              Eigen::Matrix3d RL;
              
              RL << rl(0), 0.5*rl(3), 0.5*rl(4),
                    0.5*rl(3), rl(1), 0.5*rl(5),
                    0.5*rl(4),0.5*rl(5), rl(2);

              RL = (*dS_)[i]*RL;
              rtilde_.template segment<9>(nd_ + 9*i) = Eigen::Matrix<double, 9,1>(RL.data());

            }

            ztilde_ = preconditioner_->solve(rtilde_);
            z_.segment(0, nd_) = ztilde_.segment(0, nd_);
            
            //update rotations
            /*#pragma omp parallel for 
            for (int i = 0; i < nelem_; ++i) {
              JacobiSVD<Matrix3d> svd(Map<Matrix3d>(ztilde_.segment(9*i,9).data()),
                  ComputeFullU | ComputeFullV);

              Eigen::Vector3d stemp;

              stemp[0] = 1;
              stemp[1] = 1;
              stemp[2] = (svd.matrixU()*svd.matrixV().transpose()).determinant();
              // S(x^k)
              Matrix3d S = svd.matrixV() * svd.singularValues().asDiagonal() 
                  * svd.matrixV().transpose();
              Vector6d stmp; stmp << S(0,0), S(1,1), S(2,2), S(1,0), S(2,0), S(2,1);
              S_[i] = stmp;
              R_[i] = svd.matrixU() * stemp.asDiagonal()*svd.matrixV().transpose();

              // Compute SVD derivatives
              Tensor3333d dU, dV;
              Tensor333d dS;
              dsvd(dU, dS, dV, Map<Matrix3d>(def_grad.segment<9>(9*i).data()));

              // Compute dS/dF
              S = svd.singularValues().asDiagonal();
              Matrix3d V = svd.matrixV();
              std::array<Matrix3d, 9> dS_dF;
              for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                  dS_dF[3*c + r] = dV[r][c]*S*V.transpose() + V*dS[r][c].asDiagonal()*V.transpose()
                      + V*S*dV[r][c].transpose();
                }
              }

              // Final jacobian should just be 6x9 since S is symmetric,
              // so extract the approach entries
              Matrix<double, 9, 9> J;
              for (int i = 0; i < 9; ++i) {
                J.col(i) = Vector9d(dS_dF[i].data());
              }
              Matrix<double, 6, 9> Js;
              Js.row(0) = J.row(0);
              Js.row(1) = J.row(4);
              Js.row(2) = J.row(8);
              Js.row(3) = J.row(1);
              Js.row(4) = J.row(2);
              Js.row(5) = J.row(5);
              dS_[i] = Js.transpose() * Sym;
            }*/

            #pragma omp parallel for
            for (int i = 0; i < dS_->size(); ++i) {
              
              //Eigen::MatrixXd A = (*dS_)[i];
              //Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> svd(A);
            

              //z_.template segment<6>(nd_ + 6*i) = (*dS_)[i].transpose()*(ztilde_.segment<9>(nd_ + 9*i));
              //z_.template segment<6>(nd_ + 6*i) = svd.solve(ztilde_.segment<9>(nd_ + 9*i));

              Eigen::Vector9d rl = ztilde_.segment<9>(nd_ + 9*i);
              
               Eigen::Matrix3d RL;
              
              RL << rl(0), rl(3), rl(6),
                    rl(1), rl(4),  rl(7),
                    rl(2), rl(5), rl(8);

              RL = (*dS_)[i].transpose()*RL;
              Eigen::Vector6d S;
              S << RL(0,0), RL(1,1), RL(2,2), (RL(1,0)+RL(0,1)), (RL(2,0)+RL(0,2)), (RL(2,1)+RL(1,2));
              z_.template segment<6>(nd_ + 6*i) = S;

            }

            //VectorXd def_grad = J_*(P_.transpose()*x_+b_);

            /*#pragma omp parallel for 
            for (int i = 0; i < nelem_; ++i) {
              JacobiSVD<Matrix3d> svd(Map<Matrix3d>(def_grad.segment(9*i,9).data()),
                  ComputeFullU | ComputeFullV);

              Eigen::Vector3d stemp;

              stemp[0] = 1;
              stemp[1] = 1;
              stemp[2] = (svd.matrixU()*svd.matrixV().transpose()).determinant();
              
              R_[i] = svd.matrixU() *svd.matrixV().transpose();

            }*/
              
            //z_ = preconditioner_->solve(b);
             /*VectorXd def_grad = J_*(x_+ z_.segment(0, nd_));

            #pragma omp parallel for 
            for (int i = 0; i < (*dS_).size(); ++i) {
              Vector9d thing = def_grad.segment(9*i,9) + ztilde_.segment(nd_+ 9*i, 9)/ztilde_.segment(nd_ + 9*i, 9).cwiseAbs().maxCoeff();
              JacobiSVD<Matrix3d> svd(Map<Matrix3d>(thing.data()), ComputeFullU | ComputeFullV);

              Eigen::Vector3d stemp;

              stemp[0] = 1;
              stemp[1] = 1;
              stemp[2] = 1;
              
              (*dS_)[i] = svd.matrixU() *stemp.asDiagonal()*svd.matrixV().transpose();

            }*/
              
            //z_ = preconditioner_->solve(b);

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
      std::vector<Matrix<Scalar, 3,3>> *dS_;
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
    Eigen::SparseMatrixd MinvC_;
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

    Eigen::SimplicialLDLT<Eigen::SparseMatrixd> solver_M_;

    Eigen::CorotatedPreconditioner<double> preconditioner_;
    //Eigen::SimplicialLDLT<Eigen::SparseMatrixd> preconditioner_;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::CorotatedPreconditioner<double> > cg;

    int nelem_;     // number of elements
    double E_prev_; // energy from last result of linesearch
    
    Eigen::VectorXi pinnedV_;
  };
}

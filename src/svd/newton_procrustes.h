#pragma once

#include <Eigen/Dense>
//solve orthogonal procrustes problem using newton;s method 
//find rotation such that ||R*A - B||F is minimized

static Eigen::Matrix3d cpx = [] { Eigen::Matrix3d tmp;
      tmp << 0.,0., 0., 0.,0., -1., 0.,1., 0.;
      return tmp;
}();

static Eigen::Matrix3d cpy = [] { Eigen::Matrix3d tmp;
      tmp << 0.,0., 1.,
            0.,0., 0.,
            -1.,0., 0.;
      return tmp;
}();

static Eigen::Matrix3d cpz = [] { Eigen::Matrix3d tmp;
      tmp << 0.,-1., 0.,
            1.,0., 0.,
            0.,0., 0.;
      return tmp;
}();

static Eigen::Matrix3d cpxx = [] { Eigen::Matrix3d tmp;
      
      tmp = cpx*cpx;
      return tmp;
}();

static Eigen::Matrix3d cpyy = [] { Eigen::Matrix3d tmp;
      
      tmp = cpy*cpy;
      return tmp;
}();

static Eigen::Matrix3d cpzz = [] { Eigen::Matrix3d tmp;
      
      tmp = cpz*cpz;
      return tmp;
}();

static Eigen::Matrix3d cpxy = [] { Eigen::Matrix3d tmp;
      
      tmp = 0.5*(cpx*cpy + cpy*cpx).array();
      return tmp;
}();

static Eigen::Matrix3d cpxz = [] { Eigen::Matrix3d tmp;
      
      tmp = 0.5*(cpx*cpz + cpz*cpx).array();
      return tmp;
}();

static Eigen::Matrix3d cpyz = [] { Eigen::Matrix3d tmp;
      
      tmp = 0.5*(cpy*cpz + cpz*cpy).array();
      return tmp;
}();

template<typename DerivedMat, typename DerivedVec>
void rodrigues(Eigen::MatrixBase<DerivedMat> &R, const Eigen::MatrixBase<DerivedVec> &omega) {
    
    using Scalar = typename DerivedVec::Scalar; 

    Scalar angle = omega.norm();
    //handle the singularity ... return identity for 0 angle of rotation
    if(std::fabs(angle) < 1e-8) {
        R << 1, 0, 0,
             0, 1, 0,
             0, 0, 1;
        return;
    }

    Eigen::Matrix<Scalar, 3,1> axis = omega.normalized();
    Eigen::Matrix<Scalar, 3,3> K;
    
    K << 0., -axis(2), axis(1),
         axis(2), 0., -axis(0),
         -axis(1), axis(0), 0.;
    R = Eigen::Matrix3d::Identity() + std::sin(angle)*K + (1-std::cos(angle))*K*K;
}

template<typename DerivedR, typename DerivedA, typename DerivedB >
void newton_procrustes(Eigen::MatrixBase<DerivedR> &R,  const Eigen::MatrixBase<DerivedA> &A, const Eigen::MatrixBase<DerivedB> &B, double tol = 1e-8, int max_iter = 10, bool compute_gradients = false) {

    using Scalar = typename DerivedR::Scalar;

    //constant matrices needed for computing gradient and hessian
    Eigen::Matrix<Scalar,3,3> Y = R*A*B.transpose();
    Eigen::Matrix<Scalar, 3,1> g;
    Eigen::Matrix<Scalar, 3,3> H;
    Eigen::Matrix<Scalar, 3,1> omega;
    Eigen::Matrix<Scalar, 3,3> dR;

    Scalar E0, E1;

    //newton loop
    unsigned int itr = 0;
    
    do {

        //compute useful gradients here if needed
        g << -(cpx*Y).trace(), -(cpy*Y).trace(), -(cpz*Y).trace();

        //std::cout<<"GRADIENT: "<<g<<"\n";
        if(g.norm() < tol) {
          //  std::cout<<"Exit Here "<<itr<<"\n";
            //converged to within tolerance
            return;
        }

        H << -(cpxx*Y).trace(), -(cpxy*Y).trace(), -(cpxz*Y).trace(),
              -(cpxy*Y).trace(), -(cpyy*Y).trace(), -(cpyz*Y).trace(),
              -(cpxz*Y).trace(), -(cpyz*Y).trace(), -(cpzz*Y).trace();


        omega = -H.inverse()*g;
        E0 = -(R*Y).trace();
        E1 = E0 + 1.0;

        do {

            rodrigues(dR, omega);
            E1 = -(dR*Y).trace();

            omega.noalias() =  omega*0.6;
            
        }while(E1 > E0 && omega.norm() > tol);
        
        R = dR*R;
        Y = dR*Y;

        ++itr;
        
    }while(itr < max_iter);

    
    
}


/* SCRATCH 
Matrix3 Cs;
    Cs << s(0), s(5), s(4),
          s(5), s(1), s(3),
          s(4), s(3), s(2);
    Matrix3 Y;
    Matrix3 Hr;
    Vector3 gr;
    Matrix3 dR;
    //R[ii].setIdentity();
    Precision E0;
    for(unsigned int jj=0; jj<50; ++jj) {
      //E0 = (R[ii].transpose()*(Eigen::Map<Matrix3>(li.data())*Cs.transpose())).trace();
      Y = R[ii]*Cs*(Eigen::Map<Matrix3>(li.data()) + Eigen::Map<Matrix3>(li2.data()));
      E0 = -Y.trace();
      gr << -(cpx*Y).trace(), -(cpy*Y).trace(), -(cpz*Y).trace();
      Hr << -(cpxx*Y).trace(), -(cpxy*Y).trace(), -(cpxz*Y).trace(),
            -(cpxy*Y).trace(), -(cpyy*Y).trace(), -(cpyz*Y).trace(),
            -(cpxz*Y).trace(), -(cpyz*Y).trace(), -(cpzz*Y).trace();
      Vector3 omega = -Hr.inverse()*gr;
      if(omega.norm() < 1e-6)
        break;
      // std::cout<<"OMEGA "<<omega<<"\n";
      // std::cout<<"F: "<<Eigen::Map<Matrix3>(li.data())<<"\n";
      // std::cout<<"S: "<<Cs<<"\n";
      // std::cout<<"R: "<<R[ii]<<"\n";
      // std::cout<<"DR: "<<dR<<"\n";
      Precision E1 = E0 + 10.;
      //std::cout<<"E1 "<<E1<<"\n";
      while(E1 > E0) {
        rodrigues(dR, omega);
        E1 = -(dR*Y).trace();
        omega.noalias() =  omega*0.8;
        if(omega.norm() < 1e-8) {
          omega.setZero();
          break;
        }
        //std::cout<<"E0 "<<E0<<"E1 "<<E1<<"\n";
      }
      R[ii] = (dR*R[ii]).eval();*/
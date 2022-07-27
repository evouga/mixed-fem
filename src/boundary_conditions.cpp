#include "boundary_conditions.h"
#include <omp.h>
#include "mesh/mesh.h"

using namespace mfem;
using namespace Eigen;

template <int DIM>
void BoundaryConditions<DIM>::init_boundary_groups(const Eigen::MatrixXd &V,
    std::vector<std::vector<int>> &bc_groups, double ratio) {

  // resize to match size
  Eigen::RowVectorXd bottomLeft = V.colwise().minCoeff();
  Eigen::RowVectorXd topRight = V.colwise().maxCoeff();
  Eigen::RowVectorXd range = topRight - bottomLeft;

  bc_groups.resize(2);
  for (int vI = 0; vI < V.rows(); vI++)
  {
    if (V(vI, 0) < bottomLeft[0] + range[0] * ratio)
    {
      bc_groups[0].emplace_back(vI);
    }
    else if (V(vI, 0) > topRight[0] - range[0] * ratio)
    {
      bc_groups[1].emplace_back(vI);
    }
  }
}

// template <int DIM>
// void BoundaryConditions<DIM>::init_boundary_groups(const Eigen::MatrixXd &V,
//     std::vector<std::vector<int>> &bc_groups, double ratio) {

//   // resize to match size
//   Eigen::RowVectorXd bottomLeft = V.colwise().minCoeff();
//   Eigen::RowVectorXd topRight = V.colwise().maxCoeff();
//   Eigen::RowVectorXd range = topRight - bottomLeft;

//   bc_groups.resize(2);
//   for (int vI = 0; vI < V.rows(); vI++)
//   {
//     if (V(vI, 0) < bottomLeft[0] + range[0] * ratio 
//         && V(vI, 1) > topRight[1] - range[1] * ratio )
//     {
//       bc_groups[0].emplace_back(vI);
//     }
//     else if (V(vI, 0) > topRight[0] - range[0] * ratio 
//         && V(vI, 1) > topRight[1] - range[1] * ratio )
//     {
//       bc_groups[1].emplace_back(vI);
//     }
//   }
// }

template <int DIM>
const std::vector<std::string> BoundaryConditions<DIM>::script_type_strings = {
    "null", "scaleF", "hang", "hangends","stretch", "squash", "stretchnsquash",
    "bend", "twist", "twistnstretch", "twistnsns", "twistnsns_old",
    "rubberBandPull", "onepoint", "random", "fall"};

template <int DIM>
BoundaryConditions<DIM>::BoundaryConditions(BCScriptType script_type) : script_type_(script_type)
{
}

template <int DIM>
void BoundaryConditions<DIM>::init_script(std::shared_ptr<Mesh> &mesh)
{

  switch (script_type_)
  {
  case BC_NULL:
    break;

  case BC_SCALEF:
  {
    mesh->clear_fixed_vertices();

    MatD M;
    M << 1.5, 0.0, 0.0,
        0.0, 1.5, 0.0,
        0.0, 0.0, 1.5;
    for (int i = 0; i < mesh->V_.rows(); ++i)
    {
      mesh->V_.row(i) = (M * mesh->V_.row(i).transpose()).transpose();
    }
    break;
  }

  case BC_HANG:
    mesh->clear_fixed_vertices();
    for (const auto borderI : mesh->bc_groups_)
    {
      mesh->set_fixed(borderI.back());
    }
    break;
  case BC_HANGENDS:
  {
    mesh->clear_fixed_vertices();
    bc_groups_.resize(0);

    int bg = 1;
    mesh->set_fixed(mesh->bc_groups_[bg]);
    bc_groups_.emplace_back(mesh->bc_groups_[bg]);
    break;
  }
  case BC_STRETCH:
  {
    mesh->clear_fixed_vertices();
    bc_groups_.resize(0);
    int bI = 0;
    for (const auto borderI : mesh->bc_groups_)
    {
      mesh->set_fixed(borderI);
      bc_groups_.emplace_back(borderI);
      for (const auto bVI : borderI)
      {
        group_velocity_[bVI].setZero();
        group_velocity_[bVI][0] = std::pow(-1.0, bI) * -0.1;
      }
      bI++;
    }
    break;
  }

  case BC_SQUASH:
  {
    mesh->clear_fixed_vertices();
    bc_groups_.resize(0);
    int bI = 0;
    for (const auto borderI : mesh->bc_groups_)
    {
      mesh->set_fixed(borderI);
      bc_groups_.emplace_back(borderI);
      for (const auto bVI : borderI)
      {
        group_velocity_[bVI].setZero();
        group_velocity_[bVI][0] = std::pow(-1.0, bI) * 0.03;
      }
      bI++;
    }
    break;
  }

  case BC_STRETCHNSQUASH:
  {
    mesh->clear_fixed_vertices();
    bc_groups_.resize(0);
    int bI = 0;
    for (const auto borderI : mesh->bc_groups_)
    {
      mesh->set_fixed(borderI);
      bc_groups_.emplace_back(borderI);
      for (const auto bVI : borderI)
      {
        group_velocity_[bVI].setZero();
        group_velocity_[bVI][0] = std::pow(-1.0, bI) * -0.9;
      }
      bI++;
    }

    velocity_turning_points_.first = mesh->bc_groups_[0].front();
    velocity_turning_points_.second(0, 0) =
        mesh->V_(velocity_turning_points_.first, 0) - 0.8;
    velocity_turning_points_.second(0, 1) =
        mesh->V_(velocity_turning_points_.first, 0) + 0.4;

    break;
  }

  case BC_BEND:
  {
    mesh->clear_fixed_vertices();
    bc_groups_.resize(0);
    int bI = 0;
    for (const auto borderI : mesh->bc_groups_)
    {
      mesh->set_fixed(borderI);
      bc_groups_.emplace_back(borderI);
      for (size_t bVI = 0; bVI + 1 < borderI.size(); bVI++)
      {
        angVel_bc_groups_[borderI[bVI]] = std::pow(-1.0, bI) * -0.05 * M_PI;
        rotCenter_bc_groups_[borderI[bVI]] = mesh->V_.row(borderI.back()).transpose();
      }
      bI++;
    }
    break;
  }

  case BC_TWIST:
  {
    mesh->clear_fixed_vertices();

    const RowVector3d rotCenter = mesh->bbox.colwise().mean();

    bc_groups_.resize(0);
    int bI = 0;
    for (const auto borderI : mesh->bc_groups_)
    {
      mesh->set_fixed(borderI);
      bc_groups_.emplace_back(borderI);
      for (size_t bVI = 0; bVI < borderI.size(); bVI++)
      {
        angVel_bc_groups_[borderI[bVI]] = std::pow(-1.0, bI) * -0.1 * M_PI;
        rotCenter_bc_groups_[borderI[bVI]] = rotCenter.transpose().topRows(DIM);
      }
      bI++;
    }
    break;
  }

  case BC_TWISTNSTRETCH:
  {
    mesh->clear_fixed_vertices();

    const RowVector3d rotCenter = mesh->bbox.colwise().mean();

    bc_groups_.resize(0);
    int bI = 0;
    for (const auto borderI : mesh->bc_groups_)
    {
      mesh->set_fixed(borderI);
      bc_groups_.emplace_back(borderI);
      for (size_t bVI = 0; bVI < borderI.size(); bVI++)
      {
        angVel_bc_groups_[borderI[bVI]] = std::pow(-1.0, bI) * -0.1 * M_PI;
        rotCenter_bc_groups_[borderI[bVI]] = rotCenter.transpose().topRows(DIM);

        group_velocity_[borderI[bVI]].setZero();
        group_velocity_[borderI[bVI]][0] = std::pow(-1.0, bI) * -0.1;
      }
      bI++;
    }
    break;
  }

  case BC_TWISTNSNS_OLD:
  case BC_TWISTNSNS:
  {
    mesh->clear_fixed_vertices();

    const RowVector3d rotCenter = mesh->bbox.colwise().mean();

    bc_groups_.resize(0);
    int bI = 0;
    for (const auto borderI : mesh->bc_groups_)
    {
      mesh->set_fixed(borderI);
      bc_groups_.emplace_back(borderI);
      for (size_t bVI = 0; bVI < borderI.size(); bVI++)
      {
        angVel_bc_groups_[borderI[bVI]] = std::pow(-1.0, bI) * -0.4 * M_PI;
        rotCenter_bc_groups_[borderI[bVI]] = rotCenter.transpose().topRows(DIM);

        group_velocity_[borderI[bVI]].setZero();
        if (script_type_ == BC_TWISTNSNS)
        {
          group_velocity_[borderI[bVI]][0] = std::pow(-1.0, bI) * -1.2;
        }
        else if (script_type_ == BC_TWISTNSNS_OLD)
        {
          group_velocity_[borderI[bVI]][0] = std::pow(-1.0, bI) * -0.9;
        }
      }
      bI++;
    }

    velocity_turning_points_.first = mesh->bc_groups_[0].front();
    if (script_type_ == BC_TWISTNSNS)
    {
      velocity_turning_points_.second(0, 0) =
          mesh->V_(velocity_turning_points_.first, 0) - 1.2;
    }
    else
    {
      velocity_turning_points_.second(0, 0) =
          mesh->V_(velocity_turning_points_.first, 0) - 0.8;
    }
    velocity_turning_points_.second(0, 1) =
        mesh->V_(velocity_turning_points_.first, 0) + 0.4;
    break;
  }

  case BC_RUBBERBANDPULL:
  {
    mesh->clear_fixed_vertices();
    bc_groups_.resize(0);
    bc_groups_.resize(2);

    // grab top, waist, and bottom:
    RowVectorXd bottomLeft = mesh->V_.colwise().minCoeff();
    RowVectorXd topRight = mesh->V_.colwise().maxCoeff();
    RowVectorXd range = topRight - bottomLeft;
    bool turningPointSet = false;
    for (int vI = 0; vI < mesh->V_.rows(); ++vI)
    {
      if (mesh->V_(vI, 1) < bottomLeft[1] + range[1] * 0.02)
      {
        mesh->set_fixed(vI);
        bc_groups_[1].emplace_back(vI);
        group_velocity_[vI].setZero();
        group_velocity_[vI][1] = -0.2;
      }
      else if (mesh->V_(vI, 1) > topRight[1] - range[1] * 0.02)
      {
        mesh->set_fixed(vI);
        bc_groups_[1].emplace_back(vI);
        group_velocity_[vI].setZero();
        group_velocity_[vI][1] = 0.2;
      }
      else if ((mesh->V_(vI, 1) < topRight[1] - range[1] * 0.48) &&
               (mesh->V_(vI, 1) > bottomLeft[1] + range[1] * 0.48))
      {
        mesh->set_fixed(vI);
        bc_groups_[0].emplace_back(vI);
        group_velocity_[vI].setZero();
        group_velocity_[vI][0] = -2.5; // previously -2.0
        if (!turningPointSet)
        {
          turningPointSet = true;
          velocity_turning_points_.first = vI;
          velocity_turning_points_.second(0, 0) = mesh->V_(vI, 0) - 5.0;
        }
      }
    }

    break;
  }

  // case BC_ONEPOINT:
  // {
  //   const RowVecD center = mesh->bbox.colwise().mean();
  //   mesh->V_.rowwise() = center.leftCols(DIM);
  //   mesh->V_.col(1).array() += (mesh->bbox(1, 1) - mesh->bbox(0, 1)) / 2.0;
  //   break;
  // }
  case BC_ONEPOINT:
    mesh->clear_fixed_vertices();
    mesh->set_fixed(0);
    break;

  case BC_RANDOM:
  {
    mesh->V_.setRandom();
    mesh->V_ /= 2.0;
    RowVector3d offset = mesh->bbox.colwise().mean();
    offset[1] += (mesh->bbox(1, 1) - mesh->bbox(0, 1)) / 2.0;
    offset.leftCols(DIM) -= mesh->V_.row(0);
    mesh->V_.rowwise() += offset.leftCols(DIM);
    break;
  }

  case BC_FALL:
  {
    mesh->V_.col(1).array() += 0.5 * (mesh->V_.colwise().maxCoeff() -
                                      mesh->V_.colwise().minCoeff())
                                         .norm();
    mesh->clear_fixed_vertices();
    break;
  }

  default:
    assert(0 && "invalid script_type_");
    break;
  }
}

template <int DIM>
int BoundaryConditions<DIM>::step_script(std::shared_ptr<Mesh> &mesh, double dt)
{
  VectorXd searchDir(mesh->V_.rows() * DIM);
  searchDir.setZero();
  int returnFlag = 0;
  switch (script_type_)
  {
  case BC_NULL:
    break;

  case BC_HANG:
    break;
  case BC_HANGENDS:
    break;
  case BC_STRETCH:
  case BC_SQUASH:
    for (const auto &movingVerts : group_velocity_)
    {
      searchDir.segment<DIM>(movingVerts.first * DIM) =
          movingVerts.second * dt;
    }
    break;

  case BC_STRETCHNSQUASH:
  {
    bool flip = false;
    if ((mesh->V_(velocity_turning_points_.first, 0) <=
         velocity_turning_points_.second(0, 0)) ||
        (mesh->V_(velocity_turning_points_.first, 0) >=
         velocity_turning_points_.second(0, 1)))
    {
      flip = true;
    }
    for (auto &movingVerts : group_velocity_)
    {
      if (flip)
      {
        movingVerts.second[0] *= -1.0;
      }
      searchDir.segment<DIM>(movingVerts.first * DIM) =
          movingVerts.second * dt;
    }
    break;
  }

  case BC_BEND:
    for (const auto &movingVerts : angVel_bc_groups_)
    {
      const Matrix3d rotMtr =
          AngleAxis<double>(movingVerts.second * dt,
                            Vector3d::UnitZ())
              .toRotationMatrix();
      const auto rotCenter = rotCenter_bc_groups_.find(movingVerts.first);
      assert(rotCenter != rotCenter_bc_groups_.end());

      searchDir.segment<DIM>(movingVerts.first * DIM) = (rotMtr.block<DIM, DIM>(0, 0) * (mesh->V_.row(movingVerts.first).transpose() - rotCenter->second) + rotCenter->second) - mesh->V_.row(movingVerts.first).transpose();
    }
    break;

  case BC_TWIST:
    for (const auto &movingVerts : angVel_bc_groups_)
    {
      const Matrix3d rotMtr =
          AngleAxis<double>(movingVerts.second * dt,
                            Vector3d::UnitX())
              .toRotationMatrix();
      const auto rotCenter = rotCenter_bc_groups_.find(movingVerts.first);
      assert(rotCenter != rotCenter_bc_groups_.end());

      searchDir.segment<DIM>(movingVerts.first * DIM) = (rotMtr.block<DIM, DIM>(0, 0) * (mesh->V_.row(movingVerts.first).transpose() - rotCenter->second) + rotCenter->second) - mesh->V_.row(movingVerts.first).transpose();
    }
    break;

  case BC_TWISTNSTRETCH:
  {
    for (const auto &movingVerts : angVel_bc_groups_)
    {
      const Matrix3d rotMtr =
          AngleAxis<double>(movingVerts.second * dt,
                            Vector3d::UnitX())
              .toRotationMatrix();
      const auto rotCenter = rotCenter_bc_groups_.find(movingVerts.first);
      assert(rotCenter != rotCenter_bc_groups_.end());

      searchDir.segment<DIM>(movingVerts.first * DIM) = (rotMtr.block<DIM, DIM>(0, 0) * (mesh->V_.row(movingVerts.first).transpose() - rotCenter->second) + rotCenter->second) - mesh->V_.row(movingVerts.first).transpose();
    }
    for (const auto &movingVerts : group_velocity_)
    {
      searchDir.segment<DIM>(movingVerts.first * DIM) += movingVerts.second * dt;
    }
    break;
  }

  case BC_TWISTNSNS_OLD:
  case BC_TWISTNSNS:
  {
    bool flip = false;
    if ((mesh->V_(velocity_turning_points_.first, 0) <=
         velocity_turning_points_.second(0, 0)) ||
        (mesh->V_(velocity_turning_points_.first, 0) >=
         velocity_turning_points_.second(0, 1)))
    {
      flip = true;
    }

    for (auto &movingVerts : angVel_bc_groups_)
    {
      //                    if(flip) {
      //                        movingVerts.second *= -1.0;
      //                    }

      const Matrix3d rotMtr =
          AngleAxis<double>(movingVerts.second * dt,
                            Vector3d::UnitX())
              .toRotationMatrix();
      const auto rotCenter = rotCenter_bc_groups_.find(movingVerts.first);
      assert(rotCenter != rotCenter_bc_groups_.end());

      searchDir.segment<DIM>(movingVerts.first * DIM) = (rotMtr.block<DIM, DIM>(0, 0) * (mesh->V_.row(movingVerts.first).transpose() - rotCenter->second) + rotCenter->second) - mesh->V_.row(movingVerts.first).transpose();
    }
    for (auto &movingVerts : group_velocity_)
    {
      if (flip)
      {
        movingVerts.second[0] *= -1.0;
      }
      searchDir.segment<DIM>(movingVerts.first * DIM) += movingVerts.second * dt;
    }
    break;
  }

  case BC_RUBBERBANDPULL:
  {
    if (mesh->V_(velocity_turning_points_.first, 0) <=
        velocity_turning_points_.second(0, 0))
    {
      velocity_turning_points_.second(0, 0) = -__DBL_MAX__;
      for (const auto &vI : bc_groups_[0])
      {
        mesh->free_vertex(vI);
        group_velocity_[vI].setZero();
      }
      for (const auto &vI : bc_groups_[1])
      {
        group_velocity_[vI].setZero();
      }
      returnFlag = 1;
    }
    for (const auto &movingVerts : group_velocity_)
    {
      searchDir.segment<DIM>(movingVerts.first * DIM) =
          movingVerts.second * dt;
    }
    break;
  }

  case BC_ONEPOINT:
    break;

  case BC_RANDOM:
    break;

  case BC_FALL:
    break;

  default:
    assert(0 && "invalid script_type_");
    break;
  }

  double stepSize = 1.0;
  #pragma omp parallel for
  for (int vI = 0; vI < mesh->V_.rows(); ++vI)
  {
    mesh->V_.row(vI) += stepSize * searchDir.segment<DIM>(vI * DIM).transpose();
  }

  return returnFlag;
}

template <int DIM>
void BoundaryConditions<DIM>::set_script(BCScriptType script_type)
{
  script_type_ = script_type;
}

template <int DIM>
const std::vector<std::vector<int>> &BoundaryConditions<DIM>::get_bc_groups(void) const
{
  return bc_groups_;
}

template <int DIM>
BCScriptType BoundaryConditions<DIM>::get_script_type(const std::string &str)
{

  for (size_t i = 0; i < script_type_strings.size(); i++)
  {
    if (str == script_type_strings[i])
    {
      return BCScriptType(i);
    }
  }
  return BC_NULL;
}

template <int DIM>
std::string BoundaryConditions<DIM>::get_script_name(BCScriptType script_type)
{
  assert(script_type < script_type_strings.size());
  return script_type_strings[script_type];
}

template <int DIM>
void BoundaryConditions<DIM>::get_script_names(std::vector<std::string>& names)
{
  names = script_type_strings;
}

template class mfem::BoundaryConditions<3>;
template class mfem::BoundaryConditions<2>;
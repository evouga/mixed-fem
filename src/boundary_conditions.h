#pragma once

// Taken from https://github.com/penn-graphics-research/DOT/blob/master/src/AnimScripter.hpp
#include <EigenTypes.h>
#include <memory>
#include "config.h"

namespace mfem
{

  class Mesh;

  template <int dim>
  class BoundaryConditions
  {
  protected:
    BCScriptType script_type_;

    std::vector<std::vector<int>> bc_groups_;

    std::map<int, Eigen::Matrix<double, dim, 1>> group_velocity_;
    std::pair<int, Eigen::Matrix<double, dim, 2>> velocity_turning_points_;

    std::map<int, double> angVel_bc_groups_;
    std::map<int, Eigen::Matrix<double, dim, 1>> rotCenter_bc_groups_;

  protected:
    static const std::vector<std::string> script_type_strings;

  public:
    BoundaryConditions(BCScriptType script_type = BC_NULL);

  public:

    static void init_boundary_groups(const Eigen::MatrixXd &V,
      std::vector<std::vector<int>> &bc_groups, double ratio);

    void init_script(std::shared_ptr<Mesh> &mesh);
    int step_script(std::shared_ptr<Mesh> &mesh, double dt);
    void set_script(BCScriptType script_type);

    const std::vector<std::vector<int>> &get_bc_groups(void) const;

  public:
    static BCScriptType get_script_type(const std::string &str);
    static std::string get_script_name(BCScriptType script_type);
    static void get_script_names(std::vector<std::string>& names);
  };

}

#pragma once

#include <EigenTypes.h>

namespace mfem {

  struct OptimizerData {
    

    OptimizerData(std::string output_filename)
        : output_filename_(output_filename) {
    }

    OptimizerData() : output_filename_("../data/output/results.mat") {
    }

    virtual void clear();
    virtual void write() const;
    virtual void print_data() const;

    std::string output_filename_;
    std::vector<double> energy_residuals_;
    std::vector<double> energies_;
    std::vector<double> egrad_;
    std::vector<double> egrad_x_;
    std::vector<double> egrad_s_;
    std::vector<double> egrad_la_;

  }; 


}
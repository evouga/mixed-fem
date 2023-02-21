#pragma once

#include <EigenTypes.h>
#include <chrono>
#include <unordered_map>

namespace mfem {
  
  /// @brief A simple timer class
  class Timer {

    using Time = std::chrono::high_resolution_clock;
    
    // For each key, store the clock, total time, max time,  # of measurements
    using T = std::tuple<std::chrono::time_point<Time>, double, double, int>;

    using KeyMap = std::unordered_map<std::string, T>;

  public:

    Timer() : output_filename_("timing.csv") {
    }

    // Start timer for a key (creates entry if does not exist)
    void start(const std::string& key, const std::string& tag = "Default");

    // Records elapsed time for given key
    void stop(const std::string& key, const std::string& tag = "Default");

    double total(const std::string& key,
        const std::string& tag = "Default") const;

    double average(const std::string& key,
        const std::string& tag = "Default") const;

    void print() const;

    void reset() {
      timers_.clear();
    }

    void write_csv(int step) const;

  private:

    // Map tags to key maps
    std::unordered_map<std::string, KeyMap> timers_;

    std::string output_filename_;
  };

  struct OptimizerData {
    
    static OptimizerData& get() {
      static OptimizerData instance;
      return instance;
    }

    OptimizerData(std::string output_filename)
        : output_filename_(output_filename) {
    }

    OptimizerData() : output_filename_("optimizer_data.csv") {
    }

    OptimizerData(OptimizerData const&) = delete;
    void operator=(OptimizerData const&) = delete;

    virtual void clear();
    virtual void write_csv(int step) const;
    virtual void print_data(bool print_timing = true) const;
    void add(const std::string& key, double value);

    std::string output_filename_;
    std::map<std::string, std::vector<double>> map_;	
    size_t min_length_ = 11;
    Timer timer;

  }; 

}
#include "optimizer_data.h"
#include <igl/writeDMAT.h>
#include <iostream>
#include <iomanip>      // std::setw
#include <fstream>
#include <filesystem>

using namespace mfem;
using namespace Eigen;

void OptimizerData::clear() {
  timer.reset();
  map_.clear();
}

void OptimizerData::write() const {
}

void OptimizerData::add(const std::string& key, double value) {
  auto it = map_.find(key);
  if (it == map_.end()) {
    std::vector<double> new_values(1, value);
    map_[key] = new_values;
  } else {
    std::vector<double>& values = it->second;
    values.push_back(value);
  }
}

void OptimizerData::print_data(bool print_timing) const {

  // Copy cout state so we can restore it later
  std::ios cout_state(nullptr);
  cout_state.copyfmt(std::cout);

  // Header Top
  std::cout << "┌─";
  for (auto it = map_.begin(); it != map_.end(); ) {
    int len = std::max(min_length_, it->first.length());
    for (int i = 0; i < len; ++i) {
      std::cout << "─";
    }
    if (++it == map_.end())
      std::cout << "─┐\n";
    else
      std::cout << "─┬─";
  }

  // Labels
  std::cout << "│ ";
  for (auto it = map_.begin(); it != map_.end(); ) {
    std::cout << it->first;
    int padding = std::max(min_length_, it->first.length())
        - it->first.length();
    if (padding > 0) {
      for (int i = 0; i < padding; ++i) {
        std::cout << " ";
      }
    }

    if (++it == map_.end())
      std::cout << " │\n";
    else
      std::cout << " │ ";
  }

  // Header Bottom
  std::cout << "├─";
  for (auto it = map_.begin(); it != map_.end(); ) {
    int len = std::max(min_length_, it->first.length());
    for (int i = 0; i < len; ++i) {
      std::cout << "─";
    }
    if (++it == map_.end())
      std::cout << "─┤\n";
    else
      std::cout << "─┼─";
  }


  // Data
  size_t max_size = 0;
  for (auto it = map_.begin(); it != map_.end(); ++it) {
    max_size = std::max(max_size, it->second.size());
  }

  for (size_t i = 0; i < max_size; ++i) {
    std::cout << "│ ";

    for (auto it = map_.begin(); it != map_.end(); ++it) {

      if (it->first == " Iteration") {
        std::cout << std::defaultfloat;
      } else {
        std::cout << std::scientific;
      }
      int len = std::max(min_length_, it->first.length());
      std::cout << std::setprecision(5);

      std::cout << std::setw(len) << it->second[i];
      std::cout << " │ ";
    }
    std::cout << std::endl;
  }

  // Footer
  std::cout << "└─";
  for (auto it = map_.begin(); it != map_.end(); ) {
    int len = std::max(min_length_, it->first.length());
    for (int i = 0; i < len; ++i) {
      std::cout << "─";
    }
    if (++it == map_.end())
      std::cout << "─┘\n";
    else
      std::cout << "─┴─";
  }

  if (print_timing) {
    timer.print();
  }
  
  // Restore cout format
  std::cout.copyfmt(cout_state);
}

void Timer::start(const std::string& key, const std::string& tag) {

  auto it = timers_.find(tag);
  if (it == timers_.end()) {
    timers_[tag] = KeyMap();
  }

  KeyMap& key_map = timers_[tag];
  auto it2 = key_map.find(key);
  if (it2 == key_map.end()) {
    key_map[key] = std::make_tuple(Time::now(), 0.0, 0);
  } else {
    T& tup = key_map[key];
    std::get<0>(tup) = Time::now();
  }
}

void Timer::stop(const std::string& key, const std::string& tag) {
  auto end = Time::now();
  auto it = timers_.find(tag);
  if (it == timers_.end()) {
    std::cerr << "Invalid timer tag: " << tag << std::endl;
  } else {
    KeyMap& key_map = it->second;
    auto it2 = key_map.find(key);
    if (it2 == key_map.end()) {
      std::cerr << "Invalid timer key: " << key << std::endl;
    } else {
      T& tup = it2->second;
      std::chrono::duration<double, std::milli> fp_ms = end - std::get<0>(tup);
      std::get<1>(tup) += fp_ms.count(); // add to total time
      std::get<2>(tup) += 1;             // increment measurement count
    }
  }
}

double Timer::total(const std::string& key, const std::string& tag) const {
  auto it = timers_.find(tag);
  if (it != timers_.end()) {
    const KeyMap& key_map = it->second;
    auto it2 = key_map.find(key);
    if (it2 != key_map.end()) {
      const T& tup = it2->second;
      return std::get<1>(tup);
    }
  }
  return 0;
}

double Timer::average(const std::string& key, const std::string& tag) const {
  auto it = timers_.find(tag);
  if (it != timers_.end()) {
    const KeyMap& key_map = it->second;
    auto it2 = key_map.find(key);
    if (it2 != key_map.end()) {
      const T& tup = it2->second;
      return std::get<1>(tup) / std::get<2>(tup);
    }
  }
  return 0;
}

void Timer::print() const {
  std::cout << "Timing (in ms): " << std::endl;
  auto it = timers_.begin();

  // For each tag, print out the keys
  while(it != timers_.end()) {
    std::string tag = it->first;
    std::cout << "  [" << tag << "]" << std::endl;
    const KeyMap& key_map = it->second;
    auto it2 = key_map.begin();

    // For the current tag, get the longest key length for spacing
    int max_len = 0;
    for (auto it2 = key_map.begin(); it2 != key_map.end(); ++it2) {
      std::string key = it2->first;
      max_len = std::max(max_len, (int) key.length());
    }

    // Print out each keys average and total times
    for (auto it2 = key_map.begin(); it2 != key_map.end(); ++it2) {
      std::string key = it2->first;
      const T& tup = it2->second;
      double t = std::get<1>(tup);
      int n = std::get<2>(tup);

      std::cout << "    [" << std::setw(max_len) << key << "] "
          << std::fixed << " Avg: " << std::setw(10) << t/((double) n)
          << "   Total: " << std::setw(10) << t << std::endl;
    }
    ++it;
  }
}

void Timer::write_csv(int step) const {

  // Open output file in append mode if step > 0 
  std::ios_base::openmode mode = std::ios_base::out;
  if (step != 0) {
    mode |= std::ios_base::app;
  }

  std::ofstream file(output_filename_, mode);
  if (!file.is_open()) {
    std::cerr << "Could not open file: " << output_filename_ << std::endl;
    return;
  }

  // Write header if step 0 or file is empty
  auto is_empty = (std::filesystem::file_size(output_filename_) == 0);

  if (step == 0 || is_empty) {
    file << "Step,Tag,Key,Total,Average,Num Measurements" << std::endl;
  }

  // Write data
  for (const auto& tag_pair : timers_) {
    const std::string& tag = tag_pair.first;
    const KeyMap& key_map = tag_pair.second;
    for (const auto& key_pair : key_map) {
      const std::string& key = key_pair.first;
      const T& tup = key_pair.second;
      file << step << "," << tag << "," << key << "," << std::get<1>(tup) << ","
          << std::get<1>(tup) / std::get<2>(tup) << "," << std::get<2>(tup)
          << std::endl;
    }
  }
  file.close();
}
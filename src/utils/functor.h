#pragma once

#include <Eigen/Core>
#include <type_traits>
#include <memory>

namespace mfem {

  template <typename T>
  class Functor {
    void apply(T&) = 0;
  };

  // Functor to scale the input
  template <typename T>
  class ScaleFunctor : Functor<T> {

    ScaleFunctor(double alpha) : alpha_(alpha) {
    }

    static std::unique_ptr<Functor<T>> make(double alpha) {
      return std::make_unique<ScaleFunctor<T>>(alpha);
    }

    void apply(T& t) override {
      t = alpha_ * t;
    }

    double alpha_;
  };

  // Functor to invert the input. Different behaviors
  // for scalar variables and eigen matrices.
  //template <typename T>
  //class InverseFunctor : Functor<T> {

  //  static std::unique_ptr<Functor<T>> make() {
  //    return std::make_unique<InverseFunctor<T>>();
  //  }

  //  void apply(T& t) override {
  //    if constexpr (std::is_scalar<T>) {
  //      t = 1 / t;
  //    } else {
  //      t = t.inverse();
  //    }
  //  }
  //};

}

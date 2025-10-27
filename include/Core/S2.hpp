#pragma once

#include <cmath>
#include <optional>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <manif/manif.h>
#include <manif/SO3.h>


// Based on He-2021, [https://arxiv.org/abs/2102.03804] Eq. (64-70)

namespace S2 {

  using Vector2d = Eigen::Vector2d;
  using Vector3d = Eigen::Vector3d;
  using Matrix3d = Eigen::Matrix3d; 
  using Matrix2d = Eigen::Matrix2d; 
  using Matrix2x3d = Eigen::Matrix<double, 2, 3>;
  using Matrix3x2d = Eigen::Matrix<double, 3, 2>; 

  inline Matrix3d R(const Vector3d& x) {
    return manif::SO3d::Tangent(x).exp().rotation();
  }


  inline Matrix3x2d B(const Vector3d& x) {
    Vector3d   e_i  = Vector3d::UnitZ();
    Matrix3x2d E_jk = Matrix3d::Identity().leftCols<2>();

    Matrix3x2d out;

    // Handle near-zero x: rotation ~ Identity
    if (x.norm() < 1e-12) {
        out = E_jk;
    } else {
        Vector3d cross      = e_i.cross(x);
        double   cross_norm = cross.norm();
        double   dot        = e_i.dot(x);                    

        if (cross_norm < 1e-12) {
            // x parallel to e_i: angle is 0 (c>=0) or pi (c<0), axis arbitrary ⟂ e_i
            out = dot >= 0. ? E_jk : R(Vector3d::UnitX() * M_PI) * E_jk;
        } else {
            double   theta = std::atan2(cross_norm, dot);
            Vector3d axis  = cross / cross_norm;                  // unit axis
            out = R(axis*theta) * E_jk;
        }
    }

    return out/x.norm();
  }


  inline Vector3d Exp(const Vector3d& x, const Vector2d& u) {
    if (u.norm() < 1e-12)
      return x;

    return R(x.cross(B(x) * u)) * x;
  }


  inline Vector2d Log(const Vector3d& x, const Vector3d& y) {
    std::cout << "Log x: " << x.transpose() << std::endl;
    std::cout << "Log y: " << y.transpose() << std::endl;

    Vector3d cross      = x.cross(y);
    double   cross_norm = cross.norm();
    double   dot        = x.dot(y);

    if (cross_norm >= 1e-12) {
      Vector3d axis  = cross.cross(x).normalized(); // -[x]_x^2 * y = x cross y cross x
      double   theta = std::atan2(cross_norm, dot);
      return B(x).transpose() * (axis*theta);
    }

    // Collinear case: s ~ 0
    // y in same direction as x -> zero rotation in the tangent or either
    // Antipodal: angle = pi, axis is not unique (any unit vector ⟂ x).
    // Pick a consistent tangent direction; e.g. first basis vector of B(x).
    // This makes the result finite (norm = pi) but not unique by geometry.
    return dot >= 0. ? Vector2d::Zero() : (M_PI*Vector2d::UnitX()).eval();
  }

  Matrix2x3d LogJ_a(const Vector3d& x) {
    auto a = x.normalized();
    return -B(a).transpose() * (manif::skew(a) * manif::skew(a));
  }

  Matrix2x3d LogJ_a(const Vector3d& y_, const Vector3d& x_) {
    auto x = x_.normalized();
    auto y = y_.normalized();

    Vector3d double_cross      = x.cross(y).cross(x);
    double   double_cross_norm = double_cross.norm(); 
    Vector3d cross      = x.cross(y);
    double   cross_norm = cross.norm();
    double   dot = x.dot(y);
    if (cross_norm < 1e-12)
      return LogJ_a(y_);
    
    double theta = std::atan2(cross_norm, dot);
    
    return B(x).transpose() 
           * (1/double_cross_norm * (Matrix3d::Identity() - (double_cross*double_cross.transpose())/(double_cross_norm*double_cross_norm)) * -(manif::skew(x)*manif::skew(x))*theta 
              + (cross/cross_norm) * (dot*x.transpose() - x.transpose())/cross_norm );
  }

  Matrix3x2d ExpJ_b(const Vector3d& x_) {
    auto x = x_.normalized();
    return -(manif::skew(x) * manif::skew(x)) * B(x);
  }

  Matrix3x2d ExpJ_b(const Vector3d& x_, const Vector2d& u) {
    auto x = x_.normalized();
    if (u.norm() < 1e-12)
      return ExpJ_b(x_);
    
    auto rot = R(x.cross(B(x) * u));
    auto Jr = manif::SO3d::Tangent(x.cross(B(x) * u)).rjac();
    return  manif::skew(x) * rot * manif::skew(x) * Jr * manif::skew(x) * B(x);
  }


  Matrix3d ExpJ_a() {
    return Matrix3d::Identity();
  }

}
#pragma once

#include <execution>
#include <numeric>
#include <algorithm>
#include <boost/circular_buffer.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "Core/Imu.hpp"
#include "Core/Map.hpp"
#include "Core/Octree.hpp"

#include "Utils/Config.hpp"
#include "Utils/PCL.hpp"


#include <manif/manif.h>
#include <manif/SE3.h>
#include <manif/Bundle.h>
#include <manif/Rn.h>


struct State {

  using Matrix24d = Eigen::Matrix<double, 24, 24>;
  using Matrix24x12d = Eigen::Matrix<double, 24, 12>;
  using Matrix12d = Eigen::Matrix<double, 12, 12>;
  using Matrix3x6d = Eigen::Matrix<double, 3, 6>;
  using Matrix1x6d = Eigen::Matrix<double, 1, 6>;
  using Vector24d = Eigen::Matrix<double, 24, 1>;

  using BundleT = manif::Bundle<double,
      manif::SE3, // position & rotation
      manif::SE3, // imu2lidar position & rotation
      manif::R3,  // velocity
      manif::R3,  // angular bias
      manif::R3,  // acceleartion bias
      manif::R3   // gravity
  >;

  using Tangent = typename BundleT::Tangent; 

  BundleT X;
  Matrix24d P;
  Matrix12d Q;

  Eigen::Vector3d w;      // angular velocity (IMU input)
  Eigen::Vector3d a;      // linear acceleration (IMU input)

  double stamp;

  State() : stamp(0.0) { 
    Config& cfg = Config::getInstance();
    Eigen::Vector3d zero_vec = Eigen::Vector3d(0., 0., 0.);

    X = BundleT(manif::SE3d(0., 0., 0., 0., 0., 0.), // x, y, z, roll, pitch, yaw
                manif::SE3d(0., 0., 0, cfg.sensors.extrinsics.lidar2baselink.roll,
                                       cfg.sensors.extrinsics.lidar2baselink.pitch,
                                       cfg.sensors.extrinsics.lidar2baselink.yaw),
                manif::R3d(zero_vec),
                manif::R3d(zero_vec),
                manif::R3d(zero_vec),
                manif::R3d(Eigen::Vector3d(0., 0., -cfg.sensors.extrinsics.gravity))); // NOTE
    
    P.setIdentity();
    P *= 1e-3f;

    w.setZero();
    a.setZero();

    // Control signal noise (never changes)
    Q.setZero();
 
    Q.block<3, 3>(0, 0) = cfg.ikfom.covariance.gyro       * Eigen::Matrix3d::Identity(); // n_w
    Q.block<3, 3>(3, 3) = cfg.ikfom.covariance.accel      * Eigen::Matrix3d::Identity(); // n_a
    Q.block<3, 3>(6, 6) = cfg.ikfom.covariance.bias_gyro  * Eigen::Matrix3d::Identity(); // n_{b_w}
    Q.block<3, 3>(9, 9) = cfg.ikfom.covariance.bias_accel * Eigen::Matrix3d::Identity(); // n_{b_a}
  } 

  void predict(const Imu& imu, const double& dt) {
PROFC_NODE("predict")

    Matrix24d Gx, Gf; // Adjoint_X(u)^{-1}, J_r(u)  Sola-18, [https://arxiv.org/abs/1812.01537]
    X = X.plus(f(imu.lin_accel, imu.ang_vel) * dt, Gx, Gf);

    // UPDATE COVARIANCE
    Matrix24d    Fx = Gx + Gf * df_dx(imu) * dt; // He-2021, [https://arxiv.org/abs/2102.03804] Eq. (26)
    Matrix24x12d Fw = Gf * df_dw(imu) * dt;      // He-2021, [https://arxiv.org/abs/2102.03804] Eq. (27)

    P = Fx * P * Fx.transpose() + Fw * Q * Fw.transpose(); 

    // SAVE EXTRA INFO
    a = imu.lin_accel;
    w = imu.ang_vel;

    stamp = imu.stamp;
  }


  void predict(const double& t) {
    double dt = t - this->stamp;
    assert(dt >= 0);
    
    X = X.plus(f(a, w) * dt);
  }


  Tangent f(const Eigen::Vector3d& lin_acc, const Eigen::Vector3d& ang_vel) {
    // UPDATE STATE
    Tangent u = Tangent::Zero();
    u.element<0>().coeffs() << lin_acc - b_a() /* -n_a */ + R().transpose() * g(), 
                               ang_vel - b_w() /* -n_w */;
    u.element<2>().coeffs() = R() * (lin_acc - b_a() /* -n_a */) + g();
    // u.element<3>().coeffs() = n_{b_w} 
    // u.element<4>().coeffs() = n_{b_a}

    return u;
  }


  Matrix24d df_dx(const Imu& imu) {
    Matrix24d out = Matrix24d::Zero();

    // twist
      // velocity
      out.block<3, 3>( 0,  3) = -R().transpose()*manif::skew(g()) * -R(); // w.r.t R := d(R^-1*g)/dR * d(R^-1)/dR
      out.block<3, 3>( 0, 18) = -Eigen::Matrix3d::Identity(); // w.r.t b_a
      out.block<3, 3>( 0, 21) =  R().transpose(); // w.r.t g
      // angular velocity
      out.block<3, 3>( 3, 15) = -Eigen::Matrix3d::Identity(); // w.r.t b_w

    // velocity update
    out.block<3, 3>(12, 3)  = -R() * manif::skew(imu.lin_accel - b_a()); // w.r.t R
    out.block<3, 3>(12, 18) = -R(); // w.r.t b_a 
    out.block<3, 3>(12, 21) =  Eigen::Matrix3d::Identity(); // w.r.t g

    return out;
  }


  Matrix24x12d df_dw(const Imu& imu) {
    // w = (n_w, n_a, n_{b_w}, n_{b_a})

    Matrix24x12d out = Matrix24x12d::Zero();

    // twist
      out.block<3, 3>( 0, 3) = -Eigen::Matrix3d::Identity(); // w.r.t n_a
      out.block<3, 3>( 3, 0) = -Eigen::Matrix3d::Identity(); // w.r.t n_w

    // velocity update
    out.block<3, 3>(12, 3) = -R(); // w.r.t n_a
    // b_w update
    out.block<3, 3>(15, 6) = Eigen::Matrix3d::Identity(); // w.r.t n_{b_w}
    // b_a update
    out.block<3, 3>(18, 9) = Eigen::Matrix3d::Identity(); // w.r.t n_{b_a}
    
    return out;
  }

  void update(PointCloudT::Ptr& cloud, thuni::Octree& map) {
PROFC_NODE("update")

    Config& cfg = Config::getInstance();

    if (map.size() == 0)
      return;

    Matches first_matches;
    int query_iters = cfg.ikfom.query_iters;

// OBSERVATION MODEL

    auto h_model = [&](const State& s,
                       Eigen::Matrix<double, Eigen::Dynamic, 12>& H,
                       Eigen::Matrix<double, Eigen::Dynamic,  1>& z) {

      int N = cloud->size();

      std::vector<bool> chosen(N, false);
      Matches matches(N);

      if (query_iters-- > 0) {
        std::vector<int> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        
        std::for_each(
          std::execution::par_unseq,
          indices.begin(),
          indices.end(),
          [&](int i) {
            PointT pt = cloud->points[i];
            Eigen::Vector3f p(pt.x, pt.y, pt.z);
            Eigen::Vector3f g = s.affine3f() * s.I2L_affine3f() * p; // global coords 

            std::vector<pcl::PointXYZ> neighbors;
            std::vector<float> pointSearchSqDis;
            map.knnNeighbors(pcl::PointXYZ(g(0), g(1), g(2)),
                             cfg.ikfom.plane.points,
                             neighbors,
                             pointSearchSqDis);
            
            if (neighbors.size() < cfg.ikfom.plane.points 
                or pointSearchSqDis.back() > cfg.ikfom.plane.max_sqrt_dist)
                  return;
            
            Eigen::Vector4f p_abcd = Eigen::Vector4f::Zero();
            if (not estimate_plane(p_abcd, neighbors, cfg.ikfom.plane.plane_threshold))
              return;
            
            chosen[i] = true;
            matches[i] = Match(p, g, p_abcd);
          }
        ); // end for_each

        for (int i = 0; i < N; i++) {
          if (chosen[i])
            first_matches.push_back(matches[i]);        
        }

      } else {
        for (auto& match : first_matches) {
          match.global = s.affine3f() * s.I2L_affine3f() * match.local; 
        }
      }

      H = Eigen::MatrixXd::Zero(first_matches.size(), 12);
      z = Eigen::MatrixXd::Zero(first_matches.size(), 1);

      std::vector<int> indices(first_matches.size());
      std::iota(indices.begin(), indices.end(), 0);

      // For each match, calculate its derivative and distance
      std::for_each (
        std::execution::par_unseq,
        indices.begin(),
        indices.end(),
        [&](int i) {
          Match match = first_matches[i];
          Eigen::Vector3d p_lidar = match.local.cast<double>();
          Eigen::Vector3d p_imu   = (s.I2L_affine3f() * match.local).cast<double>();

          // Normal vector to plane
          Eigen::Vector3d n = match.n.head(3).cast<double>();

          // Jacobian of SE3 act.
          Matrix3x6d J_s; // Jacobian of state (pos., rot.)
          s.X.element<0>().act(p_imu, J_s);

          Matrix1x6d A = n.transpose() * J_s; 

          H.block<1, 6>(i, 0) << A(0), A(1), A(2), A(3), A(4), A(5);

          if (cfg.ikfom.estimate_extrinsics) {
            Matrix3x6d J_e; // Jacobian of extrinsics (pos., rot.)
            s.X.element<1>().act(p_lidar, J_e);

            Matrix1x6d B = n.transpose() * s.R() * J_e;

            H.block<1, 6>(i, 6) << B(0), B(1), B(2), B(3), B(4), B(5);
          }

          z(i) = -match.dist2plane();
        }
      ); // end for_each

    }; // end h_model

// IESEKF UPDATE

    BundleT   X_predicted = X;
    Matrix24d P_predicted = P;

    Eigen::Matrix<double, Eigen::Dynamic, 12> H;
    Eigen::Matrix<double, Eigen::Dynamic,  1> z;
    Matrix24d KH;

    double R = cfg.ikfom.lidar_noise;

    int i(0);

    do {
      h_model(*this, H, z); // Update H,z and set K to zeros

      // update P
      Matrix24d J;
      Tangent dx = X.minus(X_predicted, J); // Xu-2021, [https://arxiv.org/abs/2107.06829] Eq. (11)

      P = J.inverse() * P_predicted * J.inverse().transpose();

      Matrix12d HTH = H.transpose() * H / R;
      Matrix24d P_inv = P.inverse();
      P_inv.block<12, 12>(0, 0) += HTH;
      P_inv = P_inv.inverse();

      Vector24d Kz = Vector24d::Zero(); 
      Kz = P_inv.block<24, 12>(0, 0) * H.transpose() * z / R;

      KH.setZero();
      KH.block<24, 12>(0, 0) = P_inv.block<24, 12>(0, 0) * HTH;

      dx = Kz + (KH - Matrix24d::Identity()) * J.inverse() * dx; 
      X = X.plus(dx);

      if ((dx.coeffs().array().abs() <= cfg.ikfom.tolerance).all())
        break;

    } while(i++ < cfg.ikfom.max_iters);

    X = X;
    P = (Matrix24d::Identity() - KH) * P;
  }


// Getters
  inline Eigen::Vector3d p()       const { return X.element<0>().translation();             }
  inline Eigen::Matrix3d R()       const { return X.element<0>().quat().toRotationMatrix(); }
  inline Eigen::Quaterniond quat() const { return X.element<0>().quat();                    }
  inline Eigen::Matrix3d I2L_R()   const { return X.element<1>().quat().toRotationMatrix(); }
  inline Eigen::Vector3d I2L_t()   const { return X.element<1>().translation();             }
  inline Eigen::Vector3d v()       const { return X.element<2>().coeffs();                  }
  inline Eigen::Vector3d b_w()     const { return X.element<3>().coeffs();                  }
  inline Eigen::Vector3d b_a()     const { return X.element<4>().coeffs();                  }
  inline Eigen::Vector3d g()       const { return X.element<5>().coeffs();                  }

  inline Eigen::Affine3f affine3f() const {
    Eigen::Affine3d T;
    T.linear() = R();
    T.translation() = p();
    return T.cast<float>();
  }

  inline Eigen::Affine3f I2L_affine3f() const {
    Eigen::Affine3d T;
    T.linear() = I2L_R();
    T.translation() = I2L_t();
    return T.cast<float>();
  }

// Setters
  // Eigen::Vector3d p()     { return X.element<0>.coeffs();   }
  // Eigen::Vector3d R()     { return X.element<1>.rotation(); }
  void quat(const Eigen::Quaterniond& q) { X.element<0>().quat(q); }

  // Eigen::Vector3d I2L_R() { return X.element<2>.rotation(); }
  // Eigen::Vector3d I2L_t() { return X.element<3>.coeffs();   }
  // Eigen::Vector3d v()     { return X.element<4>.coeffs();   }
  void b_w(const Eigen::Vector3d& in) { X.element<3>() = manif::R3d(in); }
  void b_a(const Eigen::Vector3d& in) { X.element<4>() = manif::R3d(in); }
  void g(const Eigen::Vector3d& in)   { X.element<5>() = manif::R3d(in); }

};

typedef boost::circular_buffer<State> States;

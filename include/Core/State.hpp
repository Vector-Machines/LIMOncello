#pragma once

#include <execution>
#include <numeric>
#include <algorithm>
#include <boost/circular_buffer.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "Core/Imu.hpp"
#include "Core/Plane.hpp"
#include "Core/Octree.hpp"
#include <Core/S2.hpp>

#include "Utils/Config.hpp"
#include "Utils/PCL.hpp"


#include <manif/manif.h>
#include <manif/SGal3.h>
#include <manif/SE3.h>
#include <manif/Bundle.h>
#include <manif/Rn.h>


struct State {

  using BundleT = manif::Bundle<double,
      manif::SGal3,  // position & rotation & velocity & t
      manif::R3,     // angular bias
      manif::R3,     // acceleartion bias
      manif::R3      // gravity
  >;

  using Tangent = typename BundleT::Tangent; 

  static constexpr int DoF = BundleT::DoF-1;                             // DoF whole state
  static constexpr int DoFNoise = 4*3;                                   // b_w, b_a, n_{b_w}, n_{b_a}
  static constexpr int DoFObs = manif::SGal3d::DoF;   // DoF obsevation equation

  using MatrixDoF       = Eigen::Matrix<double, DoF, DoF>;
  using MatrixDoF_Noise = Eigen::Matrix<double, DoF+1, DoFNoise>;
  using MatrixNoise     = Eigen::Matrix<double, DoFNoise, DoFNoise>;

  using MatrixManif = Eigen::Matrix<double, DoF+1, DoF+1>;


  BundleT X;
  MatrixDoF P;
  MatrixNoise Q;

  Eigen::Vector3d w;      // angular velocity (IMU input)
  Eigen::Vector3d a;      // linear acceleration (IMU input)

  double stamp;

  State() : stamp(-1.0) { 
    Config& cfg = Config::getInstance();
                                                                //                  Tangent (idx)
    X = BundleT(manif::SGal3d(0., 0., 0.,                       // x y z                       0
                              0., 0., 0.,                       // roll pitch yaw              6
                              0., 0., 0.,                       // vx, vy, vz                  3
                              0.),                              // delta t                     9
                manif::R3d(cfg.sensors.intrinsics.gyro_bias),   // b_w                        10
                manif::R3d(cfg.sensors.intrinsics.accel_bias),  // b_a                        13
                manif::R3d(Eigen::Vector3d::UnitZ()      
                           * -cfg.sensors.extrinsics.gravity)); // g                          16

    P.setIdentity();
    P *= cfg.ikfom.covariance.initial_cov;

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

    MatrixManif Gx_, Gf_; // Adjoint_X(u)^{-1}, J_r(u)  Sola-18, [https://arxiv.org/abs/1812.01537]
    BundleT X_tmp = X.plus(f(imu.lin_accel, imu.ang_vel) * dt, Gx_, Gf_);
    
    // Update covariance
    MatrixDoF Gx = Gx_.template topLeftCorner<DoF, DoF>();
    Gx.template bottomRightCorner<2,2>() = S2::LogJ_a(g(), g()) 
                                           * S2::composeJ_a(g(), {0., 0., 0.}) 
                                           * S2::ExpJ_b(g(), {0., 0.});

    Eigen::Matrix<double, DoF, BundleT::DoF> left = Eigen::Matrix<double, DoF, BundleT::DoF>::Identity();
    left.bottomRightCorner<2, 3>() = S2::LogJ_a(g(), g());
    Eigen::Matrix<double, BundleT::DoF, DoF> right = Eigen::Matrix<double, BundleT::DoF, DoF>::Identity();
    right.bottomRightCorner<3, 2>() = S2::ExpJ_b(g(), {0., 0.});

    Gf_.template bottomRightCorner<3,3>() = -manif::skew(g().normalized());  

    auto Fx = Gx + left * Gf_ * df_dx(imu) * dt * right; // He-2021, [https://arxiv.org/abs/2102.03804] Eq. (26)
    auto Fw = left * Gf_ * df_dw(imu) * dt;      // He-2021, [https://arxiv.org/abs/2102.03804] Eq. (27)

    P = Fx * P * Fx.transpose() + Fw * Q * Fw.transpose(); 

    X = X_tmp;

    // Save info
    a = imu.lin_accel;
    w = imu.ang_vel;

    stamp = imu.stamp;
  }


  void interpolate_to(const double& t) {
    double dt = t - this->stamp;
    assert(dt >= 0);

    X = X.plus(f(a, w) * dt);
  }


  Tangent f(const Eigen::Vector3d& lin_acc, const Eigen::Vector3d& ang_vel) {

    Tangent u = Tangent::Zero();
    u.element<0>().coeffs() << 0., 0., 0., 
                               lin_acc - b_a() /* -n_a */ + R().transpose()*g(),
                               ang_vel - b_w() /* -n_w */,
                               1.;
    // u.element<2>().coeffs() = n_{b_w} 
    // u.element<3>().coeffs() = n_{b_a}

    return u;
  }

  MatrixManif df_dx(const Imu& imu) {
    MatrixManif out = MatrixManif::Zero();

    // velocity 
    out.block<3, 3>(3,  6) = -R().transpose()*manif::skew(g()) * -R(); // w.r.t R := d(R^-1*g)/dR * d(R^-1)/dR
    out.block<3, 3>(3, 13) = -Eigen::Matrix3d::Identity(); // w.r.t b_a 
    out.block<3, 3>(3, 16) =  R().transpose(); // w.r.t g
    // rotation
    out.block<3, 3>(6, 10) = -Eigen::Matrix3d::Identity(); // w.r.t b_w

    return out;
  }

  MatrixDoF_Noise df_dw(const Imu& imu) {
    // w = (n_w, n_a, n_{b_w}, n_{b_a})
    MatrixDoF_Noise out = MatrixDoF_Noise::Zero();

    out.block<3, 3>( 3, 3) = -Eigen::Matrix3d::Identity(); // w.r.t n_a
    out.block<3, 3>( 6, 0) = -Eigen::Matrix3d::Identity(); // w.r.t n_w
    out.block<3, 3>(10, 6) =  Eigen::Matrix3d::Identity(); // w.r.t n_{b_w}
    out.block<3, 3>(13, 9) =  Eigen::Matrix3d::Identity(); // w.r.t n_{b_a}
    
    return out;
  }

  void update(PointCloudT::Ptr& cloud, charlie::Octree& map) {
PROFC_NODE("update")

    Config& cfg = Config::getInstance();

    if (map.size() == 0)
      return;

    Planes first_planes;
    int query_iters = cfg.ikfom.query_iters;

// OBSERVATION MODEL

    auto h_model = [&](const State& s,
                       Eigen::Matrix<double, Eigen::Dynamic, DoFObs>& H,
                       Eigen::Matrix<double, Eigen::Dynamic,  1>&     z) {

      int N = cloud->size();

      std::vector<bool> chosen(N, false);
      Planes planes(N);

      if (query_iters-- > 0) {
        std::vector<int> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        
        std::for_each(
          std::execution::par_unseq,
          indices.begin(),
          indices.end(),
          [&](int i) {
            PointT pt = cloud->points[i];
            Eigen::Vector3d p = pt.getVector3fMap().cast<double>();
            Eigen::Vector3d g = s.isometry() * s.L2I_isometry() * p; // global coords 

            std::vector<pcl::PointXYZ> neighbors;
            std::vector<float> pointSearchSqDis;
            map.knn(pcl::PointXYZ(g(0), g(1), g(2)),
                    cfg.ikfom.plane.points,
                    neighbors,
                    pointSearchSqDis);
            
            if (neighbors.size() < cfg.ikfom.plane.points 
                or pointSearchSqDis.back() > cfg.ikfom.plane.max_sqrt_dist)
                  return;
            
            Eigen::Vector4d p_abcd = Eigen::Vector4d::Zero();
            if (not estimate_plane(p_abcd, neighbors, cfg.ikfom.plane.plane_threshold))
              return;
            
            chosen[i] = true;
            planes[i] = Plane(p, p_abcd);
          }
        ); // end for_each

        first_planes.clear();

        for (int i = 0; i < N; i++) {
          if (chosen[i])
            first_planes.push_back(planes[i]);        
        }
      }

      H = Eigen::MatrixXd::Zero(first_planes.size(), DoFObs);
      z = Eigen::MatrixXd::Zero(first_planes.size(), 1);

      std::vector<int> indices(first_planes.size());
      std::iota(indices.begin(), indices.end(), 0);

      // For each plane, calculate its derivative and distance
      std::for_each (
        std::execution::par_unseq,
        indices.begin(),
        indices.end(),
        [&](int i) {
          Plane m = first_planes[i];

          // Differentiate w.r.t. SGal3
          Eigen::Matrix<double, 3, manif::SGal3d::DoF> J_s;
          Eigen::Vector3d g = s.X.element<0>().act(s.L2I_isometry() * m.p, J_s);

          H.block<1, manif::SGal3d::DoF>(i, 0) << m.n.head(3).transpose() * J_s;

          z(i) = -dist2plane(m.n, g);
        }
      );

    }; // end h_model

// IESEKF UPDATE

    BundleT   X_predicted = X;
    MatrixDoF P_predicted = P;

    Eigen::Matrix<double, Eigen::Dynamic, DoFObs> H;
    // Eigen::Matrix<double, DoFObs, Eigen::Dynamic> K;

    Eigen::Matrix<double, Eigen::Dynamic, 1>      z;
    MatrixDoF KH;

    double R = cfg.ikfom.lidar_noise;

    int i(0);

    do {
      h_model(*this, H, z); // Update H,z and set K to zeros

      // update P
      MatrixManif J_;
      Eigen::Matrix<double, DoF, 1> dx = X.minus(X_predicted, J_).coeffs().head(DoF);
      dx.tail(2) = S2::Log(g(), X_predicted.element<3>().coeffs());

      MatrixDoF J = J_.topLeftCorner(DoF, DoF);
      J.bottomLeftCorner(2, 2) = S2::LogJ_a(g(), X_predicted.element<3>().coeffs()) * S2::ExpJ_b(g());
      P = J.inverse() * P * J.inverse().transpose();

      Eigen::Matrix<double, DoFObs, DoFObs> HTH = H.transpose() * H / R;
      MatrixDoF P_inv = P.inverse();
      P_inv.block<DoFObs, DoFObs>(0, 0) += HTH;
      P_inv = P_inv.inverse();

      // K = P_inv.block<DoF, DoFObs>(0, 0) * H.transpose() / R;
      Eigen::Matrix<double, DoF, 1> Kz = P_inv.block<DoF, DoFObs>(0, 0) * H.transpose() * z / R;

      KH.setZero();
      KH.block<DoF, DoFObs>(0, 0) = P_inv.block<DoF, DoFObs>(0, 0) * HTH;

      dx = Kz + (KH - MatrixDoF::Identity()) * J.inverse() * dx; 
      
      Tangent tau = Tangent::Zero();
      tau.coeffs().head(BundleT::DoF-3) = dx.head(BundleT::DoF-3);
      X = X.plus(tau);

      g(S2::Exp(g(), dx.tail(2)));

      if ((dx.array().abs() <= cfg.ikfom.tolerance).all())
        break;

    } while(i++ < cfg.ikfom.max_iters);

    X = X;
    P = (MatrixDoF::Identity() - KH) * P;

  }


// Getters
  inline Eigen::Vector3d p()       const { return X.element<0>().translation();             }
  inline Eigen::Matrix3d R()       const { return X.element<0>().quat().toRotationMatrix(); }
  inline Eigen::Quaterniond quat() const { return X.element<0>().quat();                    }
  inline Eigen::Vector3d v()       const { return X.element<0>().linearVelocity();          }
  inline double t()                const { return X.element<0>().t();                       }
  inline Eigen::Vector3d b_w()     const { return X.element<1>().coeffs();                  }
  inline Eigen::Vector3d b_a()     const { return X.element<2>().coeffs();                  }
  inline Eigen::Vector3d g()       const { return X.element<3>().coeffs(); }

  inline Eigen::Isometry3d isometry() const {
    Eigen::Isometry3d T;
    T.linear() = R();
    T.translation() = p();
    return T;
  }

  inline Eigen::Isometry3d L2I_isometry() const {
    return Config::getInstance().sensors.extrinsics.lidar2imu;
  }

  inline Eigen::Isometry3d L2baselink_isometry() const {
    return Config::getInstance().sensors.extrinsics.imu2baselink * L2I_isometry();
  }

// Setters
  void b_w(const Eigen::Vector3d& in) { X.element<1>() = manif::R3d(in); }
  void b_a(const Eigen::Vector3d& in) { X.element<2>() = manif::R3d(in); }
  void g(const Eigen::Vector3d& in)   { X.element<3>() = manif::R3d(in); }

};

typedef boost::circular_buffer<State> States;

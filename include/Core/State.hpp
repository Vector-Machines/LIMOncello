#pragma once

#include <execution>
#include <numeric>
#include <algorithm>
#include <boost/circular_buffer.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <rclcpp/rclcpp.hpp>

#include "Core/Imu.hpp"

#include "Utils/Config.hpp"
#include "Utils/Profiler.hpp"
#include "Utils/PCL.hpp"

#include <manif/manif.h>
#include <manif/SGal3.h>
#include <manif/Bundle.h>
#include <manif/Rn.h>

#include <small_gicp/registration/registration_result.hpp>

struct State
{

  using BundleT = manif::Bundle<double,
                                manif::SGal3, // position & rotation & velocity
                                manif::R3,    // angular bias
                                manif::R3,    // acceleartion bias
                                manif::R3     // gravity
                                >;

  using Tangent = typename BundleT::Tangent;

  static constexpr int DoF = BundleT::DoF;                 // DoF whole state
  static constexpr int DoFNoise = 12;                      // b_w, b_a, n_{b_w}, n_{b_a}
  static constexpr int DoFObs = manif::SGal3<double>::DoF; // DoF obsevation equation

  using ProcessMatrix = Eigen::Matrix<double, DoF, DoF>;
  using MappingMatrix = Eigen::Matrix<double, DoF, DoFNoise>;
  using NoiseMatrix = Eigen::Matrix<double, DoFNoise, DoFNoise>;

  BundleT X;
  ProcessMatrix P;
  NoiseMatrix Q;

  Eigen::Vector3d w; // angular velocity (IMU input)
  Eigen::Vector3d a; // linear acceleration (IMU input)

  double stamp;

  State() : stamp(0.0) {}

  void init()
  {
    Config &cfg = Config::getInstance();
    Eigen::Vector3d zero_vec = Eigen::Vector3d(0., 0., 0.);
    //                   Tanget
    X = BundleT(manif::SGal3d(0., 0., 0.,                                                // x y z                  0
                              0., 0., 0.,                                                // roll pitch yaw         6
                              0., 0., 0.,                                                // vx, vy, vz             3
                              0.),                                                       // delta t                9
                manif::R3d(zero_vec),                                                    // b_w                   10
                manif::R3d(zero_vec),                                                    // b_a                   13
                manif::R3d(Eigen::Vector3d::UnitZ() * -cfg.sensors.extrinsics.gravity)); // g                     16

    P.setIdentity();
    P *= 1e-3f;

    w.setZero();
    a.setZero();

    // Control signal noise (never changes)
    Q.setZero();

    Q.block<3, 3>(0, 0) = cfg.ikfom.covariance.gyro * Eigen::Matrix3d::Identity();       // n_w
    Q.block<3, 3>(3, 3) = cfg.ikfom.covariance.accel * Eigen::Matrix3d::Identity();      // n_a
    Q.block<3, 3>(6, 6) = cfg.ikfom.covariance.bias_gyro * Eigen::Matrix3d::Identity();  // n_{b_w}
    Q.block<3, 3>(9, 9) = cfg.ikfom.covariance.bias_accel * Eigen::Matrix3d::Identity(); // n_{b_a}
  }

  void predict(const Imu &imu, const double &dt)
  {
    PROFC_NODE("predict")

    ProcessMatrix Gx, Gf; // Adjoint_X(u)^{-1}, J_r(u)  Sola-18, [https://arxiv.org/abs/1812.01537]
    BundleT X_tmp = X.plus(f(imu.lin_accel, imu.ang_vel, dt) * dt, Gx, Gf);

    // Update covariance
    ProcessMatrix Fx = Gx + Gf * df_dx(imu, dt) * dt; // He-2021, [https://arxiv.org/abs/2102.03804] Eq. (26)
    MappingMatrix Fw = Gf * df_dw(imu, dt) * dt;      // He-2021, [https://arxiv.org/abs/2102.03804] Eq. (27)

    P = Fx * P * Fx.transpose() + Fw * Q * Fw.transpose();

    X = X_tmp;

    // Save info
    a = imu.lin_accel;
    w = imu.ang_vel;

    stamp = imu.stamp;
  }

  void predict(const double &t)
  {
    Config &cfg = Config::getInstance();

    double dt = t - this->stamp;
    if (dt < 0)
      dt = 1. / cfg.sensors.imu.hz;

    X = X.plus(f(a, w, dt) * dt);
  }

  Tangent f(const Eigen::Vector3d &lin_acc, const Eigen::Vector3d &ang_vel, const double &dt)
  {

    Tangent u = Tangent::Zero();
    u.element<0>().coeffs() << 0., 0., 0.,
        lin_acc - b_a() /* -n_a */ + R().transpose() * g(),
        ang_vel - b_w() /* -n_w */,
        1;
    // u.element<1>().coeffs() = n_{b_w}
    // u.element<2>().coeffs() = n_{b_a}

    return u;
  }

  ProcessMatrix df_dx(const Imu &imu, const double &dt)
  {
    ProcessMatrix out = ProcessMatrix::Zero();

    // velocity
    out.block<3, 3>(3, 6) = -R().transpose() * manif::skew(g()) * -R(); // w.r.t R := d(R^-1*g)/dR * d(R^-1)/dR
    out.block<3, 3>(3, 13) = -Eigen::Matrix3d::Identity();              // w.r.t b_a
    out.block<3, 3>(3, 16) = R().transpose();                           // w.r.t g

    // rotation
    out.block<3, 3>(6, 10) = -Eigen::Matrix3d::Identity(); // w.r.t b_w

    return out;
  }

  MappingMatrix df_dw(const Imu &imu, const double &dt)
  {
    // w = (n_w, n_a, n_{b_w}, n_{b_a})
    MappingMatrix out = MappingMatrix::Zero();

    out.block<3, 3>(3, 3) = -Eigen::Matrix3d::Identity(); // w.r.t n_a
    out.block<3, 3>(6, 0) = -Eigen::Matrix3d::Identity(); // w.r.t n_w
    out.block<3, 3>(10, 6) = Eigen::Matrix3d::Identity(); // w.r.t n_{b_w}
    out.block<3, 3>(13, 9) = Eigen::Matrix3d::Identity(); // w.r.t n_{b_a}

    return out;
  }

  void update(const small_gicp::RegistrationResult &result)
  {
    PROFC_NODE("update_vgicp")

    Config &cfg = Config::getInstance();

    if (!result.converged)
    {
      RCLCPP_WARN(rclcpp::get_logger("state"), "Registration did not converge, skipping state update");
      return;
    }

    // Extract transform from registration result
    Eigen::Isometry3d T_measured = result.T_target_source;

    // Convert to SE(3) measurement vector [translation, rotation_axis_angle]
    Eigen::Vector3d t_measured = T_measured.translation();
    Eigen::Matrix3d R_measured = T_measured.rotation();

    // Convert rotation matrix to axis-angle representation
    Eigen::AngleAxisd angle_axis(R_measured);
    Eigen::Vector3d r_measured = angle_axis.angle() * angle_axis.axis();

    // Construct measurement vector z = [t_x, t_y, t_z, r_x, r_y, r_z]
    Eigen::Matrix<double, 6, 1> z;
    z.head<3>() = t_measured;
    z.tail<3>() = r_measured;

    // Predicted measurement from current state
    Eigen::Affine3d T_predicted_affine = affine3d() * I2L_affine3d();
    Eigen::Vector3d t_predicted = T_predicted_affine.translation();
    Eigen::Matrix3d R_predicted = T_predicted_affine.rotation();
    Eigen::AngleAxisd angle_axis_pred(R_predicted);
    Eigen::Vector3d r_predicted = angle_axis_pred.angle() * angle_axis_pred.axis();

    Eigen::Matrix<double, 6, 1> h;
    h.head<3>() = t_predicted;
    h.tail<3>() = r_predicted;

    // Innovation (measurement residual)
    Eigen::Matrix<double, 6, 1> innovation = z - h;

    // Measurement Jacobian H = d(h)/d(x) where x is the state vector
    // For SE(3) pose measurement, this maps state perturbations to measurement space
    Eigen::Matrix<double, 6, DoFObs> H = Eigen::Matrix<double, 6, DoFObs>::Zero();

    // Translation part: direct mapping from state position
    H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

    // Rotation part: mapping from state rotation
    H.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();

    // Measurement covariance (from registration Hessian)
    // The registration Hessian H is the information matrix, so R = H^(-1)
    Eigen::Matrix<double, 6, 6> R_measurement;
    if (result.H.determinant() > 1e-12)
    {
      R_measurement = result.H.inverse();
    }
    else
    {
      // Fallback to diagonal covariance if Hessian is singular
      R_measurement = Eigen::Matrix<double, 6, 6>::Identity() * cfg.ikfom.lidar_noise;
      RCLCPP_WARN(rclcpp::get_logger("state"), "Registration Hessian is singular, using fallback covariance");
    }

    // Kalman filter update
    Eigen::Matrix<double, 6, 6> S = H * P.block<DoFObs, DoFObs>(0, 0) * H.transpose() + R_measurement;
    Eigen::Matrix<double, DoFObs, 6> K = P.block<DoFObs, DoFObs>(0, 0) * H.transpose() * S.inverse();

    // Update state
    Eigen::Matrix<double, DoFObs, 1> dx = K * innovation;

    // Apply state update using manifold operations
    Tangent state_dx = Tangent::Zero();
    state_dx.coeffs().head<DoFObs>() = dx;
    X = X.plus(state_dx);

    // Update covariance
    Eigen::Matrix<double, DoFObs, DoFObs> I_KH = ProcessMatrix::Identity().block<DoFObs, DoFObs>(0, 0) - K * H;
    P.block<DoFObs, DoFObs>(0, 0) = I_KH * P.block<DoFObs, DoFObs>(0, 0);
  }

  // Getters
  inline Eigen::Vector3d p() const { return X.element<0>().translation(); }
  inline Eigen::Matrix3d R() const { return X.element<0>().quat().toRotationMatrix(); }
  inline Eigen::Quaterniond quat() const { return X.element<0>().quat(); }
  inline Eigen::Vector3d v() const { return X.element<0>().linearVelocity(); }
  inline double t() const { return X.element<0>().t(); }
  inline Eigen::Vector3d b_w() const { return X.element<1>().coeffs(); }
  inline Eigen::Vector3d b_a() const { return X.element<2>().coeffs(); }
  inline Eigen::Vector3d g() const { return X.element<3>().coeffs(); }

  inline Eigen::Affine3d affine3d() const
  {
    Eigen::Affine3d T;
    T.linear() = R();
    T.translation() = p();
    return T;
  }

  inline Eigen::Affine3d I2L_affine3d() const
  {
    return Config::getInstance().sensors.extrinsics.lidar2baselink_T;
  }

  // Covariance getters for ROS publishing
  inline Eigen::Matrix<double, 6, 6> pose_covariance() const
  {
    // Extract pose covariance (position + rotation) from full state covariance
    // State indices: [0-2: position, 3-5: velocity, 6-8: rotation, 9: time, 10-12: b_w, 13-15: b_a, 16-18: g]
    // ROS pose covariance order: [x, y, z, roll, pitch, yaw]
    Eigen::Matrix<double, 6, 6> pose_cov = Eigen::Matrix<double, 6, 6>::Zero();

    // Position covariance (x, y, z)
    pose_cov.block<3, 3>(0, 0) = P.block<3, 3>(0, 0);

    // Rotation covariance (roll, pitch, yaw)
    pose_cov.block<3, 3>(3, 3) = P.block<3, 3>(6, 6);

    // Cross-correlations between position and rotation
    pose_cov.block<3, 3>(0, 3) = P.block<3, 3>(0, 6);
    pose_cov.block<3, 3>(3, 0) = P.block<3, 3>(6, 0);

    return pose_cov;
  }

  inline Eigen::Matrix<double, 6, 6> twist_covariance() const
  {
    // Extract twist covariance (linear velocity + angular velocity) from full state covariance
    // State indices: [0-2: position, 3-5: velocity, 6-8: rotation, 9: time, 10-12: b_w, 13-15: b_a, 16-18: g]
    // ROS twist covariance order: [vx, vy, vz, wx, wy, wz]
    Eigen::Matrix<double, 6, 6> twist_cov = Eigen::Matrix<double, 6, 6>::Zero();

    // Linear velocity covariance (vx, vy, vz)
    twist_cov.block<3, 3>(0, 0) = P.block<3, 3>(3, 3);

    // Angular velocity covariance includes gyro bias uncertainty
    // Since angular velocity = w - b_w, we need to propagate uncertainty from both w and b_w
    // For simplicity, we'll use the gyro bias covariance as the angular velocity uncertainty
    // This is a conservative estimate since the raw gyro measurement uncertainty is not stored
    twist_cov.block<3, 3>(3, 3) = P.block<3, 3>(10, 10);

    // Cross-correlations between linear and angular velocity are typically small
    // and not directly available in our state representation, so we leave them as zero

    return twist_cov;
  }

  // Setters
  void b_w(const Eigen::Vector3d &in) { X.element<1>() = manif::R3d(in); }
  void b_a(const Eigen::Vector3d &in) { X.element<2>() = manif::R3d(in); }
  void g(const Eigen::Vector3d &in) { X.element<3>() = manif::R3d(in); }
};

typedef boost::circular_buffer<State> States;

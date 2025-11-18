# LIMOncello

Another tightly coupled LiDAR‑Inertial SLAM algorithm? Yep, based on the existing algorithms
[FAST-LIO2](https://github.com/hku-mars/FAST_LIO), [LIMO-Velo](https://github.com/Huguet57/LIMO-Velo),
[Fast LIMO](https://github.com/fetty31/fast_LIMO) and
[DLIO](https://github.com/vectr-ucla/direct_lidar_inertial_odometry), but cleaner, faster, and
without [IKFoM](https://github.com/hku-mars/IKFoM)/[ikd-Tree](https://github.com/hku-mars/ikd-Tree)
dependencies.

LIMOncello is essentially FAST-LIO 2 with the Fast LIMO (and thus DLIO) implementation, but
without any dependency on the [IKFoM](https://github.com/hku-mars/IKFoM) or
[ikd-Tree](https://github.com/hku-mars/ikd-Tree) C++ libraries. It implements an Iterated Error
State Extended Kalman Filter (IESEKF) as described in IKFoM and FAST-LIO 2, reimplemented using
the [manif](https://github.com/artivis/manif) library and a refactored version of the original
[iOctree](https://github.com/zhujun3753/i-octree) data structure.

This is the result of my efforts to truly understand LIMO‑Velo and, consequently, FAST-LIO 2.
Those insights led to an improvement in performance and accuracy: LIMOncello is the first known
LIO‑SLAM system to incorporate the SGal(3) manifold in its state representation, and it uses
iOctree, which is much faster and more efficient than iKd-Tree. On top of that, the code is
portable, not as Fast LIMO, but as simplified as possible (the entire IKFoM library is
synthesized into `State.hpp` thanks to [manif](https://github.com/artivis/manif), for example)
so it’s accessible to anyone who wants to modify it.

## Dependencies

LIMOncello is header-only and the core depends only on:
- [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page)
- [PCL 1.8](https://pointclouds.org/)
- [oneTBB](https://github.com/uxlfoundation/oneTBB)
- [manif](https://github.com/artivis/manif), needs to be installed [see](https://artivis.github.io/manif/tutorial/cpp/) 
## TODOs

LIMOncello is still in progress, but the core is considered finished and working with ROS1.

- [X] Test with LiVOX
- [X] Test with Hesai
- [ ] Incorporate GPS correction
- [ ] Equivariant EKF
- [ ] Iterated Equivariant EKF

## Approach

To truly understand the concepts, here are the papers that greatly influenced this work, apart
from Fast LIMO:
- [IKFoM](https://arxiv.org/abs/2102.03804)
- [FAST-LIO](https://arxiv.org/abs/2010.08196) and [FAST-LIO2](https://arxiv.org/abs/2107.06829)
- [Quaternion kinematics for the error-state Kalman filter](https://arxiv.org/abs/1711.02508)
- [A micro Lie theory for state estimation in robotics](https://arxiv.org/abs/1812.01537)
- [Absolute humanoid localization and mapping based on IMU Lie group and fiducial markers](https://digital.csic.es/handle/10261/206456)
- [iOctree](https://arxiv.org/pdf/2309.08315)

The most important differences between these approaches and LIMOncello are that the latter does
not implement the S2 manifold and does not include extrinsics as part of the state (they remain
fixed). Nonetheless, LIMOncello was originally designed for Formula Student race cars and has not
yet been tested on KITTI or other common benchmarks.

I should also mention [S-FAST_LIO](https://github.com/zlwang7/S-FAST_LIO), which helped me grasp
the initial intuition behind the math of IKFoM, and [manif](https://github.com/artivis/manif),
which provides accessible, elegant Lie algebra tools for nearly any manifold used in robotics.



## Configuration

Here, the configuration file for `LIMOncello` is explained.

| Parameter                        | Units    | Summary                                                                               |
|----------------------------------|----------|---------------------------------------------------------------------------------------|
| topics/input/lidar               | –        | ROS topic for incoming LiDAR point cloud                                              |
| topics/input/imu                 | –        | ROS topic for incoming IMU data                                                       |
| topics/input/stop_ioctree_update | –        | ROS topic to trigger stopping of the iOctree updates (std_msgs/Bool)                  |
| topics/output/state              | –        | ROS topic for published state estimates                                               |
| topics/output/frame              | –        | ROS topic for published full point cloud frame                                        |
| frame_id                         | –        | Coordinate frame ID used for all published data                                       |
| verbose                          | –        | Display execution time board                                                          |
| debug                            | –        | Publish intermediate point‑clouds (deskewed, processed, …) for visualization          |
| sensors/lidar/type               | –        | LiDAR model type (0 = OUSTER, 1 = VELODYNE, 2 = HESAI, 3 = LIVOX)                     |
| sensors/lidar/end_of_sweep       | –        | Whether sweep timestamp refers to end (true) or start (false) of scan                 |
| sensors/imu/hz                   | Hz       | IMU data rate                                                                         |
| calibration/gravity_align        | –        | If true, estimate gravity vector while robot is stationary                            |
| calibration/accel                | –        | If true, estimate linear accelerometer bias while robot is stationary                 |
| calibration/gyro                 | –        | If true, estimate gyroscope bias while robot is stationary                            |
| calibration/time                 | s        | Duration for which robot must remain stationary to perform above calibrations         |
| time_offset                      | –        | Whether to account for sync offset between IMU and LiDAR                              |
| TAI_offset                       | s        | Time offset (TAI) to apply to point cloud timestamps                                  |
| extrinsics/imu2CoG/t             | m        | Translation of IMU relative to the Center of Gravity (CoG)                            |
| extrinsics/imu2CoG/R             | deg      | Rotation (roll, pitch, yaw) of IMU relative to the Center of Gravity (CoG)            |
| extrinsics/lidar2imu/t           | m        | Translation of LiDAR relative to the IMU                                              |
| extrinsics/lidar2imu/R           | deg      | Rotation (roll, pitch, yaw) of LiDAR relative to the IMU                              |
| extrinsics/gravity               | m/s²     | Default gravity magnitude if no gravity calibration is performed                      |
| intrinsics/accel_bias            | m/s²     | Default accelerometer bias vector                                                     |
| intrinsics/gyro_bias             | rad/s    | Default gyroscope bias vector                                                         |
| intrinsics/sm                    | –        | Sensor‑to‑standard‑axis mapping matrix                                                |
| filters/voxel_grid/leaf_size     | m        | Voxel‑grid leaf size                                                                  |
| filters/min_distance/active      | –        | Enable minimum‑distance (sphere) crop                                                 |
| filters/min_distance/value       | m        | Radius for min‑distance crop                                                          |
| filters/fov/active               | –        | Enable field‑of‑view crop                                                             |
| filters/fov/value                | deg      | Field‑of‑view angle                                                                   |
| filters/rate_sampling/active     | –        | Enable simple rate‑based downsampling                                                 |
| filters/rate_sampling/value      | –        | Take one out of every *value* points                                                  |
| IKFoM/query_iters                | –        | Number of the first KNN‑map queries among all IESEKF updates                          |
| IKFoM/max_iters                  | –        | Maximum number of IESEKF updates                                                      |
| IKFoM/tolerance                  | –        | Convergence tolerance for IESEKF                                                      |
| IKFoM/lidar_noise                | –        | LiDAR measurement noise parameter                                                     |
| IKFoM/covariance/gyro            | rad²     | Gyroscope measurement covariance                                                      |
| IKFoM/covariance/accel           | m²/s²    | Accelerometer measurement covariance                                                  |
| IKFoM/covariance/bias_gyro       | rad/s·√s | Gyroscope bias covariance                                                             |
| IKFoM/covariance/bias_accel      | m²/s²·√s | Accelerometer bias covariance                                                         |
| IKFoM/plane/points               | –        | Number of points used to fit each plane feature                                       |
| IKFoM/plane/max_sqrt_dist        | m        | Maximum distance from query to any point in the plane (if exceeded, plane is invalid) |
| IKFoM/plane/plane_threshold      | m        | Maximum distance from any point to its plane to be considered valid                   |
| iOctree/min_extent               | m        | Minimum cell size in octree                                                           |
| iOctree/bucket_size              | –        | Maximum points per octree leaf                                                        |
| iOctree/downsample               | –        | Whether to downsample when inserting into the octree                                  |

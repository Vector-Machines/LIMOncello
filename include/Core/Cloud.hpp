#pragma once  

#include <vector>
#include <algorithm>
#include <execution>

#include <boost/circular_buffer.hpp>

#include "Core/State.hpp"
#include "Utils/PCL.hpp"
#include "Utils/Profiler.hpp"
#include "Utils/Config.hpp"


States filter_states(const States& states, const double& start, const double& end) {

  States out(750); // Always initialize circular buffer !!

  for (const auto& state : states) {
    if (state.stamp >= end)
      continue;

    if (state.stamp >= start)
      out.push_front(state);
    
    if (state.stamp < start) {
      out.push_front(state);
      break;
    }
  }

  return out;
}


PointCloudT::Ptr deskew(const PointCloudT::Ptr& cloud,
                        const State& state,
                        const States& buffer,
                        const double& offset,
                        const double& sweep_time) {
  
PROFC_NODE("deskew")

  auto binary_search = [&](const double& t) {
    int l(0), r(buffer.size()-1);
    
    while (l < r) {
      int m = (l + r) / 2;
      if (buffer[m].stamp == t)
        return m;
      else if (t < buffer[m].stamp)
        r = m - 1;
      else
        l = m + 1;
    }

    return l-1 > 0 ? l-1 : l;
  };


  PointTime point_time = point_time_func();

  PointCloudT::Ptr out(new PointCloudT);
  out->points.resize(cloud->points.size());

  std::vector<int> indices(cloud->points.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::for_each(
    std::execution::par_unseq,
    indices.begin(),
    indices.end(),
    [&](int k) {
      int i_f = binary_search(point_time(cloud->points[k], sweep_time) + offset);

      State X0 = buffer[i_f];
      X0.predict(point_time(cloud->points[k], sweep_time) + offset);

      Eigen::Affine3f T0 = (X0.affine3d() * X0.I2L_affine3d()).cast<float>();
      Eigen::Affine3f TN = (state.affine3d() * state.I2L_affine3d()).cast<float>();

      Eigen::Vector3f p;  
      p << cloud->points[k].x, cloud->points[k].y, cloud->points[k].z;

      p = TN.inverse() * T0 * p;

      PointT pt;
      pt.x = p.x();
      pt.y = p.y();
      pt.z = p.z();
      pt.intensity = cloud->points[k].intensity;

      out->points[k] = pt;
    }
  );

  return out;
}


PointCloudT::Ptr process(const PointCloudT::Ptr& cloud) {

PROFC_NODE("filter")


  Config& cfg = Config::getInstance();

  PointCloudT::Ptr out(new PointCloudT);

  int index = 0;
  std::copy_if(
    cloud->points.begin(), 
    cloud->points.end(), 
    std::back_inserter(out->points), 
    [&](const PointT& p) mutable {
        bool pass = true;
        Eigen::Vector3f p_; 
        p_ = cfg.sensors.extrinsics.lidar2baselink_T.cast<float>() * p.getVector3fMap();

        // Crop box filter
        if (cfg.filters.crop_box.active)
        {
          if (p.x >= cfg.filters.crop_box.min_pt.x() && p.x <= cfg.filters.crop_box.max_pt.x() &&
              p.y >= cfg.filters.crop_box.min_pt.y() && p.y <= cfg.filters.crop_box.max_pt.y() &&
              p.z >= cfg.filters.crop_box.min_pt.z() && p.z <= cfg.filters.crop_box.max_pt.z())
          {
            pass = false;
          }
        }

        // Distance filter
        if (pass and cfg.filters.min_distance.active)
        {
          if (Eigen::Vector3f(p.x, p.y, p.z).squaredNorm() <= cfg.filters.min_distance.value * cfg.filters.min_distance.value)
            pass = false;
        }

        // Rate filter
        if (pass and cfg.filters.rate_sampling.active) {
          if (index % cfg.filters.rate_sampling.value != 0)
              pass = false;
        }

        // Field of view filter
        if (pass and cfg.filters.fov.active) {
          if (fabs(atan2(p_.y(), p_.x())) >= cfg.filters.fov.value)
              pass = false;
        }

        ++index; // Increment index

        return pass;
    }
  );

  return out;
}


PointCloudT::Ptr voxel_grid(const PointCloudT::Ptr& cloud) {

PROFC_NODE("downsample")

  Config& cfg = Config::getInstance();

  static pcl::VoxelGrid<PointT> filter;
  filter.setLeafSize(cfg.filters.voxel_grid.leaf_size, 
                     cfg.filters.voxel_grid.leaf_size, 
                     cfg.filters.voxel_grid.leaf_size);

  PointCloudT::Ptr out(new PointCloudT);
  filter.setInputCloud(cloud);
  filter.filter(*out);

  return out;
}

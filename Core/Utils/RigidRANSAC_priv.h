#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <random>

Eigen::Isometry3f fit(const Eigen::MatrixX3f &p0, const Eigen::MatrixX3f &p1, const RigidRANSAC::VectorXb &mask = {});

Eigen::VectorXf apply(const Eigen::Isometry3f &T, const Eigen::MatrixX3f &p0, const Eigen::MatrixX3f &p1);

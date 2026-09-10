#include "tiny_api.hpp"

#include <Eigen/Dense>

namespace {

constexpr int kStateCount = 2;
constexpr int kInputCount = 1;
constexpr int kHorizon = 100;
constexpr double kDt = 0.01;
constexpr double kMaxYawAcceleration = 50.0;

TinySolver * yaw_solver = nullptr;

int setup_solver()
{
  if (yaw_solver != nullptr) return 0;

  Eigen::MatrixXd A{{1, kDt}, {0, 1}};
  Eigen::MatrixXd B{{0}, {kDt}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::MatrixXd Q = Eigen::Vector2d(9e6, 0).asDiagonal();
  Eigen::MatrixXd R = Eigen::VectorXd::Ones(1).asDiagonal();

  int status = tiny_setup(
    &yaw_solver, A, B, f, Q, R, 1.0, kStateCount, kInputCount, kHorizon, 0);
  if (status != 0) return status;

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(kStateCount, kHorizon, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(kStateCount, kHorizon, 1e17);
  Eigen::MatrixXd u_min =
    Eigen::MatrixXd::Constant(kInputCount, kHorizon - 1, -kMaxYawAcceleration);
  Eigen::MatrixXd u_max =
    Eigen::MatrixXd::Constant(kInputCount, kHorizon - 1, kMaxYawAcceleration);
  status = tiny_set_bound_constraints(yaw_solver, x_min, x_max, u_min, u_max);
  yaw_solver->settings->max_iter = 10;
  return status;
}

}  // namespace

extern "C" __declspec(dllexport) int sp_solve_yaw(
  const double * reference_yaw, const double * reference_velocity, double initial_yaw,
  double initial_velocity, double * output_yaw, double * output_velocity,
  double * output_acceleration)
{
  int status = setup_solver();
  if (status != 0) return status;

  for (int i = 0; i < kHorizon; ++i) {
    yaw_solver->work->Xref(0, i) = reference_yaw[i];
    yaw_solver->work->Xref(1, i) = reference_velocity[i];
  }

  Eigen::Vector2d x0(initial_yaw, initial_velocity);
  tiny_set_x0(yaw_solver, x0);
  status = tiny_solve(yaw_solver);

  for (int i = 0; i < kHorizon; ++i) {
    output_yaw[i] = yaw_solver->work->x(0, i);
    output_velocity[i] = yaw_solver->work->x(1, i);
  }
  for (int i = 0; i < kHorizon - 1; ++i) {
    output_acceleration[i] = yaw_solver->work->u(0, i);
  }
  return status;
}

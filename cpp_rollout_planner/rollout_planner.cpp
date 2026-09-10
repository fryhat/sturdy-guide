#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

extern "C" {

constexpr double kDt = 0.05;
constexpr double kLaunchDelay = 0.015;
constexpr double kGimbalMaxAccel = 50.0;
constexpr double kGimbalMaxOmega = 1.2617046468719202;
constexpr int kSteps = 6;
constexpr int kAlphas = 5;
constexpr int kAlphasEarly = 7;
constexpr int kMaxAlphas = 7;
constexpr double kPi = 3.14159265358979323846;
constexpr double kArmorWidth = 0.140;
constexpr double kArmorDrawThickness = 0.035;
constexpr double kScatterSigma = (kArmorWidth / 2.0) / 1.6448536269514722;
constexpr double kBulletSpeed = 25.0;
constexpr double kMinNormalHitSpeed = 12.0;
constexpr double kPlannerDt = 0.010;

struct PruneCounters {
    int nodes = 0;
    int edge_only_layers = 0;
    int full_layers = 0;
};

constexpr double kAlpha7[] = {
    -kGimbalMaxAccel,
    -kGimbalMaxAccel * 2.0 / 3.0,
    -kGimbalMaxAccel / 3.0,
    0.0,
    kGimbalMaxAccel / 3.0,
    kGimbalMaxAccel * 2.0 / 3.0,
    kGimbalMaxAccel
};

constexpr double kAlpha5[] = {
    -kGimbalMaxAccel,
    -kGimbalMaxAccel / 2.0,
    0.0,
    kGimbalMaxAccel / 2.0,
    kGimbalMaxAccel
};

static double interpolate(
    double yaw,
    const double* grid,
    const double* values,
    int grid_size
) {
    if (grid_size <= 0) return 0.0;
    if (grid_size == 1) return values[0];
    if (yaw <= grid[0] || yaw >= grid[grid_size - 1]) return 0.0;
    int lo = 0;
    int hi = grid_size - 1;
    while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        if (grid[mid] <= yaw) lo = mid;
        else hi = mid;
    }
    double fraction = (yaw - grid[lo]) / (grid[hi] - grid[lo]);
    return values[lo] + fraction * (values[hi] - values[lo]);
}

static double wrap_angle(double angle) {
    angle = std::fmod(angle + kPi, 2.0 * kPi);
    if (angle < 0.0) angle += 2.0 * kPi;
    return angle - kPi;
}

static void advance_limited(
    double& yaw,
    double& omega,
    double alpha,
    double dt
) {
    double limit = kGimbalMaxOmega;
    if (alpha > 0.0 && omega >= limit) {
        yaw += omega * dt;
        return;
    }
    if (alpha < 0.0 && omega <= -limit) {
        yaw += omega * dt;
        return;
    }
    double boundary = alpha > 0.0 ? limit : -limit;
    double time_to_boundary = (boundary - omega) / alpha;
    if (time_to_boundary < 0.0) time_to_boundary = 0.0;
    if (time_to_boundary < dt) {
        double reached = omega + alpha * time_to_boundary;
        yaw += omega * time_to_boundary + 0.5 * alpha * time_to_boundary * time_to_boundary;
        yaw += reached * (dt - time_to_boundary);
        omega = reached;
    } else {
        yaw += omega * dt + 0.5 * alpha * dt * dt;
        omega += alpha * dt;
    }
}

struct ArmorView {
    double x;
    double y;
    double normal_yaw;
};

static void armor_views_for_angle(
    double angle,
    double center_x,
    double center_y,
    ArmorView* out
) {
    const double local[4][6] = {
        {0.0, -0.3, 1.0, 0.0, 0.0, -1.0},
        {0.255, 0.0, 0.0, 1.0, 1.0, 0.0},
        {0.0, 0.3, -1.0, 0.0, 0.0, 1.0},
        {-0.255, 0.0, 0.0, -1.0, -1.0, 0.0}
    };
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    for (int i = 0; i < 4; ++i) {
        double cx = local[i][0];
        double cy = local[i][1];
        double nx = local[i][4];
        double ny = local[i][5];
        double rx = cx * cosine - cy * sine;
        double ry = cx * sine + cy * cosine;
        double rnx = nx * cosine - ny * sine;
        double rny = nx * sine + ny * cosine;
        out[i].x = center_x + rx;
        out[i].y = center_y + ry;
        out[i].normal_yaw = std::atan2(rny, rnx);
    }
}

static double angle_error(double target, double current) {
    double value = std::fmod(target - current + kPi, 2.0 * kPi);
    if (value < 0.0) value += 2.0 * kPi;
    return value - kPi;
}

static void build_one_curve(
    double impact_angle,
    double impact_omega,
    double center_x,
    double center_y,
    double horizon,
    double omega_limit,
    double alpha_limit,
    const double* entries,
    int entry_count,
    const double* grid,
    int grid_size,
    double* out_curve
) {
    if (omega_limit <= 0.0 || alpha_limit <= 0.0) {
        for (int j = 0; j < grid_size; ++j) out_curve[j] = 0.0;
        return;
    }
    double lower_accel = (-omega_limit - impact_omega) / horizon;
    if (lower_accel < -alpha_limit) lower_accel = -alpha_limit;
    double upper_accel = (omega_limit - impact_omega) / horizon;
    if (upper_accel > alpha_limit) upper_accel = alpha_limit;

    std::vector<double> angles;
    std::vector<int> counts;
    for (int i = 0; i < entry_count; ++i) {
        double sample_horizon = entries[i * 3];
        double acceleration = entries[i * 3 + 1];
        int count = static_cast<int>(entries[i * 3 + 2]);
        if (count <= 0) continue;
        if (std::fabs(sample_horizon - horizon) > kPlannerDt) continue;
        if (acceleration < lower_accel - 1e-12 || acceleration > upper_accel + 1e-12) continue;
        angles.push_back(impact_angle + 0.5 * acceleration * horizon * horizon);
        counts.push_back(count);
    }

    if (angles.empty()) {
        for (int j = 0; j < grid_size; ++j) out_curve[j] = 0.0;
        return;
    }

    const double radius = std::sqrt(center_x * center_x + center_y * center_y);
    if (radius <= 0.0) {
        for (int j = 0; j < grid_size; ++j) out_curve[j] = 0.0;
        return;
    }

    std::vector<std::array<ArmorView, 4>> armors(angles.size());
    for (std::size_t i = 0; i < angles.size(); ++i) {
        armor_views_for_angle(angles[i], center_x, center_y, armors[i].data());
    }

    const double half_angle = std::atan2(kArmorWidth / 2.0, radius);
    const double scale = (kScatterSigma / radius) * std::sqrt(2.0);
    int total_count = 0;
    for (int count : counts) total_count += count;

    for (int j = 0; j < grid_size; ++j) {
        double yaw = grid[j];
        double total = 0.0;
        for (std::size_t i = 0; i < angles.size(); ++i) {
            double best = 0.0;
            for (int a = 0; a < 4; ++a) {
                const ArmorView& armor = armors[i][a];
                double armor_yaw = std::atan2(armor.y, armor.x);
                double normal_alignment = std::max(
                    0.0,
                    -std::cos(angle_error(yaw, armor.normal_yaw))
                );
                if (normal_alignment * kBulletSpeed <= kMinNormalHitSpeed) continue;
                double delta = angle_error(yaw, armor_yaw);
                double probability = 0.5 * (
                    std::erf((delta + half_angle) / scale)
                    - std::erf((delta - half_angle) / scale)
                );
                if (probability > best) best = probability;
            }
            total += static_cast<double>(counts[i]) * best;
        }
        out_curve[j] = total / static_cast<double>(total_count);
    }
}

static void search(
    int step,
    double yaw,
    double omega,
    double score_so_far,
    int grid_size,
    const double* grid,
    const double* curves,
    double first_alpha,
    double& best_score,
    double& best_alpha,
    PruneCounters& counters
) {
    if (step >= kSteps) {
        if (score_so_far > best_score + 1e-12) {
            best_score = score_so_far;
            best_alpha = first_alpha;
        }
        return;
    }

    counters.nodes += 1;
    const double* curve = curves + step * grid_size;
    const int candidate_count = step < 3 ? kAlphasEarly : kAlphas;
    const double* candidate_set = step < 3 ? kAlpha7 : kAlpha5;
    double probability[kMaxAlphas];
    double alpha[kMaxAlphas];
    double child_first_alpha[kMaxAlphas];
    double next_omega[kMaxAlphas];
    double next_yaw[kMaxAlphas];
    bool all_zero = true;
    for (int i = 0; i < candidate_count; ++i) {
        alpha[i] = candidate_set[i];
        double launch_yaw = yaw;
        double launch_omega = omega;
        advance_limited(launch_yaw, launch_omega, alpha[i], kLaunchDelay);
        launch_yaw = wrap_angle(launch_yaw);
        probability[i] = interpolate(launch_yaw, grid, curve, grid_size);
        if (probability[i] >= 1e-12) all_zero = false;
        child_first_alpha[i] = step == 0 ? alpha[i] : first_alpha;
        next_omega[i] = omega;
        next_yaw[i] = wrap_angle(yaw);
        advance_limited(next_yaw[i], next_omega[i], alpha[i], kDt);
    }

    if (all_zero) {
        counters.edge_only_layers += 1;
    } else {
        counters.full_layers += 1;
    }

    if (all_zero) {
        const int edges[2] = {0, candidate_count - 1};
        for (int edge : edges) {
            double child_score = score_so_far + probability[edge];
            search(
                step + 1,
                next_yaw[edge],
                next_omega[edge],
                child_score,
                grid_size,
                grid,
                curves,
                child_first_alpha[edge],
                best_score,
                best_alpha,
                counters
            );
        }
    } else {
        for (int i = 0; i < candidate_count; ++i) {
            double child_score = score_so_far + probability[i];
            search(
                step + 1,
                next_yaw[i],
                next_omega[i],
                child_score,
                grid_size,
                grid,
                curves,
                child_first_alpha[i],
                best_score,
                best_alpha,
                counters
            );
        }
    }
}

static void plan_impl(
    double yaw,
    double omega,
    const double* grid,
    int grid_size,
    const double* curves,
    int curve_count,
    double* out_alpha,
    double* out_score,
    PruneCounters* out_counters
) {
    double best_score = 0.0;
    double best_alpha = 0.0;
    PruneCounters counters;
    search(
        0,
        yaw,
        omega,
        0.0,
        grid_size,
        grid,
        curves,
        0.0,
        best_score,
        best_alpha,
        counters
    );
    *out_alpha = best_alpha;
    *out_score = best_score;
    if (out_counters) *out_counters = counters;
}

__declspec(dllexport) int sp_plan_rollout(
    double yaw,
    double omega,
    const double* grid,
    int grid_size,
    const double* curves,
    int curve_count,
    double* out_alpha,
    double* out_score
) {
    if (!grid || !curves || grid_size <= 0 || curve_count != kSteps || !out_alpha || !out_score) {
        return -1;
    }
    plan_impl(yaw, omega, grid, grid_size, curves, curve_count, out_alpha, out_score, nullptr);
    return 0;
}

__declspec(dllexport) int sp_plan_rollout_debug(
    double yaw,
    double omega,
    const double* grid,
    int grid_size,
    const double* curves,
    int curve_count,
    double* out_alpha,
    double* out_score,
    int* out_nodes,
    int* out_edge_only,
    int* out_full
) {
    if (!grid || !curves || grid_size <= 0 || curve_count != kSteps || !out_alpha || !out_score) {
        return -1;
    }
    PruneCounters counters;
    plan_impl(yaw, omega, grid, grid_size, curves, curve_count, out_alpha, out_score, &counters);
    *out_nodes = counters.nodes;
    *out_edge_only = counters.edge_only_layers;
    *out_full = counters.full_layers;
    return 0;
}

__declspec(dllexport) int sp_build_curve_from_impact_center(
    double impact_angle,
    double impact_omega,
    double center_x,
    double center_y,
    double horizon,
    double omega_limit,
    double alpha_limit,
    const double* entries,
    int entry_count,
    const double* grid,
    int grid_size,
    double* out_curve
) {
    if (!grid || !entries || entry_count < 0 || grid_size <= 0 || !out_curve) return -1;
    if (omega_limit <= 0.0 || alpha_limit <= 0.0) return -1;
    build_one_curve(
        impact_angle,
        impact_omega,
        center_x,
        center_y,
        horizon,
        omega_limit,
        alpha_limit,
        entries,
        entry_count,
        grid,
        grid_size,
        out_curve
    );
    return 0;
}

__declspec(dllexport) int sp_hit_probability_from_impact_limit_center(
    double impact_angle,
    double impact_omega,
    double center_x,
    double center_y,
    double horizon,
    double omega_limit,
    double alpha_limit,
    double launch_yaw,
    const double* entries,
    int entry_count,
    double* out_probability
) {
    if (!entries || entry_count < 0 || !out_probability) return -1;
    if (omega_limit <= 0.0 || alpha_limit <= 0.0) return -1;
    double yaw_grid[1] = {launch_yaw};
    double result[1] = {0.0};
    build_one_curve(
        impact_angle,
        impact_omega,
        center_x,
        center_y,
        horizon,
        omega_limit,
        alpha_limit,
        entries,
        entry_count,
        yaw_grid,
        1,
        result
    );
    *out_probability = result[0];
    return 0;
}

__declspec(dllexport) int sp_build_curve_from_impact(
    double impact_angle,
    double impact_omega,
    double horizon,
    double omega_limit,
    double alpha_limit,
    const double* entries,
    int entry_count,
    const double* grid,
    int grid_size,
    double* out_curve
) {
    return sp_build_curve_from_impact_center(
        impact_angle,
        impact_omega,
        0.0,
        6.0,
        horizon,
        omega_limit,
        alpha_limit,
        entries,
        entry_count,
        grid,
        grid_size,
        out_curve
    );
}

__declspec(dllexport) int sp_hit_probability_from_impact_limit(
    double impact_angle,
    double impact_omega,
    double horizon,
    double omega_limit,
    double alpha_limit,
    double launch_yaw,
    const double* entries,
    int entry_count,
    double* out_probability
) {
    return sp_hit_probability_from_impact_limit_center(
        impact_angle,
        impact_omega,
        0.0,
        6.0,
        horizon,
        omega_limit,
        alpha_limit,
        launch_yaw,
        entries,
        entry_count,
        out_probability
    );
}

}

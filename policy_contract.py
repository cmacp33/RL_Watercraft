"""Shared observation and action conversions for sim and real-boat policies."""

import math

import numpy as np


POSITION_SCALE_M = 15.0
FORWARD_SPEED_SCALE_MPS = 3.0
YAW_RATE_SCALE_RADPS = 2.0
ESP32_MAX_COMMAND = 255


def policy_observation(
    x_m,
    y_m,
    yaw_rad,
    velocity_x_mps,
    velocity_y_mps,
    yaw_rate_radps,
    goal_x_m,
    goal_y_m,
    position_scale_m=POSITION_SCALE_M,
):
    """Return the normalized 6-value body-relative observation.

    Values are goal-forward, goal-left, heading-error sine/cosine,
    forward speed, and yaw rate. Heading and yaw rate are in radians.
    """
    values = np.asarray(
        [x_m, y_m, yaw_rad, velocity_x_mps, velocity_y_mps,
         yaw_rate_radps, goal_x_m, goal_y_m],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("Pose, velocity, and goal values must be finite")
    if position_scale_m <= 0:
        raise ValueError("position_scale_m must be positive")

    dx = goal_x_m - x_m
    dy = goal_y_m - y_m
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    goal_forward_m = cos_yaw * dx + sin_yaw * dy
    goal_left_m = -sin_yaw * dx + cos_yaw * dy
    heading_error = math.atan2(dy, dx) - yaw_rad
    forward_speed_mps = cos_yaw * velocity_x_mps + sin_yaw * velocity_y_mps

    observation = np.array(
        [
            goal_forward_m / position_scale_m,
            goal_left_m / position_scale_m,
            math.sin(heading_error),
            math.cos(heading_error),
            forward_speed_mps / FORWARD_SPEED_SCALE_MPS,
            yaw_rate_radps / YAW_RATE_SCALE_RADPS,
        ],
        dtype=np.float32,
    )
    return np.clip(observation, -1.0, 1.0)


class PoseObservationAdapter:
    """Convert successive marker poses in centimeters/degrees to policy state."""

    def __init__(self, position_scale_m):
        if position_scale_m <= 0:
            raise ValueError("position_scale_m must be positive")
        self.position_scale_m = position_scale_m
        self.previous_pose = None

    def reset(self):
        self.previous_pose = None

    def update(self, x_cm, y_cm, yaw_deg, timestamp_s, goal_x_cm, goal_y_cm):
        """Return normalized observation from a water-frame marker pose."""
        x_m = x_cm / 100.0
        y_m = y_cm / 100.0
        yaw_rad = math.radians(yaw_deg)
        velocity_x_mps = 0.0
        velocity_y_mps = 0.0
        yaw_rate_radps = 0.0

        if self.previous_pose is not None:
            prev_x_m, prev_y_m, prev_yaw_rad, prev_time_s = self.previous_pose
            dt = timestamp_s - prev_time_s
            if dt > 0:
                velocity_x_mps = (x_m - prev_x_m) / dt
                velocity_y_mps = (y_m - prev_y_m) / dt
                yaw_delta = math.atan2(
                    math.sin(yaw_rad - prev_yaw_rad),
                    math.cos(yaw_rad - prev_yaw_rad),
                )
                yaw_rate_radps = yaw_delta / dt

        self.previous_pose = (x_m, y_m, yaw_rad, timestamp_s)
        return policy_observation(
            x_m,
            y_m,
            yaw_rad,
            velocity_x_mps,
            velocity_y_mps,
            yaw_rate_radps,
            goal_x_cm / 100.0,
            goal_y_cm / 100.0,
            position_scale_m=self.position_scale_m,
        )


def normalized_action(action):
    """Validate and clip a left/right policy action to [-1, 1]."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (2,):
        raise ValueError("Action must contain exactly [left, right]")
    if not np.all(np.isfinite(action)):
        raise ValueError("Action values must be finite")
    return np.clip(action, -1.0, 1.0)


def gazebo_thrust_action(action):
    """Convert a policy action to Gazebo plugin command values."""
    return normalized_action(action)


def esp32_motor_action(action, max_command=ESP32_MAX_COMMAND):
    """Convert normalized left/right actions to ESP32 integer commands."""
    if max_command <= 0:
        raise ValueError("max_command must be positive")
    return np.rint(normalized_action(action) * max_command).astype(np.int16)
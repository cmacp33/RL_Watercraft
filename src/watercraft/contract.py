"""Shared observation and action conversions for sim and real-boat policies."""

import math
from typing import NamedTuple

import numpy as np
from watercraft.config import load
_POLICY = load("policy")
_REAL = load("deploy")


# Shared by sim training and the real boat; a policy only works at the scale it
# was trained with. Must exceed the max goal offset or observations clip.
POSITION_SCALE_M = _POLICY["position_scale_m"]
GOAL_RADIUS_M = load("task")["goal_radius_m"]
FORWARD_SPEED_SCALE_MPS = _POLICY["forward_speed_scale_mps"]
YAW_RATE_SCALE_RADPS = _POLICY["yaw_rate_scale_radps"]
ESP32_MAX_COMMAND = 255


class MotorCalibration(NamedTuple):
    """Real-boat corrections so that policy 'left'/'right' mean the physical thrusters.

    The policy (and Gazebo) assume identical thrusters with left/right in the boat's
    frame. Apply this only on the hardware path, never in sim.
    """
    swap_channels: bool = False  # True: ESP32 leftSpeed channel drives the physical RIGHT thruster
    left_gain: float = 1.0       # multiplier for the physical left thruster
    right_gain: float = 1.0      # multiplier for the physical right thruster
    deadband: int = 0            # smallest command that spins a motor; non-zero commands start here
    max_command: int = ESP32_MAX_COMMAND  # command where thrust stops increasing; |action| = 1 maps here
    min_action: float = 0.0      # |action| below this sends 0, so tiny outputs don't jump to the deadband


NO_CALIBRATION = MotorCalibration()

# Measured on the real boat (bench test 2026-11). Usable range is ~40..90 for both
# motors: below 40 nothing spins, above 90 thrust doesn't increase.
# Policy |action| (0, 1] maps linearly onto [40, 90].
REAL_BOAT_CALIBRATION = MotorCalibration(**_REAL["motor_calibration"])


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
    """
    Takes in observation in world coordinates.
    Return the normalized 6-value body-relative observation, in boat coordinates.

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

    # with respect to boat's own axis
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


def esp32_motor_action(action, calibration=REAL_BOAT_CALIBRATION):
    """Convert a normalized [left, right] policy action to ESP32 [leftSpeed, rightSpeed] commands.

    Per physical thruster: gain -> clip to [-1, 1] -> |x| < min_action gives 0, otherwise
    |x| maps linearly onto [deadband, max_command]. Then the pair is mapped onto ESP32 channels.
    """
    cal = calibration
    if not 0 < cal.max_command <= ESP32_MAX_COMMAND:
        raise ValueError("max_command must be in (0, %d]" % ESP32_MAX_COMMAND)
    if not 0 <= cal.deadband < cal.max_command:
        raise ValueError("deadband must be in [0, max_command)")
    left, right = normalized_action(action)
    commands = []
    for value, gain in ((left, cal.left_gain), (right, cal.right_gain)):
        scaled = float(np.clip(value * gain, -1.0, 1.0))
        if abs(scaled) < cal.min_action or scaled == 0.0:
            commands.append(0.0)
        else:
            commands.append(math.copysign(
                cal.deadband + abs(scaled) * (cal.max_command - cal.deadband), scaled))
    if cal.swap_channels:
        commands.reverse()
    return np.rint(commands).astype(np.int16)

class NumpyPolicy:
    """Run a trained policy with numpy only (no torch needed on the camera machine).

    Loads the .npz written by training_scripts/export_policy.py (or by training itself):
    the 6 -> hidden -> 2 MLP from gazebo_pool_v0_xentropy.Net, with tanh on the output,
    exactly as run_model.py does.
    """

    def __init__(self, path):
        with np.load(path) as weights:
            self.w0, self.b0 = weights['w0'], weights['b0']
            self.w1, self.b1 = weights['w1'], weights['b1']
        if self.w0.shape[1] != 6 or self.w1.shape[0] != 2:
            raise ValueError("Expected a 6-input, 2-output policy, got %s -> %s"
                             % (self.w0.shape, self.w1.shape))

    def act(self, observation):
        """Deterministic [left, right] action in [-1, 1] for a normalized observation."""
        hidden = np.maximum(0.0, self.w0 @ np.asarray(observation, dtype=np.float32) + self.b0)
        return np.tanh(self.w1 @ hidden + self.b1).astype(np.float32)

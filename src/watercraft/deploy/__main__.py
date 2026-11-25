"""Run a trained policy (or the P controller) on the real boat; settings: configs/deploy.json.

Measure a rectangle from four known-size ArUco markers.

IDs: 0 bottom-left, 1 bottom-right, 2 top-right, 3 top-left.
Place markers on one flat plane. Width/height are distances between marker
centers, not outside edges. At the rectangle center, +X follows ID0 -> ID1,
+Y follows ID0 -> ID3, and +Z is the right-handed perpendicular X cross Y.
Camera coordinates follow OpenCV: X right, Y down, Z away from the camera.
All physical lengths and translation vectors are in meters.
Only marker size and camera intrinsics provide metric scale; no rectangle
width or height is assumed. The camera and rectangle must stay fixed after
calibration; press R to recalibrate if either moves. Press Q or Esc to quit.
"""
import json
import math
import time
from pathlib import Path
import cv2
import numpy as np
import serial

from watercraft import contract
from watercraft.contract import NumpyPolicy, POSITION_SCALE_M, policy_observation
from watercraft.deploy.motors import send_motor_command, send_policy_action, stop_motors
from watercraft.deploy.vision import (detect_and_draw_markers, find_origin, rectangle_from_centers,
                                      locate_aruco_4, draw_origin_axes, display_id4_rotation, print_id4_pose,
                                      open_camera, fit_for_display)

# Serial and controller settings (no command-line arguments).
from watercraft.config import load as load_config, path as config_path
_SETTINGS = load_config("deploy")

SERIAL_PORT = _SETTINGS['serial_port']
BAUD_RATE = _SETTINGS['baud_rate']
CONTROL_PERIOD_S = _SETTINGS['control_period_s']
DRY_RUN = not _SETTINGS["arm"]
CONTROLLER = _SETTINGS['controller']
POLICY_WEIGHTS = config_path(_SETTINGS["policy_weights"])
TARGET_XY_M = tuple(_SETTINGS["goal"])
# Goal heading in degrees, counterclockwise from origin +X (ID0 -> ID1). Used by policies trained with
# policy.heading_mode = 'target'; 'bearing' policies (older runs) and the P controller ignore it.
GOAL_HEADING_DEG = _SETTINGS.get("goal_heading_deg")
# Set from the run's saved config.json by use_trained_policy_settings().
HEADING_MODE = 'bearing'
GOAL_HEADING_TOL_DEG = 180.0
STOP_DISTANCE_M = _SETTINGS["goal_radius_m"]
HEADING_TOL_DEG = _SETTINGS['heading_tolerance_deg']
FORWARD_CMD = _SETTINGS['forward_command']
TURN_CMD_MIN = _SETTINGS['turn_command_min']
TURN_CMD_MAX = _SETTINGS['turn_command_max']
TURN_KP = _SETTINGS['turn_kp']
MAX_ACTION = _SETTINGS['max_action']
# Angle from the boat marker's +X axis to the bow, counterclockwise seen from above (the camera).
# 0: bow along marker +X.  90: bow along marker +Y.  -90: bow along marker -Y.
HEADING_OFFSET_RAD = math.radians(_SETTINGS.get('heading_offset_deg', 0.0))

CAM_INDEX = _SETTINGS['camera_index']
CORNER_IDS = list(_SETTINGS["corner_ids"])
TARGET_ID = _SETTINGS['target_id']
MARKER_LENGTH_M = _SETTINGS['marker_length_m']
N_CALIB_FRAMES = _SETTINGS['calibration_frames']
AXIS_LENGTH_M = _SETTINGS['axis_length_m']
PRINT_FULL_TRANSFORMS = _SETTINGS['print_full_transforms']
CALIB_FILE = str(config_path(_SETTINGS["calibration_file"]))

# One row per control tick, to compare the real boat with the sim step logs.
REAL_LOG_FIELDS = ['time_s', 'dt_s', 'mode', 'x_m', 'y_m', 'yaw_deg', 'dist_m', 'heading_error_deg',
                   'obs_goal_forward', 'obs_goal_left', 'obs_sin_heading', 'obs_cos_heading',
                   'obs_forward_speed', 'obs_yaw_rate', 'model_left', 'model_right',
                   'sent_left', 'sent_right', 'esp_left', 'esp_right', 'hold_steps',
                   'goal_heading_deg', 'goal_heading_error_deg']


def open_real_log():
    """runs/<run>/evaluations/real/<timestamp>.csv (+ .json with the deploy settings)."""
    from watercraft.logging import CsvLog, evaluation_files
    weights = Path(POLICY_WEIGHTS)
    run = weights.parent.parent if weights.parent.name == 'checkpoints' else weights.parent
    if CONTROLLER != 'policy' or not run.is_dir():
        run = config_path('runs') / 'p_controller'
    csv_path, json_path = evaluation_files(run, 'real')
    json_path.write_text(json.dumps(dict(_SETTINGS, position_scale_m=POSITION_SCALE_M,
                                         forward_speed_scale_mps=contract.FORWARD_SPEED_SCALE_MPS,
                                         yaw_rate_scale_radps=contract.YAW_RATE_SCALE_RADPS,
                                         heading_mode=HEADING_MODE,
                                         dry_run=DRY_RUN), indent=2))
    print('Real-boat log:', csv_path)
    return CsvLog(csv_path, REAL_LOG_FIELDS)


def use_trained_policy_settings(weights):
    """Use the observation scales and heading mode saved with the run that produced the weights."""
    global POSITION_SCALE_M, HEADING_MODE, GOAL_HEADING_TOL_DEG
    for folder in (Path(weights).parent, Path(weights).parent.parent):
        saved = folder / 'config.json'
        if saved.is_file():
            saved = json.loads(saved.read_text())
            policy = saved.get('policy')
            if policy:
                POSITION_SCALE_M = policy['position_scale_m']
                contract.FORWARD_SPEED_SCALE_MPS = policy['forward_speed_scale_mps']
                contract.YAW_RATE_SCALE_RADPS = policy['yaw_rate_scale_radps']
                HEADING_MODE = policy.get('heading_mode', 'bearing')
                if HEADING_MODE == 'target':
                    GOAL_HEADING_TOL_DEG = saved['task']['goal_heading_tolerance_deg']
            else:  # historical runs recorded only the position scale
                POSITION_SCALE_M = saved.get('position_scale_m', POSITION_SCALE_M)
            return


def compute_motor_command_to_target(x_m, y_m, yaw_deg, target_xy_m=TARGET_XY_M):
    """Return physical left/right controller requests and diagnostics.

    These requests are normalized and calibrated before serial transmission.
    Positive yaw turns from origin +X toward +Y. Marker +X must face the
    boat's forward direction, as in track_aruco_v3.py.
    """
    dx, dy = np.asarray(target_xy_m, dtype=float) - [x_m, y_m]
    distance_m = math.hypot(dx, dy)
    target_heading_deg = math.degrees(math.atan2(dy, dx))
    error_deg = (target_heading_deg - yaw_deg + 180.0) % 360.0 - 180.0
    if distance_m <= STOP_DISTANCE_M:
        left, right, mode = 0, 0, 'stop'
    elif abs(error_deg) > HEADING_TOL_DEG:
        turn = int(np.clip(TURN_KP * abs(error_deg), TURN_CMD_MIN, TURN_CMD_MAX))
        left, right, mode = ((-turn, turn, 'turn_ccw') if error_deg > 0
                             else (turn, -turn, 'turn_cw'))
    else:
        left, right, mode = FORWARD_CMD, FORWARD_CMD, 'forward'
    return left, right, {'distance_m': distance_m,
                         'heading_error_deg': error_deg, 'mode': mode}


class MotorController:
    """Meter-based control with the same action/observation contract as v3.

    Pose and goal stay in the centered rectangle frame. The policy receives
    body-relative observations, so no bottom-left-origin offset is needed.
    """
    def __init__(self):
        if CONTROLLER not in ('p', 'policy'):
            raise ValueError("CONTROLLER must be 'p' or 'policy'")
        if not 0 < MAX_ACTION <= 1 or CONTROL_PERIOD_S <= 0:
            raise ValueError('MAX_ACTION must be in (0, 1] and control period positive')
        if np.asarray(TARGET_XY_M).shape != (2,) or not np.isfinite(TARGET_XY_M).all():
            raise ValueError('TARGET_XY_M must contain two finite meter coordinates')
        if CONTROLLER == 'policy' and POLICY_WEIGHTS is None:
            raise ValueError('Set POLICY_WEIGHTS to a policy .npz for CONTROLLER=policy')
        if CONTROLLER == 'policy':
            use_trained_policy_settings(POLICY_WEIGHTS)
            if HEADING_MODE == 'target' and (GOAL_HEADING_DEG is None or not np.isfinite(GOAL_HEADING_DEG)):
                raise ValueError('This policy was trained with a goal heading: set goal_heading_deg '
                                 'in configs/deploy.json (degrees counterclockwise from +X)')
        self.goal_heading_rad = (math.radians(GOAL_HEADING_DEG)
                                 if CONTROLLER == 'policy' and HEADING_MODE == 'target' else None)
        self.policy = NumpyPolicy(POLICY_WEIGHTS) if CONTROLLER == 'policy' else None
        self.previous_pose = None
        self.last_control = float('-inf')
        self.channels = (0, 0)
        self.mode = 'calibration -> stop'
        self.log = None
        self.start = None
        self.bow_heading_deg = None
        self.hold_count = 0

    def reset(self, ser):
        self.previous_pose = None
        self.hold_count = 0
        self.last_control = float('-inf')
        self.channels = send_policy_action(ser, np.zeros(2))
        self.mode = 'recalibration -> stop'

    def update(self, ser, pose, calibrated, now):
        """Use the current frame only; resend commands every control tick."""
        valid = (calibrated and pose is not None and
                 pose['heading_angle_rad'] is not None and
                 np.isfinite(pose['position_origin_m']).all() and
                 np.isfinite(pose['heading_angle_rad']))
        if not valid:
            self.previous_pose = None
            self.hold_count = 0
        if now - self.last_control < CONTROL_PERIOD_S:
            return
        dt = now - self.last_control
        self.last_control = now
        row = {}
        if not valid:
            action = np.zeros(2)
            self.mode = ('calibration -> stop' if not calibrated
                         else 'marker/heading unavailable -> stop')
        else:
            x, y = pose['position_origin_m'][:2]
            # bow heading = marker +X heading + mounting offset, wrapped to [-pi, pi]
            yaw = pose['heading_angle_rad'] + HEADING_OFFSET_RAD
            yaw = math.atan2(math.sin(yaw), math.cos(yaw))
            self.bow_heading_deg = math.degrees(yaw)
            vx = vy = yaw_rate = 0.0
            if self.previous_pose is not None:
                px, py, pyaw, previous_time = self.previous_pose
                dt = now - previous_time
                if dt > 0:
                    vx, vy = (x - px) / dt, (y - py) / dt
                    yaw_rate = math.atan2(math.sin(yaw - pyaw), math.cos(yaw - pyaw)) / dt
            self.previous_pose = (x, y, yaw, now)
            observation = policy_observation(
                x, y, yaw, vx, vy, yaw_rate, *TARGET_XY_M,
                position_scale_m=POSITION_SCALE_M, goal_heading_rad=self.goal_heading_rad)
            left, right, debug = compute_motor_command_to_target(
                x, y, math.degrees(yaw), TARGET_XY_M)
            # Steps in a row inside the goal radius (and, for goal-heading policies, within the
            # heading tolerance), as the sim counts them for task.goal_hold_steps.
            inside = debug['distance_m'] <= STOP_DISTANCE_M
            goal_heading_error_deg = ''
            if self.goal_heading_rad is not None:
                error = self.goal_heading_rad - yaw
                goal_heading_error_deg = math.degrees(math.atan2(math.sin(error), math.cos(error)))
                inside = inside and abs(goal_heading_error_deg) <= GOAL_HEADING_TOL_DEG
            self.hold_count = self.hold_count + 1 if inside else 0
            if self.policy is not None:
                # The policy keeps control inside the radius: hold runs (task.goal_hold_steps > 1)
                # were trained to brake and stay there, which cutting the motors would prevent.
                action = self.policy.act(observation)
                self.mode = f'policy | holding {self.hold_count}' if self.hold_count else 'policy'
            elif debug['distance_m'] <= STOP_DISTANCE_M:
                action, self.mode = np.zeros(2), 'at goal -> stop'
            else:
                action = np.array([left, right]) / max(FORWARD_CMD, TURN_CMD_MAX)
                self.mode = 'p:' + debug['mode']
            row = dict(x_m=float(x), y_m=float(y), yaw_deg=math.degrees(yaw), dist_m=debug['distance_m'],
                       hold_steps=self.hold_count,
                       goal_heading_deg=GOAL_HEADING_DEG if self.goal_heading_rad is not None else '',
                       goal_heading_error_deg=goal_heading_error_deg,
                       heading_error_deg=debug['heading_error_deg'],
                       model_left=float(action[0]), model_right=float(action[1]),
                       **dict(zip(REAL_LOG_FIELDS[8:14], map(float, observation))))
            action = np.clip(action, -MAX_ACTION, MAX_ACTION)
        self.channels = send_policy_action(ser, action)
        if self.log is not None:
            if self.start is None:
                self.start = now
            self.log.write(time_s=now - self.start, dt_s=dt if np.isfinite(dt) else '', mode=self.mode,
                           sent_left=float(action[0]), sent_right=float(action[1]),
                           esp_left=self.channels[0], esp_right=self.channels[1], **row)


def main():
    if MARKER_LENGTH_M <= 0:
        raise ValueError('MARKER_LENGTH_M must be positive')
    with np.load(CALIB_FILE) as calibration:
        K, D = calibration['K'], calibration['D']
    zero_distortion = np.zeros_like(D)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    controller = MotorController()
    controller.log = open_real_log()
    ser, capture = None, None
    samples, geometry, preview = [], None, None
    try:
        if not DRY_RUN:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1, write_timeout=0.1)
            time.sleep(1)  # Allow the ESP32 to reset after opening USB serial.
            send_motor_command(ser, 0, 0)
        print(f'Controller: {CONTROLLER}; target: {TARGET_XY_M} m; '
              f'{"DRY RUN" if DRY_RUN else SERIAL_PORT}')
        capture = open_camera(CAM_INDEX)
        print('Keep all four corner markers visible and still. R recalibrates; Q or Esc quits.')
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError('Cannot read camera frame.')
            print(frame.shape)
            # Keep the same K after undistortion; all subsequent poses use D=0.
            undistorted = cv2.undistort(frame, K, D)
            vis, found = detect_and_draw_markers(undistorted, detector)
            if geometry is None:
                measured = find_origin(found, K, zero_distortion, MARKER_LENGTH_M)
                if measured is not None:
                    samples.append(measured['corners'])
                    preview = rectangle_from_centers(np.median(samples, axis=0))
                if len(samples) >= N_CALIB_FRAMES:
                    geometry = rectangle_from_centers(np.median(samples, axis=0))
                    print(f"Width: {geometry['width']:.4f} m; height: {geometry['height']:.4f} m")
                    print(f"Origin in camera coordinates (m): {geometry['center']}")
                    print('T_camera_from_origin (translation in m):')
                    print(geometry['T_camera_from_origin'])
                    print('T_origin_from_camera (inverse; translation in m):')
                    print(geometry['T_origin_from_camera'])
                missing = [i for i in CORNER_IDS if i not in found]
                status = f'Calibration: {len(samples)}/{N_CALIB_FRAMES}'
                if missing:
                    status += f' | missing IDs: {missing}'
                elif measured is None:
                    status += ' | marker pose failed'
                else:
                    status += ' | preliminary estimate'
                cv2.putText(vis, status, (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.putText(vis, f'Detected IDs: {sorted(found)}', (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            # Show estimates immediately while collecting the calibration samples.
            displayed_geometry = geometry if geometry is not None else preview
            id4_pose = None
            if displayed_geometry is not None:
                draw_origin_axes(vis, displayed_geometry, K, zero_distortion)
                cv2.putText(vis, f"Width {displayed_geometry['width']:.4f} m  Height {displayed_geometry['height']:.4f} m",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                id4_pose = locate_aruco_4(found, displayed_geometry, K, zero_distortion)
                if id4_pose is not None:
                    display_id4_rotation(vis, id4_pose)
                    print_id4_pose(id4_pose)
                    T_camera_from_id4 = id4_pose['T_camera_from_id4']
                    id4_rvec, _ = cv2.Rodrigues(T_camera_from_id4[:3, :3])
                    cv2.drawFrameAxes(vis, K, zero_distortion, id4_rvec,
                                      T_camera_from_id4[:3, 3], MARKER_LENGTH_M / 2, 2)
                    xyz = id4_pose['position_origin_m']
                    cv2.putText(vis, f'ID4: X {xyz[0]:.4f} Y {xyz[1]:.4f} Z {xyz[2]:.4f} m',
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    angle = id4_pose['heading_angle_deg']
                    angle_text = ('ID4 angle from +X: undefined (heading normal to plane)'
                                  if angle is None else f'ID4 angle from +X: {angle:+.2f} deg')
                    cv2.putText(vis, angle_text, (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(vis, f"ID4 inside rectangle: {id4_pose['inside_rectangle']}",
                                (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(vis, 'ID4 not detected or pose unavailable', (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            controller.update(ser, id4_pose, geometry is not None, time.monotonic())
            bow = ('bow --' if controller.bow_heading_deg is None
                   else f'bow {controller.bow_heading_deg:+.0f} deg')
            cv2.putText(vis, f'{controller.mode} | {bow} | ESP L={controller.channels[0]} R={controller.channels[1]}',
                        (10, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            target_text = f'Target: {TARGET_XY_M} m'
            if controller.goal_heading_rad is not None:
                target_text += f', heading {GOAL_HEADING_DEG:+.0f} deg (+/-{GOAL_HEADING_TOL_DEG:.0f})'
            cv2.putText(vis, target_text, (10, 370),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.putText(vis, 'Q/Esc: quit | R: recalibrate',
                        (10, vis.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
            cv2.imshow('ArUco measured rectangle', fit_for_display(vis))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            if key in (ord('r'), ord('R')):
                controller.reset(ser)
                samples, geometry, preview = [], None, None
    finally:
        try:
            stop_motors(ser)
        finally:
            controller.log.close()
            if capture is not None:
                capture.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

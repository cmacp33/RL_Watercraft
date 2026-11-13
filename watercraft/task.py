"""Metric task calculation for physical trials; Gazebo keeps its existing task."""
import math


def reward(previous, current, config):
    value = (config['distance_weight'] * (previous['distance'] - current['distance'])
             + config['heading_weight'] * (abs(previous['heading']) - abs(current['heading']))
             - config['step_penalty'])
    return value + (config['goal_bonus'] if current['reached'] else 0.0)


def metrics(x, y, yaw, goal, radius):
    dx, dy = goal[0] - x, goal[1] - y
    heading = math.atan2(dy, dx) - yaw
    distance = math.hypot(dx, dy)
    return {'distance': distance, 'heading': math.atan2(math.sin(heading), math.cos(heading)),
            'reached': distance <= radius}

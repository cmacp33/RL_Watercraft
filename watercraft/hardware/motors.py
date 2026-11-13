"""Shared calibrated physical motor transport; no device opens on import."""
import time
import numpy as np
from watercraft.contract import MotorCalibration, esp32_motor_action, normalized_action


class Motors:
    def __init__(self, config, serial_factory=None):
        self.config = config
        self.serial = None
        self.calibration = MotorCalibration(**config['motor_calibration'])
        self.last_channels = (0, 0)
        if not config['arm']:
            raise ValueError('Set arm: true in configs/real.json before a physical trial')
        if serial_factory is None:
            import serial
            serial_factory = serial.Serial
        self.serial = serial_factory(config['serial_port'], config['baud_rate'], timeout=1,
                                     write_timeout=config['serial_write_timeout_s'])
        try:
            time.sleep(config['serial_startup_wait_s'])
            self.stop()
        except BaseException:
            self.serial.close()
            raise

    def send_channels(self, left, right):
        payload = f'{int(left)} {int(right)}\n'.encode('ascii')
        if self.serial.write(payload) != len(payload):
            raise OSError('Incomplete motor command write')
        self.last_channels = (int(left), int(right))
        return self.last_channels

    def act(self, action):
        action = np.clip(normalized_action(action), -self.config['max_action'], self.config['max_action'])
        return self.send_channels(*esp32_motor_action(action, self.calibration))

    def stop(self):
        if self.serial is not None:
            self.send_channels(0, 0)

    def close(self):
        if self.serial is not None:
            try:
                self.stop()
            finally:
                self.serial.close()
                self.serial = None

"""Serial motor commands for the ESP32 transmitter; no port opens on import."""
import time
import serial
from watercraft.contract import esp32_motor_action


def send_motor_command(ser, motor_left, motor_right):
    """Send calibrated ESP32 channels as newline-terminated ASCII integers."""
    if ser is not None:
        payload = f'{int(motor_left)} {int(motor_right)}\n'.encode('ascii')
        written = ser.write(payload)
        if written != len(payload):
            raise serial.SerialException('Incomplete motor command write')
        print(f'Serial TX: {payload!r}', flush=True)


def send_policy_action(ser, action):
    """Apply shared v3 motor gains, deadband, limits, and channel swap."""
    left, right = map(int, esp32_motor_action(action))
    send_motor_command(ser, left, right)
    return left, right


def stop_motors(ser):
    """Repeat the stop command before closing, even after a serial failure."""
    if ser is None:
        return
    try:
        for _ in range(3):
            try:
                send_motor_command(ser, 0, 0)
            except (serial.SerialException, OSError) as error:
                print(f'Failed to send motor stop: {error}')
            time.sleep(0.05)
    finally:
        ser.close()

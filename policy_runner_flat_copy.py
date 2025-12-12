#!/usr/bin/env python3
"""
Run the flat-ground locomotion policy (model_flat) in MuJoCo/Unitree simulation.

Key features
- Loads ONNX policy from model_flat/exported/policy.onnx
- Supports two velocity sources for the policy input (`--estimate`):
    * sport   : use SportModeState linear velocity (legacy)
    * kalman  : use Kalman-filtered velocity estimate
- Always runs the Kalman filter in the background and logs comparison against
  SportModeState to aid estimator tuning.
- Uses the same observation layout as the 48-dim sport policy:
    [base_lin_vel(3), base_ang_vel(3), projected_gravity(3),
     velocity_commands(3), joint_pos(12), joint_vel(12), actions(12)]
"""

import argparse
import sys
import time
from pathlib import Path
from collections import deque
from typing import Optional, Tuple

import numpy as np
import onnxruntime as ort

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, SportModeState_
from unitree_sdk2py.utils.crc import CRC

from processor.observation_processor import ObservationProcessor
from utils.config import PolicyConfig, ISAACLAB_TO_UNITREE


class MovingAverage:
    """Lightweight moving average for smoothing velocity outputs."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)

    def reset(self):
        self.values.clear()

    def update(self, value: np.ndarray) -> np.ndarray:
        self.values.append(np.asarray(value, dtype=np.float32))
        return self.average

    @property
    def average(self) -> np.ndarray:
        if not self.values:
            return np.zeros(3, dtype=np.float32)
        return np.mean(np.stack(self.values, axis=0), axis=0)


class SimpleKalman3D:
    """Minimal 3D Kalman filter (constant velocity with acceleration input)."""

    def __init__(self, process_var: float, meas_var: float, init_var: float):
        self.x = np.zeros(3, dtype=np.float32)
        self.P = np.eye(3, dtype=np.float32) * init_var
        self.Q = np.eye(3, dtype=np.float32) * process_var
        self.R = np.eye(3, dtype=np.float32) * meas_var
        self.H = np.eye(3, dtype=np.float32)
        self.F = np.eye(3, dtype=np.float32)

    def predict(self, u: np.ndarray, dt: float):
        # State prediction: x = x + dt * u
        self.x = self.F @ self.x + dt * u
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray):
        # Innovation
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(3, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P


class Go2VelocityKalman:
    """Kalman filter for base linear velocity estimation on Go2."""

    def __init__(self,
                 process_var: float = 0.2,
                 meas_var: float = 0.05,
                 init_var: float = 0.5,
                 smoothing_window: int = 20):
        self._process_var = process_var
        self._meas_var = meas_var
        self._init_var = init_var
        self.filter = SimpleKalman3D(process_var, meas_var, init_var)

        self.smoother = MovingAverage(window_size=smoothing_window)
        self.last_timestamp = None

        self.prior_err_sum = np.zeros(3, dtype=np.float64)
        self.prior_err_sq_sum = np.zeros(3, dtype=np.float64)
        self.err_samples = 0

    def reset(self):
        self.filter = SimpleKalman3D(self._process_var, self._meas_var, self._init_var)
        self.last_timestamp = None
        self.smoother.reset()
        self.prior_err_sum.fill(0.0)
        self.prior_err_sq_sum.fill(0.0)
        self.err_samples = 0

    @staticmethod
    def _quat_to_world_body(quat: np.ndarray) -> np.ndarray:
        """Return world-to-body rotation matrix (same convention as ObservationProcessor)."""
        w, x, y, z = quat
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)],
            [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
            [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float32)

    def step(self,
             imu_acc: np.ndarray,
             quat: np.ndarray,
             dt: float,
             measurement: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Run one predict/update step.

        Args:
            imu_acc: Raw accelerometer specific force in body frame (m/s^2).
            quat: Quaternion [w, x, y, z].
            dt: Delta time in seconds.
            measurement: Optional body-frame velocity measurement (e.g., SportModeState).

        Returns:
            smoothed_estimate: Smoothed velocity estimate (body frame).
            prior: Prior estimate before measurement update (for logging), None if dt invalid.
        """
        if dt <= 0:
            return self.smoother.average, None

        R_w_b = self._quat_to_world_body(quat)
        gravity_world = np.array([0.0, 0.0, -9.81], dtype=np.float32)
        gravity_body = R_w_b @ gravity_world  # rotate gravity into body frame

        # Convert specific force to actual body-frame linear acceleration
        accel_body = imu_acc + gravity_body

        # Predict step using acceleration as control input
        self.filter.predict(u=accel_body, dt=dt)

        prior = self.filter.x.copy()

        # Measurement update (if available)
        if measurement is not None:
            self.filter.update(measurement)
            # Error stats use prior (pre-update) for insight into estimator drift
            err = prior - measurement
            self.prior_err_sum += err
            self.prior_err_sq_sum += err * err
            self.err_samples += 1

        smoothed = self.smoother.update(self.filter.x.copy())
        return smoothed, prior

    def status_string(self) -> str:
        if self.err_samples == 0:
            return "Kalman stats: waiting for measurements..."
        mean_err = self.prior_err_sum / self.err_samples
        rmse = np.sqrt(self.prior_err_sq_sum / self.err_samples)
        lines = []
        lines.append("Kalman prior error (body frame) [m/s]")
        lines.append("  stat       vx        vy        vz")
        lines.append(
            "  mean   "
            f"{mean_err[0]:+8.4f} {mean_err[1]:+8.4f} {mean_err[2]:+8.4f}"
        )
        lines.append(
            "  rmse   "
            f"{rmse[0]:+8.4f} {rmse[1]:+8.4f} {rmse[2]:+8.4f}"
        )
        return "\n".join(lines)


class FlatPolicyRunner:
    """Minimal runner for the flat-ground policy with velocity estimation toggle."""

    def __init__(self, policy_path: str, estimate_source: str, interface: str, disable_noise: bool):
        self.config = PolicyConfig()
        self.estimate_source = estimate_source
        self.disable_noise = disable_noise
        self.crc = CRC()

        print(f"Loading ONNX policy from: {policy_path}")
        self.ort_session = ort.InferenceSession(policy_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

        self.obs_processor = ObservationProcessor(self.config, ros_node=None, obs_layout="flat")
        self.vel_estimator = Go2VelocityKalman()

        self.latest_lowstate: Optional[LowState_] = None
        self.latest_sportmode: Optional[SportModeState_] = None
        self.last_loop_ts = None

        self._setup_comm(interface)

    def _setup_comm(self, interface: str):
        if interface == "lo":
            ChannelFactoryInitialize(1, "lo")
        else:
            ChannelFactoryInitialize(0, interface)

        self.state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.state_sub.Init(self._lowstate_handler, 10)

        self.sport_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.sport_sub.Init(self._sportmode_handler, 10)

        self.cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.cmd_pub.Init()

        print(f"Communication ready (interface={interface})")

    def _lowstate_handler(self, msg: LowState_):
        self.latest_lowstate = msg

    def _sportmode_handler(self, msg: SportModeState_):
        self.latest_sportmode = msg

    def wait_for_state(self, timeout: float = 5.0):
        start = time.time()
        while self.latest_lowstate is None:
            if time.time() - start > timeout:
                raise TimeoutError("Failed to receive LowState")
            time.sleep(0.001)
        print("Received first LowState.")

    def _compute_policy_velocity(self, lowstate: LowState_, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Run Kalman estimator and pick velocity for policy input."""
        imu_acc = np.array(lowstate.imu_state.accelerometer, dtype=np.float32)
        quat = np.array(lowstate.imu_state.quaternion, dtype=np.float32)

        # Measurement from SportModeState (body frame) if available
        sport_vel = None
        if self.latest_sportmode is not None:
            sport_vel = np.array([
                float(self.latest_sportmode.velocity[0]),
                float(self.latest_sportmode.velocity[1]),
                float(self.latest_sportmode.velocity[2])
            ], dtype=np.float32)

        kalman_vel, _ = self.vel_estimator.step(imu_acc, quat, dt, sport_vel)

        # Pick velocity for policy input
        if self.estimate_source == "kalman" or sport_vel is None:
            policy_vel = kalman_vel.copy()
        else:
            policy_vel = sport_vel.copy()

        return policy_vel.astype(np.float32), kalman_vel.astype(np.float32)

    def _compute_torques(self,
                         target_positions_unitree: np.ndarray,
                         current_positions_unitree: np.ndarray,
                         current_velocities_unitree: np.ndarray) -> np.ndarray:
        pos_err = target_positions_unitree - current_positions_unitree
        vel_err = -current_velocities_unitree
        torques = self.config.KP * pos_err + self.config.KD * vel_err
        return np.clip(torques, -self.config.ACTION_CLIP, self.config.ACTION_CLIP)

    def _build_command(self, torques_unitree: np.ndarray) -> unitree_go_msg_dds__LowCmd_:
        cmd = unitree_go_msg_dds__LowCmd_()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0

        for i in range(20):
            cmd.motor_cmd[i].mode = 0x01
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].tau = float(torques_unitree[i]) if i < 12 else 0.0

        cmd.crc = self.crc.Crc(cmd)
        return cmd

    def _zero_command(self) -> unitree_go_msg_dds__LowCmd_:
        zeros = np.zeros(12, dtype=np.float32)
        return self._build_command(zeros)

    def run(self, duration: float):
        print(f"Running flat policy for {duration:.1f}s | estimate source: {self.estimate_source}")
        self.wait_for_state()

        # Set nominal velocity commands (can be adjusted in code as needed)
        self.obs_processor.set_velocity_commands(-1.0, 0.0, 0.0)

        start_time = time.time()
        step = 0

        try:
            while True:
                loop_start = time.perf_counter()

                if duration and (loop_start - start_time) > duration:
                    break

                if self.latest_lowstate is None:
                    time.sleep(0.001)
                    continue

                dt = self.config.CONTROL_DT if self.last_loop_ts is None else (loop_start - self.last_loop_ts)
                self.last_loop_ts = loop_start

                policy_vel, kalman_vel = self._compute_policy_velocity(self.latest_lowstate, dt)
                # Feed chosen velocity to observation processor
                self.obs_processor.update_base_velocity(policy_vel)

                obs, _ = self.obs_processor.process(self.latest_lowstate, add_noise=not self.disable_noise)

                obs_batch = obs.reshape(1, -1)
                actions_raw = self.ort_session.run([self.output_name], {self.input_name: obs_batch})[0][0]
                actions_raw = actions_raw.astype(np.float32)

                actions_scaled = actions_raw * self.config.ACTION_SCALE
                target_positions = actions_scaled + self.config.DEFAULT_JOINT_POS

                # Reorder to Unitree order for control
                target_positions_unitree = target_positions[ISAACLAB_TO_UNITREE]

                # Current joint state (Unitree order)
                current_positions_unitree = np.array([ms.q for ms in self.latest_lowstate.motor_state[:12]], dtype=np.float32)
                current_velocities_unitree = np.array([ms.dq for ms in self.latest_lowstate.motor_state[:12]], dtype=np.float32)

                torques_unitree = self._compute_torques(target_positions_unitree,
                                                        current_positions_unitree,
                                                        current_velocities_unitree)

                cmd = self._build_command(torques_unitree)
                self.cmd_pub.Write(cmd)

                # Logging every 100 steps
                if step % 25 == 0:
                    sport_vel = (
                        np.array(self.latest_sportmode.velocity, dtype=np.float32)
                        if self.latest_sportmode
                        else np.zeros(3, dtype=np.float32)
                    )
                    print("---")
                    print(f"[{step:05d}] base linear velocity (body frame) [m/s]")
                    print("  source      vx        vy        vz")
                    print(
                        "  policy  "
                        f"{policy_vel[0]:+8.4f} {policy_vel[1]:+8.4f} {policy_vel[2]:+8.4f}"
                    )
                    print(
                        "  kalman  "
                        f"{kalman_vel[0]:+8.4f} {kalman_vel[1]:+8.4f} {kalman_vel[2]:+8.4f}"
                    )
                    print(
                        "  sport   "
                        f"{sport_vel[0]:+8.4f} {sport_vel[1]:+8.4f} {sport_vel[2]:+8.4f}"
                    )
                    print("---")
                    print(self.vel_estimator.status_string())

                self.obs_processor.update_last_actions(actions_raw)

                # Control rate enforcement
                elapsed = time.perf_counter() - loop_start
                sleep_time = self.config.CONTROL_DT - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                step += 1

        except KeyboardInterrupt:
            print("Interrupted, stopping...")

        finally:
            zero_cmd = self._zero_command()
            for _ in range(10):
                self.cmd_pub.Write(zero_cmd)
                time.sleep(0.01)
            print(f"Done. Total steps: {step}")


def parse_args():
    default_policy = Path(__file__).resolve().parent / "model_flat" / "exported" / "policy.onnx"

    parser = argparse.ArgumentParser(description="Run flat-ground policy with velocity estimator toggle.")
    parser.add_argument("--policy-path", type=str,
                        default=str(default_policy),
                        help="Path to ONNX policy file.")
    parser.add_argument("--estimate", choices=["sport", "kalman"], default="sport",
                        help="Velocity source for policy input.")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Duration to run in seconds.")
    parser.add_argument("--interface", type=str, default="lo",
                        help="Network interface for Unitree SDK (lo for simulator).")
    parser.add_argument("--disable-noise", action="store_true",
                        help="Disable observation noise.")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Go2 Flat Policy Runner (MuJoCo)")
    print("=" * 60)
    print(f"Velocity source (--estimate): {args.estimate}")
    print("Kalman estimator always runs for logging; switch source with --estimate.")
    print("Ensure Unitree simulation/bridge is running before starting.\n")

    runner = None
    try:
        runner = FlatPolicyRunner(
            policy_path=args.policy_path,
            estimate_source=args.estimate,
            interface=args.interface,
            disable_noise=args.disable_noise,
        )
        runner.run(duration=args.duration)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    finally:
        if runner:
            # Explicitly delete DDS objects to flush resources
            try:
                del runner.state_sub
                del runner.sport_sub
                del runner.cmd_pub
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())

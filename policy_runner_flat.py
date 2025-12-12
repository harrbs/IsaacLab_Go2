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

# -----------------------------------------------------------------------------
# Go2 leg geometry (from provided MuJoCo MJCF)
# -----------------------------------------------------------------------------
# Hip locations in body frame (m)
HIP_POS = np.array([
    [0.1934, -0.0465, 0.0],  # FR
    [0.1934,  0.0465, 0.0],  # FL
    [-0.1934, -0.0465, 0.0], # RR
    [-0.1934,  0.0465, 0.0], # RL
], dtype=np.float32)

# Abduction-to-thigh joint offset along +Y (left) / -Y (right) in hip frame (m)
HIP_LINK_OFFSET_Y = np.array([-0.0955, 0.0955, -0.0955, 0.0955], dtype=np.float32)

# Link lengths (m)
THIGH_LEN = 0.213
CALF_LEN = 0.213

# Stance detection thresholds
STANCE_FOOT_Z_MAX = -0.05  # body-frame foot height must be below this (negative, more lenient)
STANCE_VEL_MAX = 0.60      # foot speed magnitude must be below this (m/s, more lenient)


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
    """EKF with velocity and accelerometer bias state."""

    def __init__(self,
                 process_var: float,
                 meas_var: float,
                 bias_var: float,
                 init_vel_var: float,
                 init_bias_var: float):
        self.process_var = process_var
        self.meas_var = meas_var
        self.bias_var = bias_var
        self.init_vel_var = init_vel_var
        self.init_bias_var = init_bias_var
        self.reset()

    def reset(self):
        # State: [v_b (3), b_a (3)]
        self.x = np.zeros(6, dtype=np.float32)
        self.P = np.zeros((6, 6), dtype=np.float32)
        self.P[:3, :3] = np.eye(3, dtype=np.float32) * self.init_vel_var
        self.P[3:, 3:] = np.eye(3, dtype=np.float32) * self.init_bias_var
        self.R = np.eye(3, dtype=np.float32) * self.meas_var

    def predict(self, acc_body: np.ndarray, dt: float):
        v = self.x[:3]
        b = self.x[3:]
        v_pred = v + dt * (acc_body - b)
        self.x[:3] = v_pred
        # Bias follows random walk
        
        # 상태 자코비안
        F = np.eye(6, dtype=np.float32)
        F[:3, 3:] = -dt * np.eye(3, dtype=np.float32)
        
        # 입력 자코비안
        G = np.zeros((6, 6), dtype=np.float32)
        G[:3, :3] = dt * np.eye(3, dtype=np.float32)
        G[3:, 3:] = np.eye(3, dtype=np.float32)
        
        # 과정 잡음
        Qc = np.zeros((6, 6), dtype=np.float32)
        Qc[:3, :3] = np.eye(3, dtype=np.float32) * self.process_var
        Qc[3:, 3:] = np.eye(3, dtype=np.float32) * self.bias_var

        # 공분산 전파
        self.P = F @ self.P @ F.T + G @ Qc @ G.T

    def update(self, z: np.ndarray):
        H = np.zeros((3, 6), dtype=np.float32)
        H[:, :3] = np.eye(3, dtype=np.float32)
        
        # 업데이트는 동일
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ H) @ self.P


class Go2VelocityKalman:
    """Kalman filter for base linear velocity estimation on Go2."""

    def __init__(self,
                 process_var: float = 0.2,
                 meas_var: float = 0.05,
                 bias_var: float = 0.01, # 추가
                 init_var: float = 0.5,
                 init_bias_var: float = 0.1, # 추가
                 smoothing_window: int = 20):
        self._process_var = process_var
        self._meas_var = meas_var
        self._bias_var = bias_var
        self._init_var = init_var
        self._init_bias_var = init_bias_var
        self.filter = SimpleKalman3D(process_var, meas_var, bias_var, init_var, init_bias_var)

        self.smoother = MovingAverage(window_size=smoothing_window)
        self.last_timestamp = None

        self.prior_err_sum = np.zeros(3, dtype=np.float64)
        self.prior_err_sq_sum = np.zeros(3, dtype=np.float64)
        self.err_samples = 0

    def reset(self):
        self.filter = SimpleKalman3D(self._process_var, self._meas_var, self._bias_var,
                                     self._init_var, self._init_bias_var)
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

    @staticmethod
    def _rot_x(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]], dtype=np.float32)

    @staticmethod
    def _drot_x(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[0, 0, 0],
                         [0, -s, -c],
                         [0, c, -s]], dtype=np.float32)

    @staticmethod
    def _rot_y(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]], dtype=np.float32)

    @staticmethod
    def _drot_y(theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[-s, 0, c],
                         [0, 0, 0],
                         [-c, 0, -s]], dtype=np.float32)

    def _foot_kinematics(self, leg_idx: int, q_leg: np.ndarray, dq_leg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return foot position (body frame) and analytic Jacobian wrt q_leg (3x3)."""
        q0, q1, q2 = q_leg

        R0 = self._rot_x(q0)
        R1 = self._rot_y(q1)
        R12 = self._rot_y(q1 + q2)

        dR0 = self._drot_x(q0)
        dR1 = self._drot_y(q1)
        dR12 = self._drot_y(q1 + q2)

        hip_offset = HIP_POS[leg_idx]
        hip_link = np.array([0.0, HIP_LINK_OFFSET_Y[leg_idx], 0.0], dtype=np.float32)
        thigh_vec = np.array([0.0, 0.0, -THIGH_LEN], dtype=np.float32)
        calf_vec = np.array([0.0, 0.0, -CALF_LEN], dtype=np.float32)

        p = (hip_offset
             + R0 @ hip_link
             + (R0 @ R1) @ thigh_vec
             + (R0 @ R12) @ calf_vec)

        # Jacobian columns
        dp_dq0 = dR0 @ hip_link + dR0 @ R1 @ thigh_vec + dR0 @ R12 @ calf_vec
        dp_dq1 = R0 @ dR1 @ thigh_vec + R0 @ dR12 @ calf_vec
        dp_dq2 = R0 @ dR12 @ calf_vec
        J = np.stack([dp_dq0, dp_dq1, dp_dq2], axis=1)
        return p, J

    def estimate_from_feet(self,
                           lowstate: LowState_,
                           base_ang_vel_body: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """Compute base velocity measurement using stance feet only.

        Returns:
            measurement: None if no stance feet, else averaged base velocity (body frame)
            stance_flags: bool array of length 4 (FR, FL, RR, RL) indicating stance detection
            foot_z: z position of each foot in body frame
            foot_speed: speed magnitude of each foot in body frame
        """
        stance_flags = np.zeros(4, dtype=bool)
        foot_z = np.zeros(4, dtype=np.float32)
        foot_speed = np.zeros(4, dtype=np.float32)
        meas_list = []
        omega = base_ang_vel_body.astype(np.float32)
        q_all = np.array([ms.q for ms in lowstate.motor_state[:12]], dtype=np.float32)
        dq_all = np.array([ms.dq for ms in lowstate.motor_state[:12]], dtype=np.float32)

        for leg_idx, start in enumerate([0, 3, 6, 9]):  # FR, FL, RR, RL (Unitree order)
            q_leg = q_all[start:start + 3]
            dq_leg = dq_all[start:start + 3]
            p, J = self._foot_kinematics(leg_idx, q_leg, dq_leg)
            v_leg_rel = np.cross(omega, p)
            v_foot_body = v_leg_rel + J @ dq_leg
            foot_z[leg_idx] = p[2]
            foot_speed[leg_idx] = np.linalg.norm(v_foot_body)

            # Simple stance heuristic: foot is low and moving slowly
            if p[2] < STANCE_FOOT_Z_MAX and np.linalg.norm(v_foot_body) < STANCE_VEL_MAX:
                v_base_meas = -v_foot_body
                meas_list.append(v_base_meas)
                stance_flags[leg_idx] = True

        if not meas_list:
            return None, stance_flags, foot_z, foot_speed
        return np.mean(np.stack(meas_list, axis=0), axis=0), stance_flags, foot_z, foot_speed

    def step(self,
             imu_acc: np.ndarray,
             quat: np.ndarray,
             dt: float,
             measurement: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Run one predict/update step with accelerometer bias estimation."""
        if dt <= 0:
            return self.smoother.average, None

        R_w_b = self._quat_to_world_body(quat)
        gravity_world = np.array([0.0, 0.0, -9.81], dtype=np.float32)
        gravity_body = R_w_b @ gravity_world

        accel_body = imu_acc + gravity_body

        self.filter.predict(acc_body=accel_body, dt=dt)
        prior_vel = self.filter.x[:3].copy()

        if measurement is not None:
            self.filter.update(measurement)
            err = prior_vel - measurement
            self.prior_err_sum += err
            self.prior_err_sq_sum += err * err
            self.err_samples += 1

        smoothed = self.smoother.update(self.filter.x[:3].copy())
        return smoothed, prior_vel

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

        # Logging / drift stats
        self.err_sum = np.zeros(3, dtype=np.float64)
        self.err_sq_sum = np.zeros(3, dtype=np.float64)
        self.err_samples = 0
        self.vel_sum = np.zeros(3, dtype=np.float64)
        self.vel_sq_sum = np.zeros(3, dtype=np.float64)
        self.vel_samples = 0
        self.start_vel = None
        self.last_vel = None
        self.stance_counts = np.zeros(4, dtype=np.int64)  # FR, FL, RR, RL

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
        base_ang_vel = np.array(lowstate.imu_state.gyroscope, dtype=np.float32)

        # Measurement from foot zero-velocity constraint (body frame)
        meas_vel, stance_flags, foot_z, foot_speed = self.vel_estimator.estimate_from_feet(lowstate, base_ang_vel)

        # Fallback to SportMode if no stance feet found
        if meas_vel is None and self.latest_sportmode is not None:
            meas_vel = np.array([
                float(self.latest_sportmode.velocity[0]),
                float(self.latest_sportmode.velocity[1]),
                float(self.latest_sportmode.velocity[2])
            ], dtype=np.float32)

        kalman_vel, _ = self.vel_estimator.step(imu_acc, quat, dt, meas_vel)

        policy_vel = kalman_vel.copy()

        return policy_vel.astype(np.float32), kalman_vel.astype(np.float32), stance_flags, foot_z, foot_speed

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

                policy_vel, kalman_vel, stance_flags, foot_z, foot_speed = self._compute_policy_velocity(self.latest_lowstate, dt)
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

                # Stats accumulation
                self.vel_sum += kalman_vel
                self.vel_sq_sum += kalman_vel * kalman_vel
                self.vel_samples += 1
                self.last_vel = kalman_vel
                if self.start_vel is None:
                    self.start_vel = kalman_vel.copy()

                if self.latest_sportmode is not None:
                    sport_vel = np.array(self.latest_sportmode.velocity, dtype=np.float32)
                    err = kalman_vel - sport_vel
                    self.err_sum += err
                    self.err_sq_sum += err * err
                    self.err_samples += 1

                self.stance_counts += stance_flags.astype(np.int64)

                # Logging every 25 steps
                if step % 25 == 0:
                    sport_vel = np.array(self.latest_sportmode.velocity, dtype=np.float32) if self.latest_sportmode else np.zeros(3, dtype=np.float32)
                    stance_str = ["FR", "FL", "RR", "RL"]
                    stance_active = [name for name, flag in zip(stance_str, stance_flags) if flag]
                    drift_mean = self.vel_sum / max(1, self.vel_samples)
                    drift_std = np.sqrt(self.vel_sq_sum / max(1, self.vel_samples) - drift_mean * drift_mean)
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
                    print(f"  stance  active: {stance_active if stance_active else 'none'} | counts: {self.stance_counts.tolist()}")
                    print(f"  foot z   : [{foot_z[0]:+7.4f}, {foot_z[1]:+7.4f}, {foot_z[2]:+7.4f}, {foot_z[3]:+7.4f}]")
                    print(f"  foot speed: [{foot_speed[0]:+7.4f}, {foot_speed[1]:+7.4f}, {foot_speed[2]:+7.4f}, {foot_speed[3]:+7.4f}]")
                    print(f"  drift mean (so far): [{drift_mean[0]:+7.4f}, {drift_mean[1]:+7.4f}, {drift_mean[2]:+7.4f}]")
                    print(f"  drift std  (so far): [{drift_std[0]:+7.4f}, {drift_std[1]:+7.4f}, {drift_std[2]:+7.4f}]")
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

            # Summary stats for drift and reference RMSE (if SportMode available)
            if self.vel_samples > 0:
                mean_vel = self.vel_sum / self.vel_samples
                var_vel = self.vel_sq_sum / self.vel_samples - mean_vel * mean_vel
                std_vel = np.sqrt(np.maximum(var_vel, 0.0))
                print("\nVelocity estimator stats:")
                print(f"  samples        : {self.vel_samples}")
                print(f"  start vel [m/s]: [{self.start_vel[0]:+7.4f}, {self.start_vel[1]:+7.4f}, {self.start_vel[2]:+7.4f}]")
                print(f"  last  vel [m/s]: [{self.last_vel[0]:+7.4f}, {self.last_vel[1]:+7.4f}, {self.last_vel[2]:+7.4f}]")
                print(f"  mean  vel [m/s]: [{mean_vel[0]:+7.4f}, {mean_vel[1]:+7.4f}, {mean_vel[2]:+7.4f}]")
                print(f"  std   vel [m/s]: [{std_vel[0]:+7.4f}, {std_vel[1]:+7.4f}, {std_vel[2]:+7.4f}]")

            if self.err_samples > 0:
                mean_err = self.err_sum / self.err_samples
                rmse = np.sqrt(self.err_sq_sum / self.err_samples)
                print("\nReference vs SportMode (for evaluation only):")
                print(f"  samples: {self.err_samples}")
                print(f"  mean err [m/s]: [{mean_err[0]:+7.4f}, {mean_err[1]:+7.4f}, {mean_err[2]:+7.4f}]")
                print(f"  RMSE     [m/s]: [{rmse[0]:+7.4f}, {rmse[1]:+7.4f}, {rmse[2]:+7.4f}]")

            print(f"  stance counts (FR, FL, RR, RL): {self.stance_counts.tolist()}")
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

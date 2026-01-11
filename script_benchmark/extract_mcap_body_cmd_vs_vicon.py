#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute body-frame velocity from joint commands (Mecanum 4W) and compare against
body-frame velocity obtained from rigid-body vicon differentiation.

Additions:
- Detect movement start/end using |v_x| > 0.10 m/s (10 cm/s) with hysteresis (off at 0.08 m/s).
- Plot the difference (v_x_cmd - v_x_vicon) and draw vertical lines at start/end times.
- Compute mean absolute error over the motion interval and save to CSV.

Usage:
    python extract_mcap_body_cmd_vs_vicon.py --mcap path/to.bag.mcap --out ./out_dir
"""

import argparse
import os
import math
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from mcap_ros2.reader import read_ros2_messages
except Exception as e:
    raise ImportError(
        "Cannot import mcap_ros2.reader.read_ros2_messages. "
        "Install: pip install mcap mcap-ros2-support\n"
        f"Original error: {e}"
    )

# ---------- utils ----------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def ns_to_s(ns: int) -> float:
    return ns * 1e-9

def get_topic(msg) -> Optional[str]:
    t = getattr(msg, "topic", None)
    if t is not None:
        return t
    ch = getattr(msg, "channel", None)
    if ch is not None:
        return getattr(ch, "topic", None)
    return None

def get_log_time_ns(msg) -> Optional[int]:
    for attr in ("log_time", "log_time_ns", "publish_time"):
        v = getattr(msg, attr, None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    return None

def get_ros_msg(msg):
    m = getattr(msg, "ros_msg", None)
    if m is not None:
        return m
    return getattr(msg, "message", None)

def parse_field_path(root: Any, path: str) -> Any:
    cur = root
    for seg in [s for s in path.split('/') if s]:
        if '[' in seg and seg.endswith(']'):
            name, idx_part = seg.split('[', 1)
            if name:
                cur = getattr(cur, name)
            idx = int(idx_part[:-1])
            cur = cur[idx]
        else:
            cur = getattr(cur, seg)
    return cur

# quaternion helpers
def normalize_quaternion(qx: float, qy: float, qz: float, qw: float):
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n == 0:
        return (0.0,0.0,0.0,1.0)
    return (qx/n, qy/n, qz/n, qw/n)

def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    x, y, z, w = qx, qy, qz, qw
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)]
    ], dtype=float)
    return R

def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    # ZYX intrinsic yaw
    siny_cosp = 2.0 * (qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    return math.atan2(siny_cosp, cosy_cosp)

def interp_nan(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).interpolate(limit_direction="both").to_numpy()

# ---------- movement detection ----------
def detect_motion_interval(t: np.ndarray, vx_series: np.ndarray,
                           on_threshold: float = 0.10, off_threshold: float = 0.10) -> Tuple[Optional[float], Optional[float]]:
    """
    Detect start (first |vx| > on_threshold) and end (last transition to |vx| < off_threshold after start).
    Returns (t_start, t_end). If not found, returns (None, None) or (t_start, None).
    """
    if len(t) == 0:
        return (None, None)
    mask_on = np.abs(vx_series) > on_threshold
    if not mask_on.any():
        return (None, None)
    start_idx = int(np.argmax(mask_on))  # first True
    # find end: last index where it goes back below off_threshold after start
    mask_off = np.abs(vx_series) > off_threshold
    end_idx = None
    for i in range(len(t)-1, start_idx, -1):
        if mask_off[i]:
            end_idx = i
            break
    t_start = float(t[start_idx])
    t_end = float(t[end_idx]) if end_idx is not None else None

    return (t_start, t_end)

# ---------- extraction ----------
def extract(mcap_path: str, out_dir: str):
    ensure_dir(out_dir)

    # Geometry & indices (edit if needed)
    r = 0.05    # wheel radius [m]
    lx = 0.25   # half-length x from center to wheel [m]
    ly = 0.15   # half-length y from center to wheel [m]
    idx_lf = 9  # JointState.velocity indices for command
    idx_rf = 8
    idx_lb = 10
    idx_rb = 11

    cmd_buf: List[Tuple[float, float, float, float, float]] = []  # t, w_lf, w_rf, w_lb, w_rb
    vicon_buf: List[Tuple[float, float, float, float, float, float, float]] = []  # t, px,py,pz,qx,qy,qz,qw

    first_time_ns: Optional[int] = None

    with open(mcap_path, "rb") as f:
        for msg in read_ros2_messages(f):
            topic = get_topic(msg)
            t_ns = get_log_time_ns(msg)
            if t_ns is None:
                continue
            if first_time_ns is None:
                first_time_ns = t_ns
            t = ns_to_s(t_ns - first_time_ns)
            ros_msg = get_ros_msg(msg)
            if ros_msg is None:
                continue

            # Joint command velocities
            if topic == "/joint_controller/command":
                try:
                    cmd = getattr(ros_msg, "command", None) or ros_msg
                    vel = getattr(cmd, "velocity", None)
                    if vel is not None and len(vel) > max(idx_lf, idx_rf, idx_lb, idx_rb):
                        w_lf = float(vel[idx_lf])
                        w_rf = -float(vel[idx_rf])  # your sign convention
                        w_lb = float(vel[idx_lb])
                        w_rb = -float(vel[idx_rb])  # your sign convention
                        cmd_buf.append((t, w_lf, w_rf, w_lb, w_rb))
                except Exception:
                    pass

            # Rigid body pose/orientation
            if topic == "/rigid_bodies":
                try:
                    px = float(parse_field_path(ros_msg, "rigidbodies[3]/pose/position/x"))
                    py = float(parse_field_path(ros_msg, "rigidbodies[3]/pose/position/y"))
                    pz = float(parse_field_path(ros_msg, "rigidbodies[3]/pose/position/z"))
                    qx = float(parse_field_path(ros_msg, "rigidbodies[3]/pose/orientation/x"))
                    qy = float(parse_field_path(ros_msg, "rigidbodies[3]/pose/orientation/y"))
                    qz = float(parse_field_path(ros_msg, "rigidbodies[3]/pose/orientation/z"))
                    qw = float(parse_field_path(ros_msg, "rigidbodies[3]/pose/orientation/w"))
                    vicon_buf.append((t, px, py, pz, qx, qy, qz, qw))
                except Exception:
                    pass

    if len(cmd_buf) == 0:
        raise RuntimeError("No joint command velocities found. Check the indices and topic name.")
    if len(vicon_buf) == 0:
        raise RuntimeError("No rigid body vicon data found on /rigid_bodies.")

    df_cmd = pd.DataFrame(cmd_buf, columns=["t","w_lf","w_rf","w_lb","w_rb"]).sort_values("t").reset_index(drop=True) # data frame for the command velocities
    df_vicon = pd.DataFrame(vicon_buf, columns=["t","px","py","pz","qx","qy","qz","qw"]).sort_values("t").reset_index(drop=True) # data frame for the vicon rigid body data

    # Command (wheels) -> body twist (Mecanum 45°)
    L = lx + ly
    df_cmd["vx_b_cmd"] = -(r/4.0) * (-df_cmd["w_lf"] + df_cmd["w_rf"] + df_cmd["w_lb"] - df_cmd["w_rb"])
    df_cmd["vy_b_cmd"] = (r/4.0) * (df_cmd["w_lf"] + df_cmd["w_rf"] + df_cmd["w_lb"] + df_cmd["w_rb"])
    df_cmd["omega_b_cmd"] = (r/(4.0*L)) * (-df_cmd["w_lf"] + df_cmd["w_rf"] - df_cmd["w_lb"] + df_cmd["w_rb"])

    # Vicon -> body twist
    t = df_vicon["t"].to_numpy()
    px = interp_nan(df_vicon["px"].to_numpy())
    py = interp_nan(df_vicon["py"].to_numpy())
    pz = interp_nan(df_vicon["pz"].to_numpy())
    vx_w = np.gradient(px, t)  # obtain world-frame velocity by differentiation
    vy_w = np.gradient(py, t)
    vz_w = np.gradient(pz, t)

    qx = df_vicon["qx"].to_numpy(); qy = df_vicon["qy"].to_numpy()
    qz = df_vicon["qz"].to_numpy(); qw = df_vicon["qw"].to_numpy()
    vbx = np.empty_like(vx_w); vby = np.empty_like(vy_w); vbz = np.empty_like(vz_w)
    yaw = np.empty_like(vx_w)
    for i in range(len(t)):
        q = normalize_quaternion(qx[i], qy[i], qz[i], qw[i])
        R = quat_to_rotmat(*q); Rt = R.T
        vw = np.array([vx_w[i], vy_w[i], vz_w[i]])
        vb = Rt @ vw
        vbx[i], vby[i], vbz[i] = vb   # body-frame velocity (obtained from Vicon)
        yaw[i] = quat_to_yaw(*q)
    yaw_u = np.unwrap(yaw)
    omega_b_vicon = np.gradient(yaw_u, t)

    df_vicon_vel = pd.DataFrame({"t": df_vicon["t"], "vx_b_vicon": vbx, "vy_b_vicon": vby, "omega_b_vicon": omega_b_vicon})  # body velocities from vicon rigid body

    # Align timelines
    df = pd.merge_asof(
        df_cmd[["t","vx_b_cmd","vy_b_cmd","omega_b_cmd"]].sort_values("t"),
        df_vicon_vel.sort_values("t"),
        on="t", direction="nearest"
    )

    # --- Detect start/end using vx (prefer cmd; fallback to vicon) ---
    vx_ref = df["vx_b_cmd"].to_numpy()
    if np.isnan(vx_ref).all():
        vx_ref = df["vx_b_vicon"].to_numpy()
    t_arr = df["t"].to_numpy()
    t_start, t_end = detect_motion_interval(t_arr, vx_ref, on_threshold=0.10, off_threshold=0.08)

    # --- Error analysis on v_x ---
    df["err_vx"] = df["vx_b_cmd"] - df["vx_b_vicon"]
    
    # Save base CSV (with err_vx included)
    out_csv = os.path.join(out_dir, "body_velocity_cmd_vs_vicon.csv")
    df.to_csv(out_csv, index=False)

    if t_start is not None:
        if t_end is None:
            mask = (df["t"] >= t_start)
        else:
            mask = (df["t"] >= t_start) & (df["t"] <= t_end)
        mae_vx = float(np.nanmean(np.abs(df.loc[mask, "err_vx"])))
    else:
        mask = slice(None)
        mae_vx = float(np.nanmean(np.abs(df["err_vx"])))

    # Save error CSV
    err_csv = os.path.join(out_dir, "velocity_error_summary.csv")
    pd.DataFrame({
        "t_start": [t_start],
        "t_end": [t_end],
        "mae_vx": [mae_vx]
    }).to_csv(err_csv, index=False)

    # --- Plot difference with vertical lines ---
    plt.figure()
    plt.plot(df["t"], df["err_vx"], label="vx_b_cmd - vx_b_vicon")
    if t_start is not None:
        plt.axvline(t_start, linestyle="--", color="red", linewidth=2)
    if t_end is not None:
        plt.axvline(t_end, linestyle="--", color="red", linewidth=2)
    plt.xlabel("time [s]"); plt.ylabel("velocity error [m/s]"); plt.title("v_x error (cmd - vicon)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_vx.png"), dpi=150); plt.close()

    # Also keep original comparison plots, now with start/end lines.
    def plot_compare_with_lines(col_a: str, col_b: str, ylabel: str, title: str, out_png: str):
        plt.figure()
        plt.plot(df["t"], df[col_a], label=col_a)
        plt.plot(df["t"], df[col_b], label=col_b)
        if t_start is not None:
            plt.axvline(t_start, linestyle="--", color="red", linewidth=2)
        if t_end is not None:
            plt.axvline(t_end, linestyle="--", color="red", linewidth=2)
        plt.xlabel("time [s]"); plt.ylabel(ylabel); plt.title(title); plt.legend(); plt.tight_layout()
        plt.savefig(out_png, dpi=150); plt.close()

    plot_compare_with_lines("vx_b_cmd", "vx_b_vicon", "v_x (body) [m/s]",
                            "Body v_x: cmd vs vicon", os.path.join(out_dir, "compare_vx_body.png"))
    plot_compare_with_lines("vy_b_cmd", "vy_b_vicon", "v_y (body) [m/s]",
                            "Body v_y: cmd vs vicon", os.path.join(out_dir, "compare_vy_body.png"))
    plot_compare_with_lines("omega_b_cmd", "omega_b_vicon", "omega_z (body) [rad/s]",
                            "Body omega_z: cmd vs vicon", os.path.join(out_dir, "compare_omega_body.png"))

    # --- 3D trajectory with orientation frames ---
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        px_traj = df_vicon["px"].to_numpy()
        py_traj = df_vicon["py"].to_numpy()
        pz_traj = df_vicon["pz"].to_numpy()
        
        ax.plot(px_traj, py_traj, pz_traj, 'b-', linewidth=2, label='Trajectory')
        ax.scatter(px_traj[0], py_traj[0], pz_traj[0], c='green', s=100, marker='o', label='Start')
        ax.scatter(px_traj[-1], py_traj[-1], pz_traj[-1], c='red', s=100, marker='s', label='End')
        
        # Add orientation frames at regular intervals
        n_frames = min(10, len(df_vicon))  # Show up to 10 frames
        indices = np.linspace(0, len(df_vicon)-1, n_frames, dtype=int)
        frame_length = 0.015  # Length of axis arrows (reduced)
        
        for idx in indices:
            px, py, pz = px_traj[idx], py_traj[idx], pz_traj[idx]
            qx_i = df_vicon.iloc[idx]["qx"]
            qy_i = df_vicon.iloc[idx]["qy"]
            qz_i = df_vicon.iloc[idx]["qz"]
            qw_i = df_vicon.iloc[idx]["qw"]
            
            # Normalize quaternion
            q = normalize_quaternion(qx_i, qy_i, qz_i, qw_i)
            R = quat_to_rotmat(*q)
            
            # Get rotated axis directions (x=red, y=green, z=blue)
            x_axis = R[:, 0] * frame_length
            y_axis = R[:, 1] * frame_length
            z_axis = R[:, 2] * frame_length
            
            # Draw axes as arrows
            ax.quiver(px, py, pz, x_axis[0], x_axis[1], x_axis[2], color='r', arrow_length_ratio=0.3, linewidth=1.5)
            ax.quiver(px, py, pz, y_axis[0], y_axis[1], y_axis[2], color='g', arrow_length_ratio=0.3, linewidth=1.5)
            ax.quiver(px, py, pz, z_axis[0], z_axis[1], z_axis[2], color='b', arrow_length_ratio=0.3, linewidth=1.5)
        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title('3D Trajectory with Orientation Frames')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set equal aspect ratio
        max_range = np.array([px_traj.max()-px_traj.min(), py_traj.max()-py_traj.min(), pz_traj.max()-pz_traj.min()]).max() / 2.0
        mid_x = (px_traj.max()+px_traj.min()) * 0.5
        mid_y = (py_traj.max()+py_traj.min()) * 0.5
        mid_z = (pz_traj.max()+pz_traj.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        png_path = os.path.join(out_dir, "trajectory_3d.png")
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"Saved 3D trajectory: {png_path}")
    except Exception as e:
        print(f"Warning: Could not create 3D trajectory plot: {e}")

    print("Done.")
    print(f"Saved data: {out_csv}")
    print(f"Saved error summary: {err_csv}")
    if t_start is not None:
        print(f"Motion start at t = {t_start:.3f} s")
    if t_end is not None:
        print(f"Motion end   at t = {t_end:.3f} s")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcap", required=True, help="Path to the .mcap file")
    ap.add_argument("--out", required=True, help="Output directory")
    args = ap.parse_args()
    extract(args.mcap, args.out)

if __name__ == "__main__":
    main()

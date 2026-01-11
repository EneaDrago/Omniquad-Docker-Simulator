#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute body-frame velocity from wheel speeds (Mecanum 4W) and compare against
body-frame velocity obtained from rigid-body vicon differentiation.

Additions:
- Detect movement start/end using |v_x| > 0.10 m/s (10 cm/s) with hysteresis (off at 0.08 m/s).
- Plot the difference (v_x_wheels - v_x_vicon) and draw vertical lines at start/end times.
- Compute mean absolute error over the motion interval and save to CSV.

Usage:
    python extract_mcap_body_vs_wheels_plus.py --mcap path/to.bag.mcap --out ./out_dir
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

    print(f"mask_on: {mask_on}, \n\nmask_off: {mask_off}, \nstart_idx: {start_idx}, end_idx: {end_idx}")
    return (t_start, t_end)

# ---------- extraction ----------
def extract(mcap_path: str, out_dir: str):
    ensure_dir(out_dir)

    # Geometry & indices (edit if needed)
    r = 0.05    # wheel radius [m]
    lx = 0.25   # half-length x from center to wheel [m]
    ly = 0.15   # half-length y from center to wheel [m]
    idx_lf = 9  # JointState.velocity indices
    idx_rf = 8
    idx_lb = 10
    idx_rb = 11

    wheels_buf: List[Tuple[float, float, float, float, float]] = []  # t, w_lf, w_rf, w_lb, w_rb
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

            # Wheel speeds from JointState
            if topic in ("/state_broadcaster/joints_state", "/joint_states"):
                try:
                    vel = getattr(ros_msg, "velocity", None)
                    if vel is not None and len(vel) > max(idx_lf, idx_rf, idx_lb, idx_rb):
                        w_lf = float(vel[idx_lf])
                        w_rf = -float(vel[idx_rf])  # your sign convention
                        w_lb = float(vel[idx_lb])
                        w_rb = -float(vel[idx_rb])  # your sign convention
                        wheels_buf.append((t, w_lf, w_rf, w_lb, w_rb))
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

    if len(wheels_buf) == 0:
        raise RuntimeError("No wheel velocities found. Check the indices and topic name.")
    if len(vicon_buf) == 0:
        raise RuntimeError("No rigid body vicon data found on /rigid_bodies.")

    df_state = pd.DataFrame(wheels_buf, columns=["t","w_lf","w_rf","w_lb","w_rb"]).sort_values("t").reset_index(drop=True) # data frame for the state joint velocities
    df_vicon = pd.DataFrame(vicon_buf, columns=["t","px","py","pz","qx","qy","qz","qw"]).sort_values("t").reset_index(drop=True) # data frame for the vicon rigid body data

    # State (wheels) -> body twist (Mecanum 45°)
    L = lx + ly
    df_state["vx_b_state"] = (r/4.0) * (df_state["w_lf"] + df_state["w_rf"] + df_state["w_lb"] + df_state["w_rb"])
    df_state["vy_b_state"] = (r/4.0) * (-df_state["w_lf"] + df_state["w_rf"] + df_state["w_lb"] - df_state["w_rb"])
    df_state["omega_b_state"] = (r/(4.0*L)) * (-df_state["w_lf"] + df_state["w_rf"] - df_state["w_lb"] + df_state["w_rb"])

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
        df_state[["t","vx_b_state","vy_b_state","omega_b_state"]].sort_values("t"),
        df_vicon_vel.sort_values("t"),
        on="t", direction="nearest"
    )

    # --- Detect start/end using vx (prefer state; fallback to vicon) ---
    vx_ref = df["vx_b_state"].to_numpy()
    if np.isnan(vx_ref).all():
        vx_ref = df["vx_b_vicon"].to_numpy()
    t_arr = df["t"].to_numpy()
    t_start, t_end = detect_motion_interval(t_arr, vx_ref, on_threshold=0.10, off_threshold=0.08)

    # --- Error analysis on v_x ---
    df["err_vx"] = df["vx_b_state"] - df["vx_b_vicon"]
    
    # Save base CSV (with err_vx included)
    out_csv = os.path.join(out_dir, "body_velocity_state_vs_vicon.csv")
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
    plt.plot(df["t"], df["err_vx"], label="vx_b_state - vx_b_vicon")
    if t_start is not None:
        plt.axvline(t_start, linestyle="--", color="red", linewidth=2)
    if t_end is not None:
        plt.axvline(t_end, linestyle="--", color="red", linewidth=2)
    plt.xlabel("time [s]"); plt.ylabel("velocity error [m/s]"); plt.title("v_x error (state - vicon)")
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

    plot_compare_with_lines("vx_b_state", "vx_b_vicon", "v_x (body) [m/s]",
                            "Body v_x: state vs vicon", os.path.join(out_dir, "compare_vx_body.png"))
    plot_compare_with_lines("vy_b_state", "vy_b_vicon", "v_y (body) [m/s]",
                            "Body v_y: state vs vicon", os.path.join(out_dir, "compare_vy_body.png"))
    plot_compare_with_lines("omega_b_state", "omega_b_vicon", "omega_z (body) [rad/s]",
                            "Body omega_z: state vs vicon", os.path.join(out_dir, "compare_omega_body.png"))

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

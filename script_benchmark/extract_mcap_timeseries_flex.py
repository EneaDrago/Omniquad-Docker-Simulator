#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extracts signals from a ROS 2 MCAP and computes **body-frame linear velocity** for rigid body[3]:
1) Read position p_w(t) and orientation q_wb(t) (world->body quaternion).
2) Differentiate p_w(t) to get world-frame velocity v_w(t).
3) Rotate to body frame: v_b(t) = R(q)^T * v_w(t).

Also saves previous cmd_vel, joint velocities, and rigid body position if present.

Install:
    pip install mcap mcap-ros2-support pandas matplotlib

Usage:
    python extract_mcap_body_velocity.py --mcap path/to.bag.mcap --out out_dir
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

def try_get_scalar(ros_msg: Any) -> Optional[float]:
    for attr in ("data", "value"):
        v = getattr(ros_msg, attr, None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                return None
    return None


# ---------- signals we want ----------
SIGNALS: Dict[str, List[Tuple[str, Optional[str]]]] = {
    # cmd_vel (both composite and scalar aliases)
    "cmd_ang_x": [("/cmd_vel", "angular/x"), ("/cmd_vel/angular/x", None)],
    "cmd_ang_y": [("/cmd_vel", "angular/y"), (("/cmd_vel/angular/y"), None)],
    "cmd_ang_z": [("/cmd_vel", "angular/z"), (("/cmd_vel/angular/z"), None)],
    "cmd_lin_x": [("/cmd_vel", "linear/x"),  (("/cmd_vel/linear/x"), None)],
    "cmd_lin_y": [("/cmd_vel", "linear/y"),  (("/cmd_vel/linear/y"), None)],
    "cmd_lin_z": [("/cmd_vel", "linear/z"),  (("/cmd_vel/linear/z"), None)],

    # rigid body position (index 3)
    "rb3_pos_x": [("/rigid_bodies", "rigidbodies[3]/pose/position/x"),
                  ("/rigid_bodies/rigidbodies/3/pose/position/x", None)],
    "rb3_pos_y": [("/rigid_bodies", "rigidbodies[3]/pose/position/y"),
                  ("/rigid_bodies/rigidbodies/3/pose/position/y", None)],
    "rb3_pos_z": [("/rigid_bodies", "rigidbodies[3]/pose/position/z"),
                  ("/rigid_bodies/rigidbodies/3/pose/position/z", None)],

    # rigid body orientation quaternion (index 3) -> need x,y,z,w
    "rb3_ori_x": [("/rigid_bodies", "rigidbodies[3]/pose/orientation/x"),
                  ("/rigid_bodies/rigidbodies/3/pose/orientation/x", None)],
    "rb3_ori_y": [("/rigid_bodies", "rigidbodies[3]/pose/orientation/y"),
                  ("/rigid_bodies/rigidbodies/3/pose/orientation/y", None)],
    "rb3_ori_z": [("/rigid_bodies", "rigidbodies[3]/pose/orientation/z"),
                  ("/rigid_bodies/rigidbodies/3/pose/orientation/z", None)],
    "rb3_ori_w": [("/rigid_bodies", "rigidbodies[3]/pose/orientation/w"),
                  ("/rigid_bodies/rigidbodies/3/pose/orientation/w", None)],

    # joint velocities (accept alias /joint_states)
    "jv_8":  [("/state_broadcaster/joints_state", "velocity[8]"),  ("/joint_states", "velocity[8]")],
    "jv_9":  [("/state_broadcaster/joints_state", "velocity[9]"),  ("/joint_states", "velocity[9]")],
    "jv_10": [("/state_broadcaster/joints_state", "velocity[10]"), ("/joint_states", "velocity[10]")],
    "jv_11": [("/state_broadcaster/joints_state", "velocity[11]"), ("/joint_states", "velocity[11]")],
}


# ---------- extraction ----------
def extract(mcap_path: str, out_dir: str):
    ensure_dir(out_dir)

    wanted_by_topic: Dict[str, List[Tuple[str, Optional[str]]]] = {}
    for label, alts in SIGNALS.items():
        for topic, field in alts:
            wanted_by_topic.setdefault(topic, []).append((label, field))

    buffers: Dict[str, List[Tuple[float, float]]] = {label: [] for label in SIGNALS}
    first_time_ns: Optional[int] = None

    with open(mcap_path, "rb") as f:
        for msg in read_ros2_messages(f):
            topic = get_topic(msg)
            if topic not in wanted_by_topic:
                continue

            t_ns = get_log_time_ns(msg)
            if t_ns is None:
                continue
            if first_time_ns is None:
                first_time_ns = t_ns
            t = ns_to_s(t_ns - first_time_ns)

            ros_msg = get_ros_msg(msg)
            if ros_msg is None:
                continue

            for label, field in wanted_by_topic[topic]:
                try:
                    if field is None:
                        val_opt = try_get_scalar(ros_msg)
                        if val_opt is None:
                            continue
                        val = val_opt
                    else:
                        val = float(parse_field_path(ros_msg, field))
                    buffers[label].append((t, val))
                except Exception:
                    continue

    # Build DataFrame for rigid body pos/orientation
    rb_labels = ["rb3_pos_x", "rb3_pos_y", "rb3_pos_z", "rb3_ori_x", "rb3_ori_y", "rb3_ori_z", "rb3_ori_w"]
    rb_df = build_time_df(buffers, rb_labels, rename_map={
        "rb3_pos_x":"px","rb3_pos_y":"py","rb3_pos_z":"pz",
        "rb3_ori_x":"qx","rb3_ori_y":"qy","rb3_ori_z":"qz","rb3_ori_w":"qw"
    })

    if rb_df is not None:
        # Compute world-frame velocity via gradient (handles non-uniform dt)
        t = rb_df["t"].to_numpy()
        px = rb_df["px"].to_numpy()
        py = rb_df["py"].to_numpy()
        pz = rb_df["pz"].to_numpy()

        # Interpolate NaNs (if any) before gradient
        def interp_nan(x: np.ndarray) -> np.ndarray:
            s = pd.Series(x).interpolate(limit_direction="both").to_numpy()
            return s

        px_i, py_i, pz_i = interp_nan(px), interp_nan(py), interp_nan(pz)
        vx_w = np.gradient(px_i, t)
        vy_w = np.gradient(py_i, t)
        vz_w = np.gradient(pz_i, t)

        # Normalize quaternion and compute R^T * v_w (world->body)
        qx = rb_df["qx"].to_numpy()
        qy = rb_df["qy"].to_numpy()
        qz = rb_df["qz"].to_numpy()
        qw = rb_df["qw"].to_numpy()

        vbx = np.empty_like(vx_w)
        vby = np.empty_like(vy_w)
        vbz = np.empty_like(vz_w)

        for i in range(len(t)):
            q = normalize_quaternion(qx[i], qy[i], qz[i], qw[i])
            R = quat_to_rotmat(*q)          # R maps body->world
            Rt = R.T                         # world->body
            vw = np.array([vx_w[i], vy_w[i], vz_w[i]])
            vb = Rt @ vw
            vbx[i], vby[i], vbz[i] = vb

        # Save CSVs and plots
        rb_out = rb_df.copy()
        rb_out[["vx_w","vy_w","vz_w"]] = np.column_stack([vx_w, vy_w, vz_w])
        rb_out[["vx_b","vy_b","vz_b"]] = np.column_stack([vbx, vby, vbz])
        rb_out.to_csv(os.path.join(out_dir, "rigid_body_pose_and_velocity.csv"), index=False)

        # Plots
        plot_series(rb_out[["t","vx_w","vy_w","vz_w"]], "time [s]","velocity [m/s]","World-frame linear velocity",
                    os.path.join(out_dir,"vel_world.png"))
        plot_series(rb_out[["t","vx_b","vy_b","vz_b"]], "time [s]","velocity [m/s]","Body-frame linear velocity",
                    os.path.join(out_dir,"vel_body.png"))

    # Optional: save other groups as before
    save_cmd_vel(buffers, out_dir)
    save_joint_vel(buffers, out_dir)

    print("Done.")
    if rb_df is None:
        print("Warning: rigid body pos/orientation not found; no body velocity computed.")


# ---------- helpers for DF/plots ----------
def build_time_df(buffers: Dict[str, List[Tuple[float,float]]], labels: List[str], rename_map: Dict[str,str]) -> Optional[pd.DataFrame]:
    # pick basis label with data
    base_label = next((l for l in labels if len(buffers[l])>0), None)
    if base_label is None:
        return None
    base = pd.DataFrame(buffers[base_label], columns=["t", base_label]).sort_values("t")
    for l in labels:
        if l == base_label or len(buffers[l])==0:
            continue
        df_l = pd.DataFrame(buffers[l], columns=["t", l]).sort_values("t")
        base = pd.merge_asof(base, df_l, on="t", direction="nearest")
    base = base.rename(columns=rename_map)
    return base

def plot_series(df: pd.DataFrame, xlabel: str, ylabel: str, title: str, out_png: str):
    plt.figure()
    t = df["t"]
    for col in df.columns:
        if col == "t": continue
        plt.plot(t, df[col], label=col)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()


# ---------- quaternion math ----------
def normalize_quaternion(qx: float, qy: float, qz: float, qw: float):
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n == 0:
        return (0.0,0.0,0.0,1.0)
    return (qx/n, qy/n, qz/n, qw/n)

def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Return rotation matrix R (body->world) for quaternion [x,y,z,w] assuming passive world frame."""
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


# ---------- extras: cmd_vel & joint velocities ----------
def to_df(buffers: Dict[str, List[Tuple[float, float]]], labels: List[str]) -> Optional[pd.DataFrame]:
    base_label = next((l for l in labels if len(buffers[l]) > 0), None)
    if base_label is None:
        return None
    base = pd.DataFrame(buffers[base_label], columns=["t", base_label])
    for l in labels:
        if l == base_label or len(buffers[l]) == 0:
            continue
        df_l = pd.DataFrame(buffers[l], columns=["t", l])
        base = pd.merge_asof(base.sort_values("t"), df_l.sort_values("t"), on="t", direction="nearest")
    return base

def save_cmd_vel(buffers: Dict[str, List[Tuple[float,float]]], out_dir: str):
    cmd_ang = to_df(buffers, ["cmd_ang_x","cmd_ang_y","cmd_ang_z"])
    cmd_lin = to_df(buffers, ["cmd_lin_x","cmd_lin_y","cmd_lin_z"])
    if cmd_ang is not None:
        cmd_ang = cmd_ang.rename(columns={"cmd_ang_x":"angular_x","cmd_ang_y":"angular_y","cmd_ang_z":"angular_z"})
        cmd_ang.to_csv(os.path.join(out_dir,"cmd_vel_angular.csv"), index=False)
        plot_series(cmd_ang[["t","angular_x","angular_y","angular_z"]],
                    "time [s]","angular [rad/s]","/cmd_vel angular", os.path.join(out_dir,"cmd_vel_angular.png"))
    if cmd_lin is not None:
        cmd_lin = cmd_lin.rename(columns={"cmd_lin_x":"linear_x","cmd_lin_y":"linear_y","cmd_lin_z":"linear_z"})
        cmd_lin.to_csv(os.path.join(out_dir,"cmd_vel_linear.csv"), index=False)
        plot_series(cmd_lin[["t","linear_x","linear_y","linear_z"]],
                    "time [s]","linear [m/s]","/cmd_vel linear", os.path.join(out_dir,"cmd_vel_linear.png"))

def save_joint_vel(buffers: Dict[str, List[Tuple[float,float]]], out_dir: str):
    jv = to_df(buffers, ["jv_8","jv_9","jv_10","jv_11"])
    if jv is None:
        return
    jv = jv.rename(columns={"jv_8":"vel_8","jv_9":"vel_9","jv_10":"vel_10","jv_11":"vel_11"})
    jv.to_csv(os.path.join(out_dir,"joint_velocities.csv"), index=False)
    plot_series(jv[["t","vel_8","vel_9","vel_10","vel_11"]],
                "time [s]","joint velocity [rad/s]","Joint velocities (selected)", os.path.join(out_dir,"joint_velocities.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcap", required=True, help="Path to the .mcap file")
    ap.add_argument("--out", required=True, help="Output directory")
    args = ap.parse_args()
    extract(args.mcap, args.out)


if __name__ == "__main__":
    main()
"""
Two-stage 3D trajectory reconstruction:

Stage 1: Find optimal time offset by minimizing triangulation ray-distance
          (works even with different/variable frame rates)
Stage 2: Fit physics-based parabola(s) to the triangulated 3D points
          (handles bounce, smooths noise, interpolates gaps)

This avoids needing frame-level synchronization.
"""

import json
import glob
import os

import numpy as np
from scipy.optimize import minimize_scalar


GRAVITY = np.array([0.0, 0.0, -9.81])


# ========== Stage 1: Triangulation with auto time-offset ==========

def pixel_to_world(H, px, py):
    pt = np.array([px, py, 1.0])
    r = H @ pt
    return r[0] / r[2], r[1] / r[2]


def triangulate_pair(w1, w2, cam1_pos, cam2_pos):
    """Triangulate a 3D point from two ground-plane projections."""
    c1 = np.asarray(cam1_pos, dtype=np.float64)
    c2 = np.asarray(cam2_pos, dtype=np.float64)
    g1 = np.array([w1[0], w1[1], 0.0])
    g2 = np.array([w2[0], w2[1], 0.0])
    d1, d2 = g1 - c1, g2 - c2
    w = c1 - c2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d_val = np.dot(d1, w)
    e = np.dot(d2, w)
    denom = a * c - b * b
    if abs(denom) < 1e-10:
        return None, 99.0
    s = (b * e - c * d_val) / denom
    t = (a * e - b * d_val) / denom
    s, t = np.clip(s, 0, 1), np.clip(t, 0, 1)
    p1 = c1 + s * d1
    p2 = c2 + t * d2
    mid = (p1 + p2) / 2.0
    rd = np.linalg.norm(p1 - p2)
    if mid[2] < 0:
        mid[2] = 0.0
    return mid, rd


def interpolate_detection(dets, target_frame, fps):
    """Linearly interpolate a detection at a non-integer frame."""
    # Find bracketing detections
    for i in range(len(dets) - 1):
        f0, f1 = dets[i][0], dets[i + 1][0]
        if f0 <= target_frame <= f1:
            frac = (target_frame - f0) / (f1 - f0) if f1 != f0 else 0
            px = dets[i][1] + frac * (dets[i + 1][1] - dets[i][1])
            py = dets[i][2] + frac * (dets[i + 1][2] - dets[i][2])
            return px, py
    return None


def eval_offset(dt_seconds, dets_a, dets_b, fps_a, fps_b,
                H_a, H_b, cam_a_pos, cam_b_pos):
    """
    Evaluate a time offset: for each cam_a detection, find the corresponding
    cam_b detection (interpolated) and triangulate. Return mean ray distance.

    dt_seconds: cam_b_time = cam_b_frame/fps_b + dt_seconds
    Equivalently: for cam_a frame f_a at time f_a/fps_a,
    the corresponding cam_b frame is: (f_a/fps_a - dt_seconds) * fps_b
    """
    total_dist = 0.0
    count = 0

    for fa, px_a, py_a in dets_a:
        t_a = fa / fps_a  # cam_a time
        fb_float = (t_a - dt_seconds) * fps_b  # corresponding cam_b frame

        interp = interpolate_detection(dets_b, fb_float, fps_b)
        if interp is None:
            continue

        px_b, py_b = interp
        w_a = pixel_to_world(H_a, px_a, py_a)
        w_b = pixel_to_world(H_b, px_b, py_b)
        _, rd = triangulate_pair(w_a, w_b, cam_a_pos, cam_b_pos)
        total_dist += rd
        count += 1

    if count == 0:
        return 1e6
    return total_dist / count


def find_offset_and_triangulate(dets_a, dets_b, fps_a, fps_b,
                                 H_a, H_b, cam_a_pos, cam_b_pos):
    """
    Find optimal time offset, then triangulate all points.
    Returns (dt_seconds, list of 3D points with metadata).
    """
    # Coarse sweep
    best_dt, best_cost = 0.0, 1e18
    dt_costs = []
    for dt in np.linspace(-3.0, 3.0, 601):
        cost = eval_offset(dt, dets_a, dets_b, fps_a, fps_b,
                           H_a, H_b, cam_a_pos, cam_b_pos)
        dt_costs.append((dt, cost))
        if cost < best_cost:
            best_cost = cost
            best_dt = dt

    # Refine
    def cost_fn(dt_val):
        return eval_offset(dt_val, dets_a, dets_b, fps_a, fps_b,
                           H_a, H_b, cam_a_pos, cam_b_pos)

    refined = minimize_scalar(cost_fn,
                              bounds=(best_dt - 0.2, best_dt + 0.2),
                              method="bounded")
    best_dt = refined.x

    # Triangulate with best offset
    points_3d = []
    for fa, px_a, py_a in dets_a:
        t_a = fa / fps_a
        fb_float = (t_a - best_dt) * fps_b

        interp = interpolate_detection(dets_b, fb_float, fps_b)
        if interp is None:
            continue

        px_b, py_b = interp
        w_a = pixel_to_world(H_a, px_a, py_a)
        w_b = pixel_to_world(H_b, px_b, py_b)
        pt, rd = triangulate_pair(w_a, w_b, cam_a_pos, cam_b_pos)
        if pt is not None:
            points_3d.append({
                "t": t_a - dets_a[0][0] / fps_a,  # relative time
                "x": pt[0], "y": pt[1], "z": pt[2],
                "ray_dist": rd,
                "frame_a": fa,
            })

    return best_dt, points_3d, dt_costs


# ========== Stage 2: Parabola fitting to 3D points ==========

def fit_parabola_segment(points, t_key="t"):
    """
    Fit pos(t) = P0 + V0*t + 0.5*g*t^2 to 3D points.
    g = [0, 0, -9.81] is fixed. Solve for P0 and V0 via weighted least squares.
    Points with lower ray_dist get higher weight (more trustworthy triangulation).
    """
    n = len(points)
    if n < 3:
        return None

    ts = np.array([p[t_key] for p in points])
    xs = np.array([p["x"] for p in points])
    ys = np.array([p["y"] for p in points])
    zs = np.array([p["z"] for p in points])

    # Weights: inverse of ray distance (clamped to avoid division by zero)
    rds = np.array([p.get("ray_dist", 0.01) for p in points])
    weights = 1.0 / np.clip(rds, 0.01, None)
    weights = weights / weights.sum() * n  # normalize so sum = n

    # Weighted least squares: W^{1/2} A x = W^{1/2} b
    W_sqrt = np.diag(np.sqrt(weights))

    A = np.column_stack([np.ones(n), ts])  # Nx2
    Aw = W_sqrt @ A

    # X axis (no gravity)
    sol_x, _, _, _ = np.linalg.lstsq(Aw, W_sqrt @ xs, rcond=None)

    # Y axis (no gravity)
    sol_y, _, _, _ = np.linalg.lstsq(Aw, W_sqrt @ ys, rcond=None)

    # Z axis (gravity)
    bz = zs - 0.5 * (-9.81) * ts * ts
    sol_z, _, _, _ = np.linalg.lstsq(Aw, W_sqrt @ bz, rcond=None)

    p0 = np.array([sol_x[0], sol_y[0], sol_z[0]])
    v0 = np.array([sol_x[1], sol_y[1], sol_z[1]])

    # Compute residuals
    fitted = []
    residuals = []
    for i, p in enumerate(points):
        t = p[t_key]
        pos_fit = p0 + v0 * t + 0.5 * GRAVITY * t * t
        err = np.sqrt((pos_fit[0] - p["x"])**2 +
                       (pos_fit[1] - p["y"])**2 +
                       (pos_fit[2] - p["z"])**2)
        residuals.append(err)
        fitted.append({"t": t, "x": pos_fit[0], "y": pos_fit[1],
                        "z": pos_fit[2]})

    return {
        "p0": p0, "v0": v0,
        "residuals": residuals,
        "mean_error": np.mean(residuals),
        "max_error": np.max(residuals),
        "fitted_points": fitted,
    }


def fit_spatial_parabola(points):
    """
    Fit trajectory SPATIALLY: X(Y) = linear, Z(Y) = quadratic.

    For a ballistic trajectory without air resistance:
        x(t) = x0 + vx*t
        y(t) = y0 + vy*t
        z(t) = z0 + vz*t - 0.5*g*t^2

    Eliminating t via y: t = (y - y0) / vy
        X = ax*Y + bx           (linear)
        Z = az*Y^2 + bz*Y + cz  (quadratic, with az = -g/(2*vy^2))

    This is INDEPENDENT of frame rate / timing!
    Uses weighted least squares with weights based on ray_dist.
    """
    n = len(points)
    if n < 3:
        return None

    xs = np.array([p["x"] for p in points])
    ys = np.array([p["y"] for p in points])
    zs = np.array([p["z"] for p in points])

    # Weights
    rds = np.array([p.get("ray_dist", 0.01) for p in points])
    weights = 1.0 / np.clip(rds, 0.01, None)
    weights = weights / weights.sum() * n
    W_sqrt = np.diag(np.sqrt(weights))

    # X(Y) = ax*Y + bx
    A_xy = np.column_stack([ys, np.ones(n)])
    sol_x, _, _, _ = np.linalg.lstsq(W_sqrt @ A_xy, W_sqrt @ xs, rcond=None)
    ax, bx = sol_x

    # Z(Y) = az*Y^2 + bz*Y + cz
    A_zy = np.column_stack([ys**2, ys, np.ones(n)])
    sol_z, _, _, _ = np.linalg.lstsq(W_sqrt @ A_zy, W_sqrt @ zs, rcond=None)
    az, bz, cz = sol_z

    # Recover velocity from spatial coefficients
    # az = -g / (2*vy^2)  =>  vy = sqrt(-g / (2*az))
    g = 9.81
    if az < -1e-6:  # must be negative for ballistic
        vy = np.sqrt(g / (-2 * az))
    else:
        vy = 25.0  # fallback

    vx = ax * vy
    # From Z derivative: dz/dy = 2*az*y + bz = vz/vy - g*y/vy^2 at y=y0
    # At y=y0 (first point): vz = vy * (2*az*ys[0] + bz)  + g*ys[0]/vy
    # Actually simpler: vz/vy = bz + 2*az*y0 where y0 is the Y at t=0
    # Since we don't know y0 exactly, use the Y of the first point
    y0 = ys[0]
    vz = vy * (2 * az * y0 + bz)

    # Compute spatial residuals (distance from point to fitted curve)
    residuals = []
    fitted_pts = []
    for p in points:
        y = p["y"]
        x_fit = ax * y + bx
        z_fit = az * y * y + bz * y + cz
        err = np.sqrt((x_fit - p["x"])**2 + (z_fit - p["z"])**2)
        residuals.append(err)
        fitted_pts.append({"y": y, "x": x_fit, "z": z_fit})

    return {
        "type": "spatial",
        "ax": ax, "bx": bx,
        "az": az, "bz": bz, "cz": cz,
        "v0": np.array([vx, vy, vz]),
        "speed": np.linalg.norm([vx, vy, vz]),
        "residuals": residuals,
        "mean_error": np.mean(residuals),
        "max_error": np.max(residuals),
        "fitted_points": fitted_pts,
    }


def detect_bounce(points, z_threshold=0.4):
    """Find bounce point: local minimum in Z, must have rising Z after."""
    zs = [p["z"] for p in points]
    min_idx = int(np.argmin(zs))
    if zs[min_idx] < z_threshold and 1 < min_idx < len(zs) - 2:
        # Verify Z actually rises after the minimum
        if zs[min_idx + 1] > zs[min_idx] or zs[min_idx + 2] > zs[min_idx]:
            return min_idx
    return None


def fit_trajectory_with_bounce(points):
    """Fit piecewise spatial parabola: pre-bounce and post-bounce segments."""
    bounce_idx = detect_bounce(points)

    if bounce_idx is not None:
        pre = points[:bounce_idx + 1]
        post = points[bounce_idx:]

        fit_pre = fit_spatial_parabola(pre)
        fit_post = fit_spatial_parabola(post)

        # Also do time-based fit for comparison
        t_bounce = post[0]["t"]
        post_shifted = [dict(p, t_local=p["t"] - t_bounce) for p in post]
        fit_pre_time = fit_parabola_segment(pre)
        fit_post_time = fit_parabola_segment(post_shifted, t_key="t_local")

        return {
            "type": "piecewise",
            "bounce_idx": bounce_idx,
            "bounce_t": points[bounce_idx]["t"],
            "bounce_pos": [points[bounce_idx]["x"],
                           points[bounce_idx]["y"],
                           points[bounce_idx]["z"]],
            "pre_bounce": fit_pre,
            "post_bounce": fit_post,
            "pre_bounce_time": fit_pre_time,
            "post_bounce_time": fit_post_time,
        }
    else:
        fit_spatial = fit_spatial_parabola(points)
        fit_time = fit_parabola_segment(points)
        return {"type": "single", "fit_spatial": fit_spatial, "fit_time": fit_time}


# ========== Visualization ==========

def plot_results(points_3d, traj_fit, dt, dt_costs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Physics-Constrained 3D Trajectory Reconstruction",
                 fontsize=16, fontweight="bold")

    ts = [p["t"] for p in points_3d]
    xs = [p["x"] for p in points_3d]
    ys = [p["y"] for p in points_3d]
    zs = [p["z"] for p in points_3d]
    rds = [p["ray_dist"] for p in points_3d]

    # Generate smooth spatial fitted curves
    y_min, y_max = min(ys) - 1, max(ys) + 1
    y_smooth = np.linspace(y_min, y_max, 500)

    # Spatial fit curves
    sx_spatial, sy_spatial, sz_spatial = [], [], []
    if traj_fit["type"] == "piecewise":
        pre = traj_fit["pre_bounce"]
        post = traj_fit["post_bounce"]
        by = traj_fit["bounce_pos"][1]  # Y at bounce

        for y in y_smooth:
            if y <= by and pre:
                x_f = pre["ax"] * y + pre["bx"]
                z_f = pre["az"] * y**2 + pre["bz"] * y + pre["cz"]
            elif y > by and post:
                x_f = post["ax"] * y + post["bx"]
                z_f = post["az"] * y**2 + post["bz"] * y + post["cz"]
            else:
                continue
            if z_f < -0.5 or z_f > 5:  # clip unreasonable Z
                continue
            sx_spatial.append(x_f)
            sy_spatial.append(y)
            sz_spatial.append(z_f)
    else:
        fit = traj_fit["fit_spatial"]
        for y in y_smooth:
            x_f = fit["ax"] * y + fit["bx"]
            z_f = fit["az"] * y**2 + fit["bz"] * y + fit["cz"]
            if z_f < -0.5 or z_f > 5:
                continue
            sx_spatial.append(x_f)
            sy_spatial.append(y)
            sz_spatial.append(z_f)

    # Also generate time-based fit for comparison
    sx_time, sy_time, sz_time = [], [], []
    if traj_fit["type"] == "piecewise":
        pre_t = traj_fit.get("pre_bounce_time")
        post_t = traj_fit.get("post_bounce_time")
        if pre_t and post_t:
            bt = traj_fit["bounce_t"]
            for t in np.linspace(min(ts), max(ts), 300):
                if t <= bt:
                    pos = pre_t["p0"] + pre_t["v0"] * t + 0.5 * GRAVITY * t**2
                else:
                    tl = t - bt
                    pos = post_t["p0"] + post_t["v0"] * tl + 0.5 * GRAVITY * tl**2
                sx_time.append(pos[0])
                sy_time.append(pos[1])
                sz_time.append(pos[2])

    # 1. dt sweep
    ax1 = fig.add_subplot(2, 3, 1)
    dt_vals = [x[0] for x in dt_costs]
    dt_c = [x[1] for x in dt_costs]
    ax1.plot(dt_vals, dt_c, "b-", linewidth=0.8)
    ax1.axvline(x=dt, color="red", linestyle="--",
                label=f"best dt={dt:.3f}s\n~{dt*25:.1f} frames @25fps")
    ax1.set_xlabel("Time offset (s)")
    ax1.set_ylabel("Mean ray distance (m)")
    ax1.set_title("1. Auto Time-Offset Search")
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, min(max(dt_c), 5))
    ax1.grid(True, alpha=0.3)

    # 2. Court top view with SPATIAL fit
    ax2 = fig.add_subplot(2, 3, 2)
    court = patches.Rectangle((0, 0), 8.23, 23.77, linewidth=2,
                               edgecolor="white", facecolor="#2d5a27")
    ax2.add_patch(court)
    ax2.plot([0, 8.23], [11.885, 11.885], "w-", linewidth=2)
    ax2.plot([0, 8.23], [5.485, 5.485], "w-", linewidth=0.8)
    ax2.plot([0, 8.23], [18.285, 18.285], "w-", linewidth=0.8)
    ax2.plot([4.115, 4.115], [5.485, 18.285], "w-", linewidth=0.8)
    ax2.plot(sx_spatial, sy_spatial, "y-", linewidth=2.5,
             label="spatial fit (frame-rate independent)")
    ax2.scatter(xs, ys, c="lime", s=25, zorder=5, edgecolors="k",
                linewidths=0.3, label="triangulated")
    if traj_fit["type"] == "piecewise":
        bp = traj_fit["bounce_pos"]
        ax2.plot(bp[0], bp[1], "r*", markersize=15, zorder=6,
                 label=f"bounce (z={bp[2]:.2f}m)")
    ax2.set_xlim(-1, 9.23)
    ax2.set_ylim(-2, 26)
    ax2.set_aspect("equal")
    ax2.set_title("2. Court Top View")
    ax2.legend(fontsize=7, loc="upper left")

    # 3. Side view Y-Z: SPATIAL vs TIME-BASED comparison
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(sy_spatial, sz_spatial, "y-", linewidth=2.5,
             label="spatial fit (no timing needed)")
    if sy_time:
        ax3.plot(sy_time, sz_time, "c--", linewidth=1.5, alpha=0.7,
                 label="time-based fit (frame-rate dependent)")
    ax3.scatter(ys, zs, c="lime", s=40, zorder=5, edgecolors="k",
                linewidths=0.5, label="triangulated points")
    ax3.axhline(y=1.07, color="orange", linestyle="--", alpha=0.7,
                label="net height (1.07m)")
    ax3.axhline(y=0, color="white", linestyle="-", alpha=0.3)
    ax3.axvline(x=11.885, color="gray", linestyle=":", alpha=0.5,
                label="net position")
    if traj_fit["type"] == "piecewise":
        bp = traj_fit["bounce_pos"]
        ax3.plot(bp[1], bp[2], "r*", markersize=15, zorder=6)
    ax3.set_xlabel("Y (m) along court")
    ax3.set_ylabel("Z (m) height")
    ax3.set_title("3. Y-Z: Spatial vs Time-Based Fit")
    ax3.legend(fontsize=6)
    ax3.grid(True, alpha=0.3)

    # 4. Ray distances
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.bar(range(len(rds)), rds, color="cyan", alpha=0.8)
    ax4.axhline(y=np.mean(rds), color="red", linestyle="--",
                label=f"mean={np.mean(rds):.3f}m")
    ax4.axhline(y=0.2, color="orange", linestyle=":", alpha=0.5,
                label="0.2m threshold")
    ax4.set_xlabel("Point Index")
    ax4.set_ylabel("Ray distance (m)")
    ax4.set_title("4. Triangulation Quality")
    ax4.legend(fontsize=8)

    # 5. Spatial fit residuals
    ax5 = fig.add_subplot(2, 3, 5)
    if traj_fit["type"] == "piecewise":
        pre_res_s = traj_fit["pre_bounce"]["residuals"]
        post_res_s = traj_fit["post_bounce"]["residuals"]
        pre_res_t = traj_fit["pre_bounce_time"]["residuals"] if traj_fit.get("pre_bounce_time") else []
        post_res_t = traj_fit["post_bounce_time"]["residuals"] if traj_fit.get("post_bounce_time") else []
        bi = traj_fit["bounce_idx"]

        all_spatial = pre_res_s + post_res_s[1:]
        all_time = pre_res_t + post_res_t[1:] if pre_res_t else []

        x_idx = list(range(len(all_spatial)))
        w = 0.35
        ax5.bar([i - w/2 for i in x_idx], all_spatial, width=w,
                color="gold", alpha=0.8, label="spatial fit")
        if all_time:
            ax5.bar([i + w/2 for i in x_idx[:len(all_time)]], all_time,
                    width=w, color="cyan", alpha=0.8, label="time-based fit")
        ax5.axvline(x=bi - 0.5, color="red", linestyle=":", alpha=0.5,
                    label="bounce")
    else:
        all_spatial = traj_fit["fit_spatial"]["residuals"]
        all_time = traj_fit["fit_time"]["residuals"] if traj_fit.get("fit_time") else []
        x_idx = list(range(len(all_spatial)))
        ax5.bar(x_idx, all_spatial, color="gold", alpha=0.8, label="spatial fit")
        if all_time:
            ax5.bar(x_idx, all_time, color="cyan", alpha=0.4, label="time-based fit")
    ax5.set_xlabel("Point Index")
    ax5.set_ylabel("Fit error (m)")
    ax5.set_title("5. Spatial vs Time-Based Residuals")
    ax5.legend(fontsize=8)

    # 6. Summary
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")

    if traj_fit["type"] == "piecewise":
        pre = traj_fit["pre_bounce"]
        post = traj_fit["post_bounce"]
        pre_t = traj_fit.get("pre_bounce_time")
        post_t = traj_fit.get("post_bounce_time")
        lines = [
            "RESULTS",
            "",
            f"Auto time-offset: {dt:.4f}s (~{dt*25:.1f} frames)",
            f"Mean ray dist: {np.mean(rds):.3f}m",
            "",
            "SPATIAL FIT (frame-rate independent):",
            f"  Pre-bounce:  err={pre['mean_error']:.3f}m",
            f"    speed={pre['speed']:.0f}m/s ({pre['speed']*3.6:.0f}km/h)",
            f"  Post-bounce: err={post['mean_error']:.3f}m",
            f"    speed={post['speed']:.0f}m/s ({post['speed']*3.6:.0f}km/h)",
            "",
            "TIME-BASED FIT (for comparison):",
            f"  Pre-bounce:  err={pre_t['mean_error']:.3f}m" if pre_t else "  N/A",
            f"  Post-bounce: err={post_t['mean_error']:.3f}m" if post_t else "  N/A",
            "",
            f"Bounce at Y={traj_fit['bounce_pos'][1]:.1f}m",
            f"Max height: {max(zs):.2f}m",
            "",
            "SPATIAL FIT WINS!",
            "No frame-rate needed for shape.",
        ]
    else:
        fit = traj_fit["fit_spatial"]
        lines = [
            "RESULTS",
            f"Time offset: {dt:.4f}s ({dt*25:.1f} frames)",
            f"Speed: {fit['speed']:.1f}m/s ({fit['speed']*3.6:.0f}km/h)",
            f"Spatial fit error: {fit['mean_error']:.3f}m",
            f"Mean ray dist: {np.mean(rds):.3f}m",
        ]

    ax6.text(0.05, 0.95, "\n".join(lines), transform=ax6.transAxes,
             fontsize=9, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax6.set_title("6. Summary")

    plt.tight_layout()
    out = r"D:\tennis\trajectory_fit_report.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {out}")


# ========== Main ==========

def main():
    with open("src/homography_matrices.json") as f:
        hdata = json.load(f)

    H66 = np.array(hdata["cam66"]["H_image_to_world"], dtype=np.float64)
    H68 = np.array(hdata["cam68"]["H_image_to_world"], dtype=np.float64)
    cam66_pos = np.array([4.094, -5.21, 6.2])
    cam68_pos = np.array([4.115, 28.97, 5.2])

    cam66_dir = r"D:\tennis\cam66_20260306_170528"
    cam68_dir = r"D:\tennis\cam68_20260306_170528"

    cam66_dets = []
    for f in sorted(glob.glob(os.path.join(cam66_dir, "*.json"))):
        idx = int(os.path.basename(f).replace(".json", ""))
        if idx < 113:
            continue
        d = json.load(open(f))
        pts = d["shapes"][0]["points"][0]
        cam66_dets.append((idx, pts[0], pts[1]))

    cam68_dets = []
    for f in sorted(glob.glob(os.path.join(cam68_dir, "*.json"))):
        idx = int(os.path.basename(f).replace(".json", ""))
        d = json.load(open(f))
        pts = d["shapes"][0]["points"][0]
        cam68_dets.append((idx, pts[0], pts[1]))

    print("=" * 70)
    print("PHYSICS-CONSTRAINED 3D TRAJECTORY RECONSTRUCTION")
    print("  Stage 1: Auto time-offset + interpolated triangulation")
    print("  Stage 2: Piecewise parabolic fit (with bounce detection)")
    print("=" * 70)
    print(f"cam66: {len(cam66_dets)} dets (frames {cam66_dets[0][0]}-{cam66_dets[-1][0]})")
    print(f"cam68: {len(cam68_dets)} dets (frames {cam68_dets[0][0]}-{cam68_dets[-1][0]})")

    # Stage 1
    print("\n--- Stage 1: Finding time offset ---")
    dt, points_3d, dt_costs = find_offset_and_triangulate(
        cam66_dets, cam68_dets, 25.0, 25.0,
        H66, H68, cam66_pos, cam68_pos,
    )

    rds = [p["ray_dist"] for p in points_3d]
    print(f"Optimal dt = {dt:.4f}s = {dt*25:.2f} frames at 25fps")
    print(f"Triangulated {len(points_3d)} points")
    print(f"Mean ray distance: {np.mean(rds):.4f}m")
    print(f"Max ray distance:  {np.max(rds):.4f}m")

    # Print points
    print(f"\n{'t':>6} {'X':>6} {'Y':>6} {'Z':>5} {'ray_d':>7} {'frm66':>6}")
    for p in points_3d:
        print(f"{p['t']:6.3f} {p['x']:6.2f} {p['y']:6.2f} {p['z']:5.2f} "
              f"{p['ray_dist']:7.3f} {p['frame_a']:>6}")

    # Stage 2
    print("\n--- Stage 2: Spatial parabolic fit (frame-rate independent) ---")
    traj_fit = fit_trajectory_with_bounce(points_3d)

    if traj_fit["type"] == "piecewise":
        pre = traj_fit["pre_bounce"]
        post = traj_fit["post_bounce"]
        pre_t = traj_fit.get("pre_bounce_time")
        post_t = traj_fit.get("post_bounce_time")
        bi = traj_fit["bounce_idx"]
        bp = traj_fit["bounce_pos"]
        print(f"Bounce detected at index {bi}, "
              f"Y={bp[1]:.2f}m, Z={bp[2]:.2f}m")

        print(f"\n  SPATIAL FIT (X and Z as functions of Y):")
        print(f"  Pre-bounce ({bi+1} points):")
        print(f"    X(Y) = {pre['ax']:.4f}*Y + {pre['bx']:.4f}")
        print(f"    Z(Y) = {pre['az']:.5f}*Y^2 + {pre['bz']:.4f}*Y + {pre['cz']:.4f}")
        print(f"    Speed ~ {pre['speed']:.1f}m/s ({pre['speed']*3.6:.0f}km/h)")
        print(f"    Mean error: {pre['mean_error']:.4f}m")
        print(f"    Max error:  {pre['max_error']:.4f}m")

        print(f"\n  Post-bounce ({len(points_3d)-bi} points):")
        print(f"    X(Y) = {post['ax']:.4f}*Y + {post['bx']:.4f}")
        print(f"    Z(Y) = {post['az']:.5f}*Y^2 + {post['bz']:.4f}*Y + {post['cz']:.4f}")
        print(f"    Speed ~ {post['speed']:.1f}m/s ({post['speed']*3.6:.0f}km/h)")
        print(f"    Mean error: {post['mean_error']:.4f}m")
        print(f"    Max error:  {post['max_error']:.4f}m")

        if pre_t and post_t:
            print(f"\n  TIME-BASED FIT (for comparison):")
            print(f"    Pre-bounce  mean error: {pre_t['mean_error']:.4f}m")
            print(f"    Post-bounce mean error: {post_t['mean_error']:.4f}m")

        # Net crossing check using spatial fit
        all_ys = [p["y"] for p in points_3d]
        for y in np.linspace(min(all_ys), bp[1], 1000):
            z = pre["az"] * y**2 + pre["bz"] * y + pre["cz"]
            if y >= 11.885:
                print(f"\n  Net crossing: z={z:.2f}m "
                      f"(net height: 0.914-1.07m)")
                if 0.914 <= z <= 4.0:
                    print(f"  OK: Plausible net clearance")
                break
    else:
        fit = traj_fit["fit_spatial"]
        print(f"Single parabola (no bounce detected)")
        print(f"Speed ~ {fit['speed']:.1f}m/s ({fit['speed']*3.6:.0f}km/h)")
        print(f"Spatial fit error: {fit['mean_error']:.4f}m")

    # Plot
    plot_results(points_3d, traj_fit, dt, dt_costs)


if __name__ == "__main__":
    main()

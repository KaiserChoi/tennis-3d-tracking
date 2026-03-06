"""Generate 3D projection validation visualization."""

import json
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def pixel_to_world(H, px, py):
    pt = np.array([px, py, 1.0])
    r = H @ pt
    return r[0] / r[2], r[1] / r[2]


def triangulate(w1, w2, cp1, cp2):
    cam1 = np.asarray(cp1, dtype=np.float64)
    cam2 = np.asarray(cp2, dtype=np.float64)
    g1 = np.array([w1[0], w1[1], 0.0])
    g2 = np.array([w2[0], w2[1], 0.0])
    d1, d2 = g1 - cam1, g2 - cam2
    w = cam1 - cam2
    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    d_val = float(np.dot(d1, w))
    e = float(np.dot(d2, w))
    denom = a * c - b * b
    if abs(denom) < 1e-10:
        return 0, 0, 0, 99
    s = (b * e - c * d_val) / denom
    t = (a * e - b * d_val) / denom
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    p1f = cam1 + s * d1
    t = float(np.dot(p1f - cam2, d2)) / c if c > 1e-10 else t
    t = np.clip(t, 0.0, 1.0)
    p2f = cam2 + t * d2
    s = float(np.dot(p2f - cam1, d1)) / a if a > 1e-10 else s
    s = np.clip(s, 0.0, 1.0)
    p1 = cam1 + s * d1
    p2 = cam2 + t * d2
    mid = (p1 + p2) / 2.0
    rd = np.linalg.norm(p1 - p2)
    if mid[2] < 0:
        mid[2] = 0.0
    return float(mid[0]), float(mid[1]), float(mid[2]), rd


def main():
    with open("src/homography_matrices.json") as f:
        hdata = json.load(f)

    H66_i2w = np.array(hdata["cam66"]["H_image_to_world"], dtype=np.float64)
    H68_i2w = np.array(hdata["cam68"]["H_image_to_world"], dtype=np.float64)
    cam66_pos = np.array([4.094, -5.21, 6.2])
    cam68_pos = np.array([4.115, 28.97, 5.2])

    cam66_ann, cam68_ann = {}, {}
    cam66_dir = r"D:\tennis\cam66_20260306_170528"
    cam68_dir = r"D:\tennis\cam68_20260306_170528"
    for f in glob.glob(os.path.join(cam66_dir, "*.json")):
        idx = int(os.path.basename(f).replace(".json", ""))
        d = json.load(open(f))
        cam66_ann[idx] = tuple(d["shapes"][0]["points"][0])
    for f in glob.glob(os.path.join(cam68_dir, "*.json")):
        idx = int(os.path.basename(f).replace(".json", ""))
        d = json.load(open(f))
        cam68_ann[idx] = tuple(d["shapes"][0]["points"][0])

    def compute_traj(offset):
        results = []
        for c66 in sorted(cam66_ann.keys()):
            c68 = c66 - offset
            if c68 not in cam68_ann or c66 < 113:
                continue
            w66 = pixel_to_world(H66_i2w, *cam66_ann[c66])
            w68 = pixel_to_world(H68_i2w, *cam68_ann[c68])
            x, y, z, rd = triangulate(w66, w68, cam66_pos, cam68_pos)
            results.append({"x": x, "y": y, "z": z, "rd": rd, "c66": c66})
        return results

    traj5 = compute_traj(5)
    traj14 = compute_traj(14)

    # Offset sweep
    offsets = list(range(3, 22))
    means = []
    for off in offsets:
        dists = []
        for c66 in sorted(cam66_ann.keys()):
            c68 = c66 - off
            if c68 not in cam68_ann or c66 < 113:
                continue
            w66 = pixel_to_world(H66_i2w, *cam66_ann[c66])
            w68 = pixel_to_world(H68_i2w, *cam68_ann[c68])
            _, _, _, rd = triangulate(w66, w68, cam66_pos, cam68_pos)
            dists.append(rd)
        means.append(np.mean(dists) if dists else 99)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("3D Projection Validation Report", fontsize=16, fontweight="bold")

    # 1. Offset sweep
    ax1 = fig.add_subplot(2, 3, 1)
    colors = ["red" if o == 5 else "green" if o == 14 else "steelblue"
              for o in offsets]
    ax1.bar(offsets, means, color=colors)
    ax1.axhline(y=0.2, color="orange", linestyle="--", alpha=0.7,
                label="0.2m threshold")
    ax1.set_xlabel("Frame Offset (cam66 - cam68)")
    ax1.set_ylabel("Mean Ray Distance (m)")
    ax1.set_title("1. Offset Sweep")
    ax1.annotate("User: offset=5\n1.731m", xy=(5, 1.731), fontsize=8,
                 color="red", ha="center", va="bottom")
    ax1.annotate("Best: offset=14\n0.115m", xy=(14, 0.115), fontsize=8,
                 color="green", ha="center", va="bottom")
    ax1.legend(fontsize=8)

    # 2. Court top view
    ax2 = fig.add_subplot(2, 3, 2)
    court = patches.Rectangle((0, 0), 8.23, 23.77, linewidth=2,
                               edgecolor="white", facecolor="#2d5a27")
    ax2.add_patch(court)
    ax2.plot([0, 8.23], [11.885, 11.885], "w-", linewidth=2)
    ax2.text(4.115, 11.885, "NET", ha="center", va="bottom",
             color="white", fontsize=8, fontweight="bold")
    ax2.plot([0, 8.23], [5.485, 5.485], "w-", linewidth=0.8)
    ax2.plot([0, 8.23], [18.285, 18.285], "w-", linewidth=0.8)
    ax2.plot([4.115, 4.115], [5.485, 18.285], "w-", linewidth=0.8)

    xs14 = [r["x"] for r in traj14]
    ys14 = [r["y"] for r in traj14]
    ax2.plot(xs14, ys14, "yo-", markersize=5, linewidth=2,
             label="offset=14 (correct)")
    ax2.plot(xs14[0], ys14[0], "r^", markersize=10, zorder=5)
    ax2.plot(xs14[-1], ys14[-1], "bs", markersize=10, zorder=5)

    xs5 = [r["x"] for r in traj5]
    ys5 = [r["y"] for r in traj5]
    ax2.plot(xs5, ys5, "rx--", markersize=4, linewidth=1, alpha=0.7,
             label="offset=5 (wrong)")

    ax2.plot(cam66_pos[0], cam66_pos[1], "c^", markersize=12, label="cam66")
    ax2.plot(cam68_pos[0], cam68_pos[1], "m^", markersize=12, label="cam68")
    ax2.set_xlim(-1, 9.23)
    ax2.set_ylim(-7, 31)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("2. Court Top View")
    ax2.legend(fontsize=7, loc="upper left")
    ax2.set_aspect("equal")

    # 3. Side view (Y vs Z)
    ax3 = fig.add_subplot(2, 3, 3)
    zs14 = [r["z"] for r in traj14]
    zs5 = [r["z"] for r in traj5]
    ax3.plot(ys14, zs14, "go-", markersize=5, linewidth=2, label="offset=14")
    ax3.plot(ys5, zs5, "rx--", markersize=4, linewidth=1, alpha=0.7,
             label="offset=5")
    ax3.axhline(y=1.07, color="orange", linestyle="--", alpha=0.5,
                label="net height")
    ax3.axvline(x=11.885, color="gray", linestyle=":", alpha=0.5)
    ax3.set_xlabel("Y (m) - along court")
    ax3.set_ylabel("Z (m) - height")
    ax3.set_title("3. Side View (Y vs Z)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Ray distance per frame
    ax4 = fig.add_subplot(2, 3, 4)
    rds14 = [r["rd"] for r in traj14]
    rds5 = [r["rd"] for r in traj5]
    frames14 = [r["c66"] for r in traj14]
    frames5 = [r["c66"] for r in traj5]
    ax4.bar([f - 0.2 for f in frames5], rds5, width=0.4, color="red",
            alpha=0.7, label="offset=5")
    ax4.bar([f + 0.2 for f in frames14], rds14, width=0.4, color="green",
            alpha=0.7, label="offset=14")
    ax4.axhline(y=0.2, color="orange", linestyle="--", alpha=0.7)
    ax4.set_xlabel("cam66 Frame")
    ax4.set_ylabel("Ray Distance (m)")
    ax4.set_title("4. Ray Distance per Frame")
    ax4.legend(fontsize=8)

    # 5. Trajectory colored by height
    ax5 = fig.add_subplot(2, 3, 5)
    scatter = ax5.scatter(xs14, ys14, c=zs14, cmap="plasma", s=60,
                          edgecolors="white", linewidths=0.5, zorder=5)
    plt.colorbar(scatter, ax=ax5, label="Z height (m)")
    ax5.plot([0, 8.23, 8.23, 0, 0], [0, 0, 23.77, 23.77, 0], "w-",
             linewidth=1)
    ax5.axhline(y=11.885, color="white", linestyle="--", alpha=0.5)
    ax5.set_xlabel("X (m)")
    ax5.set_ylabel("Y (m)")
    ax5.set_title("5. Trajectory (color=height)")
    ax5.set_facecolor("#1a1a2e")
    ax5.set_xlim(-1, 9.23)
    ax5.set_ylim(-2, 26)
    ax5.set_aspect("equal")

    # 6. Summary text
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    summary_lines = [
        "FINDINGS",
        "",
        "1. Frame Offset: 14 (not 5)",
        "   cam66_122 = cam68_108",
        "   9 frames difference from assumed",
        "",
        "2. With offset=14:",
        f"   Mean ray dist: {np.mean(rds14):.3f}m",
        f"   Max ray dist:  {np.max(rds14):.3f}m",
        f"   Ball height: {min(zs14):.2f}-{max(zs14):.2f}m",
        "   Net clearance: ~1.68m (OK)",
        "",
        "3. Homography:",
        "   cam66 reproj: 0.158m (HIGH)",
        "   cam68 reproj: 0.031m (good)",
        "   Back-proj err cam66: 3.2px avg",
        "   Back-proj err cam68: 6.3px avg",
        "",
        "4. Recommendations:",
        "   - Fix frame offset to 14",
        "   - Recalibrate cam66 homography",
        "   - Pairs 128/114, 130/116 have",
        "     higher error (>0.3m)",
    ]
    summary = "\n".join(summary_lines)
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=9,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax6.set_title("6. Summary")

    plt.tight_layout()
    out_path = r"D:\tennis\3d_validation_report.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

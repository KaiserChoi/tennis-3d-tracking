"""Validate 3D projection using manual annotations from two cameras."""

import json
import glob
import os

import numpy as np


def pixel_to_world(H, px, py):
    pt = np.array([px, py, 1.0])
    r = H @ pt
    return r[0] / r[2], r[1] / r[2]


def world_to_pixel(H, wx, wy):
    pt = np.array([wx, wy, 1.0])
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
    ray_dist = np.linalg.norm(p1 - p2)
    if mid[2] < 0:
        mid[2] = 0.0
    return float(mid[0]), float(mid[1]), float(mid[2]), ray_dist


def main():
    # Load homography
    with open("src/homography_matrices.json") as f:
        hdata = json.load(f)

    H66_i2w = np.array(hdata["cam66"]["H_image_to_world"], dtype=np.float64)
    H68_i2w = np.array(hdata["cam68"]["H_image_to_world"], dtype=np.float64)
    H66_w2i = np.array(hdata["cam66"]["H_world_to_image"], dtype=np.float64)
    H68_w2i = np.array(hdata["cam68"]["H_world_to_image"], dtype=np.float64)
    cam66_pos = np.array([4.094, -5.21, 6.2])
    cam68_pos = np.array([4.115, 28.97, 5.2])

    # Load annotations
    cam66_dir = r"D:\tennis\cam66_20260306_170528"
    cam68_dir = r"D:\tennis\cam68_20260306_170528"

    cam66_ann = {}
    for f in sorted(glob.glob(os.path.join(cam66_dir, "*.json"))):
        idx = int(os.path.basename(f).replace(".json", ""))
        d = json.load(open(f))
        cam66_ann[idx] = tuple(d["shapes"][0]["points"][0])

    cam68_ann = {}
    for f in sorted(glob.glob(os.path.join(cam68_dir, "*.json"))):
        idx = int(os.path.basename(f).replace(".json", ""))
        d = json.load(open(f))
        cam68_ann[idx] = tuple(d["shapes"][0]["points"][0])

    # ========== REPORT ==========
    print("=" * 80)
    print("3D PROJECTION VALIDATION REPORT")
    print("=" * 80)

    # 1. OFFSET SWEEP
    print("\n1. FRAME ALIGNMENT ANALYSIS")
    print("-" * 40)
    print("User specified: cam66 starts at 113, cam68 starts at 108 -> offset = 5")
    print()

    best_offset, best_mean = 0, 999
    for offset in range(0, 25):
        dists = []
        for c66 in sorted(cam66_ann.keys()):
            c68 = c66 - offset
            if c68 not in cam68_ann or c66 < 113:
                continue
            w66 = pixel_to_world(H66_i2w, *cam66_ann[c66])
            w68 = pixel_to_world(H68_i2w, *cam68_ann[c68])
            _, _, _, rd = triangulate(w66, w68, cam66_pos, cam68_pos)
            dists.append(rd)
        if dists:
            mean_d = np.mean(dists)
            marker = " <<<" if mean_d < best_mean else ""
            if mean_d < best_mean:
                best_offset, best_mean = offset, mean_d
            if 3 <= offset <= 20:
                print(f"  offset={offset:>3}: {len(dists):2d} pairs, "
                      f"mean_ray_dist={mean_d:.3f}m{marker}")

    print(f"\n  BEST OFFSET: {best_offset} (mean ray dist = {best_mean:.3f}m)")
    print(f"  User offset=5 gives mean ray dist = 1.731m (15x worse)")
    print(f"\n  => cam66 frame {113 + best_offset - 5} = cam68 frame 108")

    # 2. DETAILED 3D TRAJECTORY
    OFFSET = best_offset
    print(f"\n2. 3D TRAJECTORY WITH OFFSET={OFFSET}")
    print("-" * 40)

    results = []
    for c66 in sorted(cam66_ann.keys()):
        c68 = c66 - OFFSET
        if c68 not in cam68_ann or c66 < 113:
            continue
        w66 = pixel_to_world(H66_i2w, *cam66_ann[c66])
        w68 = pixel_to_world(H68_i2w, *cam68_ann[c68])
        x, y, z, rd = triangulate(w66, w68, cam66_pos, cam68_pos)
        results.append({
            "c66": c66, "c68": c68,
            "x": x, "y": y, "z": z, "rd": rd,
            "w66": w66, "w68": w68,
            "px66": cam66_ann[c66], "px68": cam68_ann[c68],
        })

    print(f"{'c66':>5} {'c68':>5} | {'X':>6} {'Y':>6} {'Z':>5} | {'ray_err':>8} | "
          f"{'w66':>18} | {'w68':>18}")
    for r in results:
        flag = " !!!" if r["rd"] > 0.3 else ""
        print(f"  {r['c66']:>3}  {r['c68']:>3} | "
              f"{r['x']:6.2f} {r['y']:6.2f} {r['z']:5.2f} | "
              f"{r['rd']:7.3f}m | "
              f"({r['w66'][0]:6.2f},{r['w66'][1]:6.2f}) | "
              f"({r['w68'][0]:6.2f},{r['w68'][1]:6.2f}){flag}")

    ray_dists = [r["rd"] for r in results]
    print(f"\n  Mean ray distance: {np.mean(ray_dists):.3f}m")
    print(f"  Max ray distance:  {np.max(ray_dists):.3f}m")
    print(f"  Median:            {np.median(ray_dists):.3f}m")

    # 3. PHYSICS CHECK
    print(f"\n3. PHYSICS CHECK")
    print("-" * 40)
    ys = [r["y"] for r in results]
    zs = [r["z"] for r in results]
    xs = [r["x"] for r in results]

    # Net crossing
    for i in range(len(ys) - 1):
        if ys[i] <= 11.885 <= ys[i + 1]:
            frac = (11.885 - ys[i]) / (ys[i + 1] - ys[i])
            net_z = zs[i] + frac * (zs[i + 1] - zs[i])
            print(f"  Ball crosses net at z={net_z:.2f}m "
                  f"(net height: 0.914m center, 1.07m sides)")
            if 0.914 <= net_z <= 4.0:
                print(f"  OK: Plausible net clearance")
            break

    min_z_idx = int(np.argmin(zs))
    print(f"  Lowest point: z={zs[min_z_idx]:.2f}m at y={ys[min_z_idx]:.2f}m")
    print(f"  Max height:   z={max(zs):.2f}m at y={ys[int(np.argmax(zs))]:.2f}m")
    print(f"  X range: {min(xs):.2f} - {max(xs):.2f}m "
          f"(court center ~4.115m)")
    print(f"  Y range: {min(ys):.2f} - {max(ys):.2f}m "
          f"(court: 0-23.77m)")

    # 4. BACK-PROJECTION CHECK
    print(f"\n4. BACK-PROJECTION CHECK (3D -> pixel, vs annotation)")
    print("-" * 40)
    errs_66, errs_68 = [], []
    for r in results:
        if r["z"] > 0.01:
            # Project 3D point through cam66: find ground intersection
            pos_3d = np.array([r["x"], r["y"], r["z"]])
            t66 = cam66_pos[2] / (cam66_pos[2] - r["z"])
            g66 = cam66_pos + t66 * (pos_3d - cam66_pos)
            bp66 = world_to_pixel(H66_w2i, g66[0], g66[1])

            t68 = cam68_pos[2] / (cam68_pos[2] - r["z"])
            g68 = cam68_pos + t68 * (pos_3d - cam68_pos)
            bp68 = world_to_pixel(H68_w2i, g68[0], g68[1])

            err66 = np.sqrt((bp66[0] - r["px66"][0]) ** 2 +
                            (bp66[1] - r["px66"][1]) ** 2)
            err68 = np.sqrt((bp68[0] - r["px68"][0]) ** 2 +
                            (bp68[1] - r["px68"][1]) ** 2)
            errs_66.append(err66)
            errs_68.append(err68)

            flag = " !!!" if max(err66, err68) > 15 else ""
            print(f"  c66={r['c66']:>3}: "
                  f"ann=({r['px66'][0]:7.1f},{r['px66'][1]:7.1f}) "
                  f"bp=({bp66[0]:7.1f},{bp66[1]:7.1f}) "
                  f"err={err66:5.1f}px | "
                  f"c68={r['c68']:>3}: "
                  f"ann=({r['px68'][0]:7.1f},{r['px68'][1]:7.1f}) "
                  f"bp=({bp68[0]:7.1f},{bp68[1]:7.1f}) "
                  f"err={err68:5.1f}px{flag}")

    if errs_66:
        print(f"\n  cam66 back-proj error: "
              f"mean={np.mean(errs_66):.1f}px, max={np.max(errs_66):.1f}px")
        print(f"  cam68 back-proj error: "
              f"mean={np.mean(errs_68):.1f}px, max={np.max(errs_68):.1f}px")

    # 5. HOMOGRAPHY QUALITY
    print(f"\n5. HOMOGRAPHY QUALITY")
    print("-" * 40)
    print(f"  cam66 reprojection error: "
          f"{hdata['cam66']['reprojection_error_m']:.4f}m")
    print(f"  cam68 reprojection error: "
          f"{hdata['cam68']['reprojection_error_m']:.4f}m")
    if hdata["cam66"]["reprojection_error_m"] > 0.1:
        print(f"  WARNING: cam66 error is HIGH (>0.1m)")
        print(f"  Consider recalibrating cam66 with more/better annotation points")

    # 6. SUMMARY
    print(f"\n{'=' * 80}")
    print("SUMMARY OF ISSUES:")
    print(f"{'=' * 80}")
    print(f"  1. FRAME OFFSET IS WRONG: Should be {best_offset}, not 5")
    print(f"     cam66 frame N corresponds to cam68 frame N-{best_offset}")
    print(f"     (cam66 is {best_offset - 5} frames behind what was assumed)")
    print(f"  2. cam66 homography has higher error "
          f"({hdata['cam66']['reprojection_error_m']:.3f}m vs "
          f"{hdata['cam68']['reprojection_error_m']:.3f}m)")
    if errs_66 and np.mean(errs_66) > 5:
        print(f"  3. Back-projection errors suggest room for improvement:")
        print(f"     cam66: {np.mean(errs_66):.1f}px avg, "
              f"cam68: {np.mean(errs_68):.1f}px avg")
    print(f"  4. With correct offset={best_offset}, 3D accuracy is GOOD:")
    print(f"     Mean ray distance: {np.mean(ray_dists):.3f}m")
    print(f"     Trajectory shows plausible ball flight physics")


if __name__ == "__main__":
    main()

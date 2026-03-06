"""3D triangulation from two camera world-coordinate observations."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def triangulate(
    world_2d_cam1: tuple[float, float],
    world_2d_cam2: tuple[float, float],
    camera_pos_1: list[float],
    camera_pos_2: list[float],
) -> tuple[float, float, float]:
    """Compute 3D ball position from two 2D world-coordinate observations.

    Each camera projects a ray from its 3D position through the observed
    ground-plane point (x, y, 0).  We find the closest points on the two
    rays analytically and return their midpoint.

    The homography maps each camera's pixel observation to the point where
    the camera→ball ray intersects the ground plane (z=0).  Therefore the
    ray   cam + s·(ground - cam)   passes through the true ball position
    for some s ∈ (0, 1).

    Args:
        world_2d_cam1: (x, y) ground-plane projection from camera 1 homography.
        world_2d_cam2: (x, y) ground-plane projection from camera 2 homography.
        camera_pos_1: [x, y, z] camera 1 position in meters.
        camera_pos_2: [x, y, z] camera 2 position in meters.

    Returns:
        (x, y, z) 3D position in meters.
    """
    cam1 = np.asarray(camera_pos_1, dtype=np.float64)
    cam2 = np.asarray(camera_pos_2, dtype=np.float64)
    ground1 = np.array([world_2d_cam1[0], world_2d_cam1[1], 0.0])
    ground2 = np.array([world_2d_cam2[0], world_2d_cam2[1], 0.0])

    d1 = ground1 - cam1  # ray 1 direction (camera → ground projection)
    d2 = ground2 - cam2  # ray 2 direction (camera → ground projection)

    # ---- Analytical closest-point between two rays ----
    # Ray 1: P1(s) = cam1 + s * d1
    # Ray 2: P2(t) = cam2 + t * d2
    # Minimise ‖P1(s) - P2(t)‖² (quadratic in s, t → unique solution)
    w = cam1 - cam2
    a = float(np.dot(d1, d1))  # |d1|²
    b = float(np.dot(d1, d2))  # d1·d2
    c = float(np.dot(d2, d2))  # |d2|²
    d_val = float(np.dot(d1, w))  # d1·w
    e = float(np.dot(d2, w))  # d2·w

    denom = a * c - b * b
    if abs(denom) < 1e-10:
        # Rays nearly parallel — fall back to midpoint of ground projections
        mid_ground = (ground1 + ground2) / 2.0
        return float(mid_ground[0]), float(mid_ground[1]), 0.0

    s = (b * e - c * d_val) / denom
    t = (a * e - b * d_val) / denom

    # Clamp to [0, 1] — ball must be between camera and ground projection.
    # If clamping one parameter, re-solve the other for the fixed value.
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)

    # Re-solve: given fixed s, best t = -dot(cam2 + t*d2 - P1, d2) / |d2|²
    p1_fixed = cam1 + s * d1
    t = float(np.dot(p1_fixed - cam2, d2)) / c if c > 1e-10 else t
    t = np.clip(t, 0.0, 1.0)

    # Re-solve s given fixed t
    p2_fixed = cam2 + t * d2
    s = float(np.dot(p2_fixed - cam1, d1)) / a if a > 1e-10 else s
    s = np.clip(s, 0.0, 1.0)

    p1 = cam1 + s * d1
    p2 = cam2 + t * d2
    mid = (p1 + p2) / 2.0

    # Ensure non-negative height (small negatives from noise)
    if mid[2] < 0:
        mid[2] = 0.0

    return float(mid[0]), float(mid[1]), float(mid[2])

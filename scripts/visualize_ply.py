"""
visualize_ply.py
----------------
Simple Open3D-based visualization utility
for inspecting .ply or .pcd point clouds.
"""

from pathlib import Path
import numpy as np
import open3d as o3d


# ============================================================
# Configuration
# ============================================================

POINT_CLOUD_PATH = Path("ply_folder/phone_holder.ply")


# ============================================================
# Main
# ============================================================

def main() -> None:
    if not POINT_CLOUD_PATH.exists():
        raise FileNotFoundError(f"Point cloud not found: {POINT_CLOUD_PATH}")

    # --------------------------------------------------------
    # 1. Load point cloud
    # --------------------------------------------------------
    pcd = o3d.io.read_point_cloud(str(POINT_CLOUD_PATH))

    # --------------------------------------------------------
    # 2. Basic inspection
    # --------------------------------------------------------
    print("[INFO] Point cloud summary:")
    print(pcd)

    points = np.asarray(pcd.points)
    print(f"[INFO] Number of points: {points.shape[0]}")

    if pcd.has_colors():
        print("[INFO] Point cloud contains RGB color data")

    # --------------------------------------------------------
    # 3. Visualization
    # --------------------------------------------------------
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="TRELLIS Point Cloud Viewer",
        width=1280,
        height=720,
        point_show_normal=False
    )


if __name__ == "__main__":
    main()

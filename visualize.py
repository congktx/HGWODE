import matplotlib.pyplot as plt
import numpy as np
# ----------------------------- Visualization -----------------------------
def visualize_path_3d(start, end, threats, path):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    # Vẽ đường bay
    ax.plot(path[:,0], path[:,1], path[:,2], '-o', color="blue", label="UAV Path")

    # Vẽ điểm start/end
    ax.scatter(start[0], start[1], start[2], color="green", s=100, label="Start")
    ax.scatter(end[0], end[1], end[2], color="red", s=100, label="End")

    # Vẽ threats dưới dạng trụ (simplified as cylinders)
    for (cx, cy, r, h) in threats:
        z = np.linspace(0, h, 20)
        theta = np.linspace(0, 2*np.pi, 30)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = cx + r*np.cos(theta_grid)
        y_grid = cy + r*np.sin(theta_grid)
        ax.plot_surface(x_grid, y_grid, z_grid, color='orange', alpha=0.3)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D UAV Path Planning with GWO")
    ax.legend()
    plt.show()
    
def visualize_paths_3d_and_topview(start, end, threats, paths_dict):
    fig = plt.figure(figsize=(14,6))

    # --------- 3D View ---------
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(start[0], start[1], start[2], color="green", s=100, label="Start")
    ax1.scatter(end[0], end[1], end[2], color="red", s=100, label="End")

    for (cx, cy, r, h) in threats:
        z = np.linspace(0, h, 20)
        theta = np.linspace(0, 2*np.pi, 30)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = cx + r*np.cos(theta_grid)
        y_grid = cy + r*np.sin(theta_grid)
        ax1.plot_surface(x_grid, y_grid, z_grid, color='orange', alpha=0.3)

    colors = ["blue","purple","brown","cyan","magenta","black"]
    for i,(step,path) in enumerate(paths_dict.items()):
        ax1.plot(path[:,0], path[:,1], path[:,2], '-o', color=colors[i%len(colors)], label=f"Iter {step}")

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("3D UAV Path Evolution")
    ax1.legend()

    # --------- Top View (XY) ---------
    ax2 = fig.add_subplot(122)
    ax2.scatter(start[0], start[1], color="green", s=100, label="Start")
    ax2.scatter(end[0], end[1], color="red", s=100, label="End")

    # Threats projected as circles
    for (cx, cy, r, h) in threats:
        circle = plt.Circle((cx, cy), r, color='orange', alpha=0.3)
        ax2.add_patch(circle)

    for i,(step,path) in enumerate(paths_dict.items()):
        ax2.plot(path[:,0], path[:,1], '-o', color=colors[i%len(colors)], label=f"Iter {step}")

    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_aspect('equal')
    ax2.set_title("Top View (XY)")
    ax2.legend()

    plt.tight_layout()
    plt.show()

import numpy as np

# ---------- B-spline: open-uniform, clamped, cubic ----------
def _find_knot_span(n, p, u, U):
    if u >= U[n+1]:
        return n
    low, high = p, n+1
    mid = (low + high)//2
    while u < U[mid] or u >= U[mid+1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high)//2
    return mid

def _de_boor(p, U, P, u):
    n = P.shape[0]-1
    k = _find_knot_span(n, p, u, U)
    d = [P[j].copy() for j in range(k-p, k+1)]
    for r in range(1, p+1):
        for j in range(p, r-1, -1):
            i = k - p + j
            denom = U[i+p+1-r] - U[i]
            alpha = 0.0 if denom == 0 else (u - U[i]) / denom
            d[j] = (1.0 - alpha) * d[j-1] + alpha * d[j]
    return d[p]

def bspline_curve(points, degree=3, num_samples=300):
    """
    Trả về các điểm trên đường B-spline clamped cubic đi qua điểm đầu/cuối.
    'points' là polyline (start -> control points -> end).
    """
    P = np.asarray(points, dtype=float)
    n = P.shape[0]-1
    p = degree
    if n < p:  
        return P.copy()

    m = n + p + 1
    U = np.zeros(m+1)
    U[-(p+1):] = 1.0
    if (n - p) > 0:
        internal = np.linspace(1/(n-p+1), (n-p)/(n-p+1), n-p)
        U[p+1:m-p] = internal

    us = np.linspace(0, 1, num_samples)
    C = np.vstack([_de_boor(p, U, P, float(u)) for u in us])
    return C


def visualize_paths_3d_and_topview_with_bspline(start, end, threats, paths_dict, samples=300):
    fig = plt.figure(figsize=(14,6))
    color_cycle = ["tab:blue","tab:purple","tab:brown","tab:cyan","tab:orange","tab:gray"]

    # -------- 3D view --------
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(start[0], start[1], start[2], s=100, label="Start")
    ax1.scatter(end[0], end[1], end[2], s=100, label="End")

    # Threats
    for (cx, cy, r, h) in threats:
        z = np.linspace(0, h, 20)
        theta = np.linspace(0, 2*np.pi, 30)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = cx + r*np.cos(theta_grid)
        y_grid = cy + r*np.sin(theta_grid)
        ax1.plot_surface(x_grid, y_grid, z_grid, alpha=0.25)

    for i, (step, path) in enumerate(sorted(paths_dict.items())):
        color = color_cycle[i]
        C = bspline_curve(path, degree=3, num_samples=samples)
        ax1.plot(C[:,0], C[:,1], C[:,2], color=color, linewidth=2.0, label=f"Iter {step} (B-spline)")

    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.set_title("3D UAV Path Evolution (B-spline)")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02,1.0))

    # -------- Top view (XY) --------
    ax2 = fig.add_subplot(122)
    ax2.scatter(start[0], start[1], s=100, label="Start")
    ax2.scatter(end[0], end[1], s=100, label="End")

    for (cx, cy, r, h) in threats:
        circle = plt.Circle((cx, cy), r, alpha=0.25)
        ax2.add_patch(circle)

    for i,(step, path) in enumerate(sorted(paths_dict.items())):
        color=color_cycle[i]
        C = bspline_curve(path, degree=3, num_samples=samples)
        ax2.plot(C[:,0], C[:,1], color=color, linewidth=2.0, label=f"Iter {step} (B-spline)")

    ax2.set_aspect('equal', 'box')
    ax2.set_xlabel("X"); ax2.set_ylabel("Y")
    ax2.set_title("Top View (XY)")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02,1.0))
    plt.tight_layout(); plt.show()
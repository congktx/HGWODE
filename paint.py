import numpy as np
import matplotlib.pyplot as plt

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
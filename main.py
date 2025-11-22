import numpy as np
import paint
import multi_finess

def path_length(points):
    return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))

def collision_penalty(points, threats):
    penalty = 0
    for i in range(len(points)-1):
        p1, p2 = points[i], points[i+1]
        for (cx, cy, r, h) in threats:
            for t in np.linspace(0, 1, 5):  
                p = p1 + t*(p2-p1)
                dist_xy = np.sqrt((p[0]-cx)**2 + (p[1]-cy)**2)
                if dist_xy < r and 0 <= p[2] <= h:
                    penalty += 1e4
    return penalty

def angle_penalty(points, theta_max=np.pi/3):
    penalty = 0
    for i in range(1, len(points)-1):
        v1 = points[i] - points[i-1]
        v2 = points[i+1] - points[i]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        if angle > theta_max:
            penalty += (angle - theta_max) * 100
    return penalty

def fitness_function(path, threats):
    return path_length(path) + collision_penalty(path, threats) + angle_penalty(path)

# -----------------------------
# Hybrid Grey Wolf Optimizer and Differential Evolution
# -----------------------------
def HGWODE_UAV_3D(start, end, threats, num_wolves=20, num_points=5, 
                  max_iter=100, lb=0, ub=1000, F=0.5, CR=0.9):
    dim = num_points*3
    wolves = np.random.uniform(lb, ub, (num_wolves, dim))

    def decode(wolf):
        controls = wolf.reshape(num_points, 3)
        return np.vstack([start, controls, end])

    def fitness_function(path):
        return path_length(path) + collision_penalty(path, threats) + angle_penalty(path)

    fitness = np.array([fitness_function(decode(w)) for w in wolves])
    alpha, beta, delta = np.argsort(fitness)[:3]
    alpha_pos, alpha_score = wolves[alpha].copy(), fitness[alpha]

    for t in range(max_iter):
        a = 2 - 2 * (t / max_iter)

        # --- GWO  ---
        for i in range(num_wolves):
            X = wolves[i].copy()
            for leader_pos in [wolves[alpha], wolves[beta], wolves[delta]]:
                r1, r2 = np.random.rand(dim), np.random.rand(dim)
                A = 2*a*r1 - a
                C = 2*r2
                D = np.abs(C*leader_pos - X)
                X1 = leader_pos - A*D
                if np.array_equal(leader_pos, wolves[alpha]):
                    X_alpha = X1
                elif np.array_equal(leader_pos, wolves[beta]):
                    X_beta = X1
                else:
                    X_delta = X1
            new_pos = (X_alpha + X_beta + X_delta) / 3
            new_pos = np.clip(new_pos, lb, ub)
            wolves[i] = new_pos
            fitness[i] = fitness_function(decode(new_pos))

        # --- DE  ---
        for i in range(num_wolves):
            idxs = list(range(num_wolves))
            idxs.remove(i)
            r1, r2, r3 = wolves[np.random.choice(idxs, 3, replace=False)]
            V = np.clip(r1 + F*(r2 - r3), lb, ub)

            cross_points = np.random.rand(dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            U = np.where(cross_points, V, wolves[i])
            U_fit = fitness_function(decode(U))

            if U_fit < fitness[i]:
                wolves[i], fitness[i] = U, U_fit

        # Cập nhật alpha, beta, delta
        best_indices = np.argsort(fitness)[:3]
        alpha = best_indices[0]
        alpha_pos, alpha_score = wolves[alpha].copy(), fitness[alpha]

    return decode(alpha_pos), alpha_score

start = np.array([0, 0, 0])
end   = np.array([1000, 1000, 1000])
threats = [
  (300,150,75,1000),
  (250,600,100,800),
  (600,100,100,500),
  (500,750,100,1000),
  (850,550,75,500),
  (450,300,75,750),
  (750,350,50,1000),
  (200,200,75,1000),
  (800,800,50,800),
  (600,600,80,800),
]

best_path, best_cost = HGWODE_UAV_3D(
  start, 
  end, 
  threats, 
  num_wolves=30, 
  num_points=8, 
  max_iter=200
)
print("Best cost: ", best_cost)
print("Best path: ", best_path)
paint.visualize_paths_3d_and_topview_with_bspline(
  start, 
  end, 
  threats, 
  {"0":best_path},
  samples=400
)

# best_pos, best_val = multi_finess.HGWODE(multi_finess.cec_functions["f3"], dim=30, max_iter=500)
# print("f3 best value:", best_val)

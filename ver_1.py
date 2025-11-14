import numpy as np
import visualize

# ----------------------------- Fitness components -----------------------------
def path_length(points):
    return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))

def intersects_cylinder(p1, p2, cx, cy, r, h):
    p1, p2 = np.array(p1), np.array(p2)
    d = p2 - p1 
    a = d[0]**2 + d[1]**2
    b = 2*((p1[0]-cx)*d[0] + (p1[1]-cy)*d[1])
    c = (p1[0]-cx)**2 + (p1[1]-cy)**2 - r**2

    if abs(a) < 1e-12:
        if c <= 0:
            zmin, zmax = sorted([p1[2], p2[2]])
            return not (zmax < 0 or zmin > h)
        else:
            return False

    disc = b*b - 4*a*c
    if disc < 0:
        return False  

    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc)/(2*a)
    t2 = (-b + sqrt_disc)/(2*a)

    for t in [t1, t2]:
        if 0 <= t <= 1:
            z = p1[2] + d[2]*t
            if 0 <= z <= h:
                return True

    for t in [0,1]:
        x,y,z = p1 + d*t
        if (x-cx)**2+(y-cy)**2 <= r**2 and 0 <= z <= h:
            return True

    return False

def collision_penalty(points, threats):
    penalty = 0
    for i in range(len(points)-1):
        p1, p2 = points[i], points[i+1]
        for (cx, cy, r, h) in threats:
          if intersects_cylinder(p1,p2,cx,cy,r,h):
            penalty += 1e9
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

# ----------------------------- HGWODE for UAV 3D -----------------------------
def HGWODE_UAV_3D(start, end, threats, num_wolves=20, num_points=5, 
                  max_iter=600, lb=0, ub=1000, F=0.5, CR=0.9,
                  checkpoints=[100,200,300,400,500,600]):

    dim = num_points*3
    wolves = np.random.uniform(lb, ub, (num_wolves, dim))

    def decode(wolf):
        controls = wolf.reshape(num_points, 3)
        return np.vstack([start, controls, end])

    fitness = np.array([fitness_function(decode(w), threats) for w in wolves])
    alpha, beta, delta = np.argsort(fitness)[:3]
    alpha_pos, alpha_score = wolves[alpha].copy(), fitness[alpha]

    best_paths = {}
    best_scores = {}

    for t in range(1, max_iter+1):
        a = 2 - 2*(t/max_iter)

        # --- GWO update ---
        for i in range(num_wolves):
            X = wolves[i].copy()
            for leader_pos in [wolves[alpha], wolves[beta], wolves[delta]]:
                r1, r2 = np.random.rand(dim), np.random.rand(dim)
                A = 2*a*r1 - a
                C = 2*r2
                D = np.abs(C*leader_pos - X)
                X1 = leader_pos - A*D
                if np.array_equal(leader_pos, wolves[alpha]): X_alpha = X1
                elif np.array_equal(leader_pos, wolves[beta]): X_beta = X1
                else: X_delta = X1
            new_pos = (X_alpha + X_beta + X_delta)/3
            new_pos = np.clip(new_pos, lb, ub)
            wolves[i] = new_pos
            fitness[i] = fitness_function(decode(new_pos), threats)

        # --- DE update ---
        for i in range(num_wolves):
            idxs = list(range(num_wolves)); idxs.remove(i)
            r1, r2, r3 = wolves[np.random.choice(idxs, 3, replace=False)]
            V = np.clip(r1 + F*(r2-r3), lb, ub)
            cross_points = np.random.rand(dim) < CR
            if not np.any(cross_points): cross_points[np.random.randint(0, dim)] = True
            U = np.where(cross_points, V, wolves[i])
            U_fit = fitness_function(decode(U), threats)
            if U_fit < fitness[i]:
                wolves[i], fitness[i] = U, U_fit

        # Update leaders
        best_indices = np.argsort(fitness)[:3]
        alpha = best_indices[0]
        alpha_pos, alpha_score = wolves[alpha].copy(), fitness[alpha]

        if t in checkpoints:
            best_paths[t] = decode(alpha_pos)
            best_scores[t] = alpha_score

    return best_paths, best_scores

# ----------------------------- Demo run -----------------------------
start = np.array([0,0,0])
end   = np.array([1000,1000,1000])
threats = [
  (150,220,50,500),
  (300,300,100,800),
  (600,600,150,600),
  (500,200,120,1000)
]

best_paths, best_scores = HGWODE_UAV_3D(start,end,threats,num_wolves=25,num_points=5,max_iter=2000,checkpoints=[200,500,1000,1500,2000])

print(best_scores)

visualize.visualize_paths_3d_and_topview_with_bspline(start,end,threats,best_paths)

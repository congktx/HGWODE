import numpy as np
import paint

def path_length(points):
    return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))

def collision_penalty(points, threats, samples=20):
    penalty = 0
    for i in range(len(points)-1):
        p1, p2 = points[i], points[i+1]
        for (cx, cy, r, h) in threats:
            for t in np.linspace(0, 1, samples):
                p = p1 + t*(p2-p1)
                dist_xy = np.hypot(p[0]-cx, p[1]-cy)
                if dist_xy < r and 0 <= p[2] <= h:
                    penetration = r - dist_xy
                    penalty += 1e6 * penetration  
    return penalty

def angle_penalty(points, theta_max=np.pi/3):
    penalty = 0
    for i in range(1, len(points)-1):
        v1 = points[i] - points[i-1]
        v2 = points[i+1] - points[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            continue
        cos_angle = np.dot(v1, v2) / (n1 * n2)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        deviation = np.pi - angle
        if deviation > theta_max:
            penalty += (deviation - theta_max) ** 2 * 1e5
    return penalty

def base_fitness(path, threats):
    return path_length(path) + collision_penalty(path, threats) + angle_penalty(path)
class QLearner:
    def __init__(self, n_states=3, n_actions=3, alpha=0.2, gamma=0.9, epsilon=0.1, seed=None):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def choose(self, state):
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.Q.shape[1])
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, reward, s_next):
        best_next = np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (reward + self.gamma * best_next - self.Q[s, a])

def state_from_iter(t, max_iter, n_states=3):
    idx = int((t / max_iter) * n_states)
    return min(n_states-1, max(0, idx))

def HGWODE_QLEARN_UAV_3D(start, end, threats, num_wolves=20, num_points=5,
                         max_iter=100, lb=0, ub=1000, F=0.5, CR=0.9,
                         seed=None, q_params=None):
    rng = np.random.default_rng(seed)
    dim = num_points * 3
    
    diagonal = np.linspace(start, end, num_points)
    wolves = []
    for _ in range(num_wolves):
        wolf = []
        for point in diagonal:
            noisy_point = point + rng.uniform(-200, 200, 3)
            wolf.extend(np.clip(noisy_point, lb, ub))
        print(wolf)
        wolves.append(wolf)
    wolves = np.array(wolves)

    def decode(wolf):
        controls = wolf.reshape(num_points, 3)
        return np.vstack([start, controls, end])
    
    def fitness_of_wolf_vec(w):
        return base_fitness(decode(w), threats)

    fitness = np.array([fitness_of_wolf_vec(w) for w in wolves])
    best_idx = np.argmin(fitness)
    sorted_idx = np.argsort(fitness)
    alpha_idx, beta_idx, delta_idx = sorted_idx[0], sorted_idx[1], sorted_idx[2]
    alpha_pos, alpha_score = wolves[alpha_idx].copy(), fitness[alpha_idx]

    q_params = q_params or {}
    q = QLearner(seed=seed, **{k: q_params.get(k, v) for k, v in
                              [('n_states', 3), ('n_actions', 3), ('alpha', 0.2), ('gamma', 0.9), ('epsilon', 0.1)]})

    for t in range(max_iter):
        a_coef = 2 - 2 * (t / max_iter)
        state = state_from_iter(t, max_iter, n_states=q.Q.shape[0])

        leader_pos = [wolves[alpha_idx].copy(), wolves[beta_idx].copy(), wolves[delta_idx].copy()]

        for i in range(num_wolves):
            old_fit = fitness[i].copy()
            action = q.choose(state) 

            new_pos = wolves[i].copy()

            if action in (0, 2):
                X = wolves[i].copy()
                Xs = []
                for lp in leader_pos:
                    r1 = rng.random(dim)
                    r2 = rng.random(dim)
                    A = 2 * a_coef * r1 - a_coef
                    C = 2 * r2
                    D = np.abs(C * lp - X)
                    Xs.append(lp - A * D)
                new_pos = (Xs[0] + Xs[1] + Xs[2]) / 3.0
                new_pos = np.clip(new_pos, lb, ub)

            if action in (1, 2):
                idxs = [idx for idx in range(num_wolves) if idx != i]
                r1, r2, r3 = wolves[rng.choice(idxs, 3, replace=False)]
                V = np.clip(r1 + F * (r2 - r3), lb, ub)
                cross = rng.random(dim) < CR
                if not np.any(cross):
                    cross[rng.integers(0, dim)] = True
                U = np.where(cross, V, wolves[i])
                U_fit = fitness_of_wolf_vec(U)
                if U_fit < base_fitness(decode(new_pos), threats):
                    new_pos, new_fit = U, U_fit
                else:
                    new_fit = base_fitness(decode(new_pos), threats)
            else:
                new_fit = fitness_of_wolf_vec(new_pos)

            if new_fit < fitness[i]:
                wolves[i], fitness[i] = new_pos, new_fit
            else:
                new_fit = fitness[i]

            reward = old_fit - new_fit  
            next_state = state_from_iter(t+1, max_iter, n_states=q.Q.shape[0])
            q.update(state, action, reward, next_state)

        best_indices = np.argsort(fitness)[:3]
        alpha_idx, beta_idx, delta_idx = best_indices[0], best_indices[1], best_indices[2]
        alpha_pos, alpha_score = wolves[alpha_idx].copy(), fitness[alpha_idx]

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

best_path, best_cost = HGWODE_QLEARN_UAV_3D(
  start, end, threats, 
  num_wolves=30, 
  num_points=6, 
  max_iter=6000, 
  seed=42
)
print("Best cost (UAV Q\-learn):", best_cost)
print("Best path (UAV Q\-learn):", best_path)
paint.visualize_paths_3d_and_topview(
  start, 
  end, 
  threats, 
  {"0":best_path}
)

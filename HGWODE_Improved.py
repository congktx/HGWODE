import numpy as np
from typing import Callable, Tuple, Optional
import paint

class QLearner:
    def __init__(self, n_states=3, n_actions=3, alpha=0.2, gamma=0.9, 
                 epsilon_start=0.3, epsilon_end=0.05, seed=None):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon = epsilon_start
        self.rng = np.random.default_rng(seed)
    
    def choose(self, state: int) -> int:
        """Choose action using epsilon-greedy strategy"""
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.Q.shape[1])
        return int(np.argmax(self.Q[state]))
    
    def update(self, s: int, a: int, reward: float, s_next: int):
        """Update Q-table using Q-learning rule"""
        best_next = np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (reward + self.gamma * best_next - self.Q[s, a])
    
    def decay_epsilon(self, t: int, max_iter: int):
        """Decay epsilon linearly from epsilon_start to epsilon_end"""
        progress = t / max_iter
        self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * progress

def state_from_iter(t: int, max_iter: int, n_states: int = 3) -> int:
    """Map iteration to state (early/mid/late phase)"""
    idx = int((t / max_iter) * n_states)
    return min(n_states - 1, max(0, idx))

def HGWODE_QLEARN_OPTIMIZER(
    f: Callable[[np.ndarray], float],
    dim: int,
    lb: float = -100.0,
    ub: float = 100.0,
    NP: int = 50,
    max_iter: int = 3000,
    F: float = 0.5,
    CR: float = 0.9,
    seed: Optional[int] = None,
    q_params: Optional[dict] = None
) -> Tuple[np.ndarray, float, list]:
    
    rng = np.random.default_rng(seed)
    
    # Q-learning parameters
    if q_params is None:
        q_params = {}
    n_states = q_params.get('n_states', 3)
    n_actions = q_params.get('n_actions', 3)
    alpha = q_params.get('alpha', 0.2)
    gamma = q_params.get('gamma', 0.9)
    epsilon_start = q_params.get('epsilon_start', 0.3)
    epsilon_end = q_params.get('epsilon_end', 0.05)
    
    # Initialize Q-learner
    q = QLearner(n_states, n_actions, alpha, gamma, epsilon_start, epsilon_end, seed)
    
    # Initialize population
    wolves = rng.uniform(lb, ub, (NP, dim))
    fitness = np.array([f(w) for w in wolves])
    
    # Track best solution
    best_idx = np.argmin(fitness)
    best_x = wolves[best_idx].copy()
    best_f = fitness[best_idx]
    history = [best_f]
    
    # Main loop
    for t in range(max_iter):
        # Decay epsilon
        q.decay_epsilon(t, max_iter)
        
        # Sort population by fitness
        sorted_idx = np.argsort(fitness)
        alpha_idx = sorted_idx[0]
        beta_idx = sorted_idx[1]
        delta_idx = sorted_idx[2]
        
        alpha_pos = wolves[alpha_idx]
        beta_pos = wolves[beta_idx]
        delta_pos = wolves[delta_idx]
        
        # Linearly decrease a from 2 to 0
        a_coef = 2.0 * (1.0 - t / max_iter)
        
        # Ranking-based probability for mutation
        ranks = np.arange(NP, 0, -1)
        probs = ranks / ranks.sum()
        
        # Get current state
        state = state_from_iter(t, max_iter, n_states)
        
        # Update each wolf
        for i in range(NP):
            X = wolves[i]
            old_fit = fitness[i]
            
            # Choose action
            action = q.choose(state)
            
            # ==================== ACTION 0: GWO UPDATE ====================
            if action == 0:
                # Calculate fitness-based weights (Eq. 22 from paper)
                f_alpha = fitness[alpha_idx]
                f_beta = fitness[beta_idx]
                f_delta = fitness[delta_idx]
                
                # Inverse fitness weights (smaller fitness = larger weight)
                w_alpha = 1.0 / (f_alpha + 1e-10)
                w_beta = 1.0 / (f_beta + 1e-10)
                w_delta = 1.0 / (f_delta + 1e-10)
                w_sum = w_alpha + w_beta + w_delta
                
                w_alpha /= w_sum
                w_beta /= w_sum
                w_delta /= w_sum
                
                # Generate random coefficients for each leader
                r1_alpha, r2_alpha = rng.random(dim), rng.random(dim)
                r1_beta, r2_beta = rng.random(dim), rng.random(dim)
                r1_delta, r2_delta = rng.random(dim), rng.random(dim)
                
                A_alpha = 2 * a_coef * r1_alpha - a_coef
                C_alpha = 2 * r2_alpha
                A_beta = 2 * a_coef * r1_beta - a_coef
                C_beta = 2 * r2_beta
                A_delta = 2 * a_coef * r1_delta - a_coef
                C_delta = 2 * r2_delta
                
                # Calculate positions influenced by alpha, beta, delta (Eq. 23)
                D_alpha = np.abs(C_alpha * alpha_pos - X)
                X_alpha = alpha_pos - A_alpha * D_alpha
                
                D_beta = np.abs(C_beta * beta_pos - X)
                X_beta = beta_pos - A_beta * D_beta
                
                D_delta = np.abs(C_delta * delta_pos - X)
                X_delta = delta_pos - A_delta * D_delta
                
                # Weighted combination
                new_pos = w_alpha * X_alpha + w_beta * X_beta + w_delta * X_delta
                new_pos = np.clip(new_pos, lb, ub)
                new_fit = f(new_pos)
                
            # ==================== ACTION 1: DE UPDATE ====================
            elif action == 1:
                # Rank-based selection for r1, r2, r3
                candidates = list(range(NP))
                candidates.remove(i)
                
                # Select r1, r2, r3 using ranking probability
                r1 = rng.choice(candidates, p=probs[candidates] / probs[candidates].sum())
                candidates_r2 = [c for c in candidates if c != r1]
                r2 = rng.choice(candidates_r2, p=probs[candidates_r2] / probs[candidates_r2].sum())
                candidates_r3 = [c for c in candidates if c not in [r1, r2]]
                r3 = rng.choice(candidates_r3, p=probs[candidates_r3] / probs[candidates_r3].sum())
                
                # Mutation: DE/rand/1
                V = wolves[r1] + F * (wolves[r2] - wolves[r3])
                V = np.clip(V, lb, ub)
                
                # Crossover
                jrand = rng.integers(0, dim)
                U = np.copy(X)
                for j in range(dim):
                    if rng.random() < CR or j == jrand:
                        U[j] = V[j]
                
                new_pos = U
                new_fit = f(new_pos)
                
            # ==================== ACTION 2: COMBINED (GWO + DE BLEND) ====================
            else:
                # GWO component
                f_alpha = fitness[alpha_idx]
                f_beta = fitness[beta_idx]
                f_delta = fitness[delta_idx]
                
                w_alpha = 1.0 / (f_alpha + 1e-10)
                w_beta = 1.0 / (f_beta + 1e-10)
                w_delta = 1.0 / (f_delta + 1e-10)
                w_sum = w_alpha + w_beta + w_delta
                
                w_alpha /= w_sum
                w_beta /= w_sum
                w_delta /= w_sum
                
                r1_alpha, r2_alpha = rng.random(dim), rng.random(dim)
                r1_beta, r2_beta = rng.random(dim), rng.random(dim)
                r1_delta, r2_delta = rng.random(dim), rng.random(dim)
                
                A_alpha = 2 * a_coef * r1_alpha - a_coef
                C_alpha = 2 * r2_alpha
                A_beta = 2 * a_coef * r1_beta - a_coef
                C_beta = 2 * r2_beta
                A_delta = 2 * a_coef * r1_delta - a_coef
                C_delta = 2 * r2_delta
                
                D_alpha = np.abs(C_alpha * alpha_pos - X)
                X_alpha = alpha_pos - A_alpha * D_alpha
                
                D_beta = np.abs(C_beta * beta_pos - X)
                X_beta = beta_pos - A_beta * D_beta
                
                D_delta = np.abs(C_delta * delta_pos - X)
                X_delta = delta_pos - A_delta * D_delta
                
                gwo_pos = w_alpha * X_alpha + w_beta * X_beta + w_delta * X_delta
                
                # DE component
                candidates = list(range(NP))
                candidates.remove(i)
                
                r1 = rng.choice(candidates, p=probs[candidates] / probs[candidates].sum())
                candidates_r2 = [c for c in candidates if c != r1]
                r2 = rng.choice(candidates_r2, p=probs[candidates_r2] / probs[candidates_r2].sum())
                candidates_r3 = [c for c in candidates if c not in [r1, r2]]
                r3 = rng.choice(candidates_r3, p=probs[candidates_r3] / probs[candidates_r3].sum())
                
                V = wolves[r1] + F * (wolves[r2] - wolves[r3])
                V = np.clip(V, lb, ub)
                
                jrand = rng.integers(0, dim)
                U = np.copy(X)
                for j in range(dim):
                    if rng.random() < CR or j == jrand:
                        U[j] = V[j]
                
                de_pos = U
                
                # Blend GWO and DE (50-50 mix)
                new_pos = 0.5 * gwo_pos + 0.5 * de_pos
                new_pos = np.clip(new_pos, lb, ub)
                new_fit = f(new_pos)
            
            # Greedy selection
            if new_fit < old_fit:
                wolves[i] = new_pos
                fitness[i] = new_fit
                reward = 1.0
            else:
                reward = 0.0
            
            # Update Q-table
            next_state = state_from_iter(t + 1, max_iter, n_states)
            q.update(state, action, reward, next_state)
            
            # Update global best
            if new_fit < best_f:
                best_f = new_fit
                best_x = new_pos.copy()
        
        history.append(best_f)
    
    return best_x, best_f, history


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

def decode(wolf):
  controls = wolf.reshape(num_points, 3)
  return np.vstack([start, controls, end])

def fitness_of_wolf_vec(w):
    return base_fitness(decode(w), threats)

num_points=6
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
  # (600,600,80,800),
  (610,600,80,800)
]
best_x = np.array([  
  9.55869034, 10.0148494 ,153.36517271 ,  8.02696404  , 9.99830731,
  152.04071277 , 19.74487468  , 0.76145582, 174.06964243 , 17.32320497,
  0.94818376, 173.76195864, 411.66408976 , 94.37096487 ,410.28601599,
  411.19130161, 93.24581141 ,409.8808726 
])
decode_x = decode(best_x)
print(decode_x)
decode_x = np.array([
  [   0.  ,          0.   ,         0.        ],
  [  92.62992964 , 194.24876745 , 147.9041077 ],
  [  92.62348131,  194.24784731  ,147.90249298],
  [ 476.39614173,  603.87264061,  552.64035432],
  [ 476.36217391 , 603.88817857 , 552.61287575],
  [ 696.04163875,  777.79984759 , 746.99273859],
  [ 696.03288 ,    777.79569601 , 746.99604366],
  [1000.  ,       1000.      ,   1000.        ]
])          
# best_x, best_f, history = HGWODE_QLEARN_OPTIMIZER(
#   f=fitness_of_wolf_vec,
#   dim=num_points*3,
#   lb=0,
#   ub=1000,
#   NP=50,
#   max_iter=400,
#   F=0.5,
#   CR=0.9,
#   seed=0
# )
# print("Best cost: ", best_f)
# print("Best path: ", best_x)
paint.visualize_paths_3d_and_topview_with_bspline(
  start, 
  end, 
  threats, 
  {"0":decode_x}
)
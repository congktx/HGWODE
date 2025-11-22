import numpy as np
import numpy as np
import opfunu.cec_based.cec2014 as CEC2014
from tqdm import tqdm
import pandas as pd

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
    return min(n_states - 1, max(0, idx))


# ======================================
#         GENERAL VERSION
#  For CEC benchmark (CEC2014/2020/2024)
# ======================================

def HGWODE_QLEARN_OPTIMIZER(
        f,                      # fitness function: f(x)
        dim,                    # dimension
        num_wolves=30,
        max_iter=500,
        lb=-100,
        ub=100,
        F=0.5,
        CR=0.9,
        seed=None,
        q_params=None):
    
    rng = np.random.default_rng(seed)

    wolves = rng.uniform(lb, ub, (num_wolves, dim))

    fitness = np.array([f(w) for w in wolves])

    sorted_idx = np.argsort(fitness)
    alpha_idx, beta_idx, delta_idx = sorted_idx[:3]

    # ---- Q-learning agent ---
    q_params = q_params or {}
    q = QLearner(
        seed=seed,
        **{k: q_params.get(k, v) for k, v in [
            ('n_states', 3), ('n_actions', 3),
            ('alpha', 0.2), ('gamma', 0.9), ('epsilon', 0.1)
        ]}
    )

    for t in range(max_iter):
        a_coef = 2 - 2 * (t / max_iter)
        state = state_from_iter(t, max_iter, n_states=q.Q.shape[0])

        alpha_pos = wolves[alpha_idx]
        beta_pos = wolves[beta_idx]
        delta_pos = wolves[delta_idx]
        leader_pos = [alpha_pos, beta_pos, delta_pos]

        for i in range(num_wolves):
            old_fit = fitness[i]
            action = q.choose(state)

            X = wolves[i].copy()
            dim = len(X)

            # ----------------------------------------------
            #  ACTION 0 → GWO update
            #  ACTION 1 → DE update
            #  ACTION 2 → COMBINED (GWO + DE)
            # ----------------------------------------------

            # ====== GWO update ====== (Grey Wolf)
            if action in (0, 2):
                Xs = []
                for lp in leader_pos:
                    r1 = rng.random(dim)
                    r2 = rng.random(dim)
                    A = 2 * a_coef * r1 - a_coef
                    C = 2 * r2
                    D = np.abs(C * lp - X)
                    Xs.append(lp - A * D)
                gwo_new = np.mean(Xs, axis=0)
                gwo_new = np.clip(gwo_new, lb, ub)
            else:
                gwo_new = X

            # ====== DE update ======
            if action in (1, 2):
                idxs = [idx for idx in range(num_wolves) if idx != i]
                r1, r2, r3 = wolves[rng.choice(idxs, 3, replace=False)]
                V = np.clip(r1 + F * (r2 - r3), lb, ub)

                cross = rng.random(dim) < CR
                if not np.any(cross):
                    cross[rng.integers(0, dim)] = True
                U = np.where(cross, V, X)

                de_fit = f(U)
                gwo_fit = f(gwo_new)

                if de_fit < gwo_fit:
                    new_pos = U
                    new_fit = de_fit
                else:
                    new_pos = gwo_new
                    new_fit = gwo_fit
            else:
                new_pos = gwo_new
                new_fit = f(new_pos)

            # ==== greedy selection ====
            if new_fit < fitness[i]:
                wolves[i] = new_pos
                fitness[i] = new_fit

            reward = old_fit - fitness[i]
            next_state = state_from_iter(t+1, max_iter, q.Q.shape[0])
            q.update(state, action, reward, next_state)

        # --- update alpha, beta, delta ---
        sorted_idx = np.argsort(fitness)
        alpha_idx, beta_idx, delta_idx = sorted_idx[:3]

    alpha_pos = wolves[alpha_idx]
    alpha_score = fitness[alpha_idx]
    return alpha_pos, alpha_score

dim = 30
funcs = [
  CEC2014.F12014(ndim=dim), 
  CEC2014.F22014(ndim=dim), 
  CEC2014.F32014(ndim=dim), 
  CEC2014.F42014(ndim=dim), 
  CEC2014.F52014(ndim=dim), 
  CEC2014.F62014(ndim=dim), 
  CEC2014.F72014(ndim=dim), 
  CEC2014.F82014(ndim=dim), 
  CEC2014.F92014(ndim=dim), 
  CEC2014.F102014(ndim=dim), 
  CEC2014.F112014(ndim=dim), 
  CEC2014.F122014(ndim=dim), 
  CEC2014.F132014(ndim=dim), 
  CEC2014.F142014(ndim=dim), 
  CEC2014.F152014(ndim=dim), 
  CEC2014.F162014(ndim=dim), 
  CEC2014.F172014(ndim=dim), 
  CEC2014.F182014(ndim=dim), 
  CEC2014.F192014(ndim=dim), 
  CEC2014.F202014(ndim=dim), 
  CEC2014.F212014(ndim=dim), 
  CEC2014.F222014(ndim=dim), 
  CEC2014.F232014(ndim=dim), 
  CEC2014.F242014(ndim=dim), 
  CEC2014.F252014(ndim=dim), 
  CEC2014.F262014(ndim=dim), 
  CEC2014.F272014(ndim=dim), 
  CEC2014.F282014(ndim=dim), 
  CEC2014.F292014(ndim=dim), 
  CEC2014.F302014(ndim=dim), 
]
for func in funcs:
  best_x, best_f = HGWODE_QLEARN_OPTIMIZER(
      f=lambda x: func.evaluate(x),
      dim=func.ndim,
      num_wolves=50,
      max_iter=3000,
      lb=func._bounds[:, 0],
      ub=func._bounds[:, 1],
  )
  print(best_f)

# func = CEC2014.F22014(ndim=dim)
# best_x, best_f = HGWODE_QLEARN_OPTIMIZER(
#     f=lambda x: func.evaluate(x),
#     dim=func.ndim,
#     num_wolves=50,
#     max_iter=6000,
#     lb=func._bounds[:, 0],
#     ub=func._bounds[:, 1],
# )
# print(best_f)
# print(func.evaluate(best_x))
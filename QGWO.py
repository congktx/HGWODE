import numpy as np
rng = np.random.default_rng(0)

# ---------- Obstacles: cylinders (cx, cy, r, h) ----------
Obstacle = dict  # {"cx":..,"cy":..,"r":..,"h":..}

# ---------- Geometry & costs ----------
def seg_len(a, b): return np.linalg.norm(a-b)

def path_length(P):
    return float(np.sum([seg_len(P[i], P[i+1]) for i in range(len(P)-1)]))

def collision_penalty(P, obstacles, samples=7, big=1e4):
    pen = 0.0
    for i in range(len(P)-1):
        a, b = P[i], P[i+1]
        for obs in obstacles:
            cx, cy, r, h = obs["cx"], obs["cy"], obs["r"], obs["h"]
            # sample along segment
            for t in np.linspace(0,1,samples):
                p = a + t*(b-a)
                in_xy = (p[0]-cx)**2 + (p[1]-cy)**2 < r*r
                in_z  = (0.0 <= p[2] <= h)
                if in_xy and in_z:
                    pen += big
                    break  # phạt 1 lần/đoạn là đủ
    return pen

def curvature_penalty(P, theta_max=np.deg2rad(60)):
    pen = 0.0
    for i in range(1, len(P)-1):
        v1 = P[i] - P[i-1]; v2 = P[i+1] - P[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8: continue
        cosang = np.clip(np.dot(v1,v2)/(n1*n2), -1, 1)
        ang = np.arccos(cosang)
        if ang > theta_max:
            pen += (ang - theta_max)**2
    return pen

def cost(P, obstacles, w=(1.0, 2000.0, 1.0)):
    w_d, w_c, w_k = w
    return (w_d*path_length(P)
            + w_c*collision_penalty(P, obstacles)
            + w_k*curvature_penalty(P))

# ---------- Local path editors (actions) ----------
def perturb_local(P, i, sigma=2.0):
    if i<=0 or i>=len(P)-1: return P
    Q = P.copy(); Q[i] = P[i] + rng.normal(0, sigma, size=3)
    return Q

def laplacian_smooth(P, i, lam=0.6):
    if i<=0 or i>=len(P)-1: return P
    Q = P.copy()
    Q[i] = P[i] + lam*(0.5*(P[i-1] + P[i+1]) - P[i])
    return Q

def insert_waypoint(P, i):
    if i>=len(P)-1: return P
    mid = 0.5*(P[i] + P[i+1])
    return np.insert(P, i+1, mid, axis=0)

def delete_waypoint(P, i):
    if len(P)<=3 or i<=0 or i>=len(P)-1: return P
    return np.delete(P, i, axis=0)

def escape_from_cylinders(P, obstacles, strength=2.0):
    Q = P.copy()
    for i in range(1, len(P)-1):
        p = P[i]
        # tìm trụ gần nhất theo khoảng cách XY
        best = None; best_d = 1e18
        for obs in obstacles:
            dxy = np.hypot(p[0]-obs["cx"], p[1]-obs["cy"])
            if dxy < best_d: best_d, best = dxy, obs
        if best is None: continue
        # vector đẩy theo hướng từ tâm trụ ra ngoài (chỉ XY)
        vx, vy = p[0]-best["cx"], p[1]-best["cy"]
        n = np.hypot(vx, vy) + 1e-9
        push = strength * np.array([vx/n, vy/n, 0.0])
        Q[i] = p + push
    return Q

ACTIONS = ["PERTURB", "SMOOTH", "INSERT", "DELETE", "ESCAPE"]

def apply_action(P, action):
    K = len(P)
    if action == "PERTURB":
        return perturb_local(P, rng.integers(1, K-1), sigma=2.5)
    if action == "SMOOTH":
        return laplacian_smooth(P, rng.integers(1, K-1), lam=0.6)
    if action == "INSERT":
        return insert_waypoint(P, rng.integers(0, K-1))
    if action == "DELETE":
        return delete_waypoint(P, rng.integers(1, K-1))
    if action == "ESCAPE":
        return escape_from_cylinders(P, OBS)
    return P

# ---------- GWO update (waypoint-wise) ----------
def gwo_update(P, A_lead, B_lead, D_lead, a_coeff):
    Q = P.copy()
    for i in range(1, len(P)-1):
        X = P[i]
        def pull(L):
            C = 2.0*rng.random(3)
            A = 2.0*a_coeff*rng.random(3) - a_coeff
            Dv = np.abs(C*L[i] - X)
            return L[i] - A*Dv
        X1 = pull(A_lead); X2 = pull(B_lead); X3 = pull(D_lead)
        Q[i] = (X1 + X2 + X3) / 3.0
    return Q

# ---------- Simple Q-table ----------
class QTable:
    def __init__(self, shape, nA):
        self.Q = np.zeros(shape + (nA,))
    def eps_greedy(self, s, eps):
        if rng.random() < eps: return rng.integers(0, self.Q.shape[-1])
        return int(np.argmax(self.Q[s]))
    def update(self, s, a, r, s2, lr, gamma):
        self.Q[s][a] += lr*(r + gamma*np.max(self.Q[s2]) - self.Q[s][a])

def approx_collision_count(P, obstacles):
    return int(collision_penalty(P, obstacles, samples=5) / 1e4)

def sharp_ratio(P, theta_max=np.deg2rad(60)):
    cnt = 0
    for i in range(1,len(P)-1):
        v1=P[i]-P[i-1]; v2=P[i+1]-P[i]
        n1,n2=np.linalg.norm(v1),np.linalg.norm(v2)
        if n1<1e-8 or n2<1e-8: continue
        ang=np.arccos(np.clip(np.dot(v1,v2)/(n1*n2),-1,1))
        if ang>theta_max: cnt+=1
    base=max(1,len(P)-2)
    return cnt/base

def bins(x, edges): return int(np.digitize([x], edges)[0])

def featurize(P, A, B, D, obstacles):
    # diversity vs leaders
    dA = np.mean(np.linalg.norm(P-A, axis=1))
    dB = np.mean(np.linalg.norm(P-B, axis=1))
    dD = np.mean(np.linalg.norm(P-D, axis=1))
    div = (dA+dB+dD)/3.0
    col = approx_collision_count(P, obstacles)
    sharp = sharp_ratio(P)
    return (
        bins(div,   [2,5,10,20]),
        bins(col,   [0.5,1.5,3.5]),
        bins(sharp, [0.1,0.3,0.6]),
    )

# ---------- Demo world ----------
OBS = [
    {"cx":20.0,"cy":20.0,"r":8.0,"h":30.0},
    {"cx":45.0,"cy":35.0,"r":6.0,"h":25.0},
]
S = np.array([0.0, 0.0, 10.0])
G = np.array([60.0, 40.0, 10.0])

def init_path(K=10):
    xs = np.linspace(S[0], G[0], K+1)
    ys = np.linspace(S[1], G[1], K+1)
    zs = np.linspace(S[2], G[2], K+1)
    P = np.stack([xs,ys,zs], axis=1)
    P[1:-1] += rng.normal(0, 1.0, size=(K-1,3))
    return P

# ---------- Hybrid loop ----------
N = 12; T = 120
wolves = [init_path(10) for _ in range(N)]
w_cost = (1.0, 4000.0, 1.0)   # ưu tiên an toàn: w_c lớn

Qtab = QTable(shape=(5,4,5), nA=len(ACTIONS))
lr, gamma = 0.2, 0.95
eps0, epsT = 0.3, 0.05

bestJ = np.inf; bestP = None

for t in range(T):
    Js = np.array([cost(P, OBS, w_cost) for P in wolves])
    order = np.argsort(Js)
    A_lead, B_lead, D_lead = [wolves[i] for i in order[:3]]

    a_coeff = 2.0*(1 - t/(T-1))           # GWO 'a' tuyến tính
    eps = epsT + (eps0-epsT)*max(0,(T-1-t)/(T-1))

    new_wolves = []
    for P in wolves:
        # Global step (GWO)
        P_g = gwo_update(P, A_lead, B_lead, D_lead, a_coeff)

        # RL local refinement
        s = featurize(P_g, A_lead, B_lead, D_lead, OBS)
        a_idx = Qtab.eps_greedy(s, eps)
        P_new = apply_action(P_g, ACTIONS[a_idx])

        J_prev = cost(P_g, OBS, w_cost)
        J_new  = cost(P_new, OBS, w_cost)

        # reward: giảm J tốt; va chạm bị phạt thêm
        col_new = approx_collision_count(P_new, OBS)
        r = (J_prev - J_new) - 2000.0*(col_new>0)

        s2 = featurize(P_new, A_lead, B_lead, D_lead, OBS)
        Qtab.update(s, a_idx, r, s2, lr, gamma)

        # accept-if-better
        new_wolves.append(P_new if J_new < J_prev else P_g)

    wolves = new_wolves

    # track best
    Js = np.array([cost(P, OBS, w_cost) for P in wolves])
    b = int(np.argmin(Js))
    if Js[b] < bestJ:
        bestJ, bestP = Js[b], wolves[b].copy()

print("Best cost:", bestJ)
print("Best path waypoints:", bestP.shape[0])

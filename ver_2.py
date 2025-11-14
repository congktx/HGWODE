import numpy as np
import visualize

# ---------- 1) Segment–Cylinder collision (phát hiện chính xác hơn) ----------
def segment_cylinder_collision(p1, p2, cx, cy, r, h):
    # Làm việc trên mặt phẳng XY trước
    a = p1[:2]
    b = p2[:2]
    d = b - a
    if np.allclose(d, 0):  # đoạn quá ngắn trong XY
        dist_xy = np.linalg.norm(a - np.array([cx, cy]))
        z_min, z_max = sorted([p1[2], p2[2]])
        return (dist_xy < r) and (z_min <= h and 0 <= z_max)

    # Điểm gần tâm trụ nhất trên đoạn (clamp t ∈ [0,1])
    c = np.array([cx, cy])
    t_star = np.clip(np.dot(c - a, d) / (np.dot(d, d) + 1e-12), 0.0, 1.0)
    q_xy = a + t_star * d
    dist_xy = np.linalg.norm(q_xy - c)

    # Z tại t*
    z_t = p1[2] + t_star * (p2[2] - p1[2])

    # Kiểm tra giao cắt: gần trụ và trong khoảng chiều cao
    hit_core = (dist_xy <= r + 1e-9) and (0 <= z_t <= h)

    if hit_core:
        return True

    # Bổ sung: kiểm tra vùng cắt theo sampling dày để an toàn (phòng biên)
    for t in np.linspace(0, 1, max(25, int(np.linalg.norm(d) / (r/3 + 1e-6)))):
        p = p1 + t * (p2 - p1)
        dist_xy = np.hypot(p[0]-cx, p[1]-cy)
        if dist_xy <= r and 0 <= p[2] <= h:
            return True

    return False

# ---------- 2) Tính tổng vi phạm ràng buộc (CV) thay vì chỉ +penalty ----------
def compute_constraints(path, threats, theta_max=np.pi/3):
    # C1: va chạm
    collide = False
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i+1]
        for (cx, cy, r, h) in threats:
            if segment_cylinder_collision(p1, p2, cx, cy, r, h):
                collide = True
                break
        if collide:
            break
    C1 = 0 if not collide else 1  # cờ vi phạm (dùng 1 thay cho độ lớn vì mình đã kiểm chính xác)

    # C2: góc rẽ
    C2_sum = 0.0
    for i in range(1, len(path)-1):
        v1 = path[i] - path[i-1]
        v2 = path[i+1] - path[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9: 
            continue
        cos_ang = np.dot(v1, v2) / (n1 * n2)
        angle = np.arccos(np.clip(cos_ang, -1, 1))
        if angle > theta_max:
            C2_sum += (angle - theta_max)
    C2 = C2_sum  # độ lớn vi phạm góc

    CV = C1 + C2
    feasible = (CV == 0)
    return feasible, CV

# ---------- 3) Fitness: chỉ dùng chiều dài khi feasible; nếu infeasible đẩy lớn ----------
def path_length(points):
    return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))

def fitness_uav(path, threats, theta_max=np.pi/3):
    feasible, CV = compute_constraints(path, threats, theta_max)
    L = path_length(path)
    if feasible:
        return L, feasible, CV
    # death penalty/penalty lớn theo CV để 100% loại nghiệm vi phạm
    return L + 1e6 * (1 + CV), feasible, CV

# ---------- 4) So sánh theo constraint-domination ----------
def better(sol_new, sol_old):
    # sol = (cost, feasible, CV, pos_array)
    c_new, f_new, cv_new, _ = sol_new
    c_old, f_old, cv_old, _ = sol_old
    if f_new and not f_old:
        return True
    if f_old and not f_new:
        return False
    if f_new and f_old:
        return c_new < c_old
    # cả hai đều infeasible
    if cv_new != cv_old:
        return cv_new < cv_old
    return c_new < c_old


# ----------------------------- HGWODE for UAV 3D -----------------------------
def HGWODE_UAV_3D(start, end, threats, num_wolves=25, num_points=5, 
                  max_iter=6000, lb=0, ub=1000, F=0.5, CR=0.9,
                  checkpoints=[1000,2000,3000,4000,5000,6000],
                  theta_max=np.pi/3, anneal_DE=True):

    dim = num_points*3
    wolves = np.random.uniform(lb, ub, (num_wolves, dim))

    def decode(vec):
        ctrls = vec.reshape(num_points, 3)
        return np.vstack([start, ctrls, end])

    # evaluate pack
    def eval_vec(vec):
        path = decode(vec)
        cost, feasible, CV = fitness_uav(path, threats, theta_max)
        return cost, feasible, CV, path

    sols = [eval_vec(w) for w in wolves]
    # locate alpha/beta/delta
    order = np.argsort([s[0] + (0 if s[1] else 1e12) for s in sols])
    alpha, beta, delta = order[:3]
    alpha_sol = sols[alpha]

    # best feasible archive
    best_feasible = alpha_sol if alpha_sol[1] else None

    best_paths = {}
    best_costs = {}
    for t in range(1, max_iter+1):
        a = 2 - 2*(t/max_iter)

        # (optional) anneal DE params
        if anneal_DE:
            F_t  = F * (0.3 + 0.7*(1 - t/max_iter))  # giảm dần còn ~0.3F về cuối
            CR_t = CR * (0.5 + 0.5*(1 - t/max_iter)) # giảm dần CR
        else:
            F_t, CR_t = F, CR

        # ---------- GWO update ----------
        for i in range(num_wolves):
            X = wolves[i].copy()
            leaders = [wolves[alpha], wolves[beta], wolves[delta]]
            Xs = []
            for leader in leaders:
                r1, r2 = np.random.rand(dim), np.random.rand(dim)
                A = 2*a*r1 - a
                C = 2*r2
                D = np.abs(C*leader - X)
                Xs.append(leader - A*D)
            new_pos = np.clip(np.mean(Xs, axis=0), lb, ub)

            new_sol = eval_vec(new_pos)
            if better(new_sol, sols[i]):
                wolves[i] = new_pos
                sols[i]   = new_sol

        # ---------- DE update ----------
        for i in range(num_wolves):
            idxs = list(range(num_wolves)); idxs.remove(i)
            r1, r2, r3 = wolves[np.random.choice(idxs, 3, replace=False)]
            V = np.clip(r1 + F_t*(r2 - r3), lb, ub)
            mask = (np.random.rand(dim) < CR_t)
            if not mask.any():
                mask[np.random.randint(0, dim)] = True
            U = np.where(mask, V, wolves[i])
            U_sol = eval_vec(U)
            if better(U_sol, sols[i]):
                wolves[i] = U
                sols[i]   = U_sol

        # cập nhật alpha/beta/delta
        order = sorted(range(num_wolves), key=lambda k: (not sols[k][1], sols[k][0], sols[k][2]))
        alpha, beta, delta = order[:3]
        alpha_sol = sols[alpha]

        # lưu best feasible
        if alpha_sol[1]:
            if (best_feasible is None) or (alpha_sol[0] < best_feasible[0]):
                best_feasible = alpha_sol

        if t in checkpoints:
            # ưu tiên lưu đường feasible; nếu chưa có thì lưu alpha hiện tại (để quan sát)
            path_to_save = (best_feasible or alpha_sol)[3]
            best_paths[t] = path_to_save
            best_costs[t] = path_length(path_to_save)

    # Trả về nghiệm feasible tốt nhất nếu có, ngược lại alpha cuối cùng
    final_path = (best_feasible or alpha_sol)[3]
    return final_path, best_paths, best_costs

# ----------------------------- Demo run -----------------------------
start = np.array([0,0,0])
end   = np.array([1000,1000,1000])
# threats = [
#   (150,220,50,500),
#   (300,300,100,800),
#   (600,600,150,600),
#   (500,200,120,1000)
# ]
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
max_iter=500
checkpoints=[100,200,300,400,500]

final_path, best_paths, best_costs = HGWODE_UAV_3D(
  start,end,
  threats,
  num_wolves=25,
  num_points=5,
  max_iter=max_iter,
  checkpoints=checkpoints
)

print(best_costs)

visualize.visualize_paths_3d_and_topview_with_bspline(start,end,threats,best_paths,samples=400)


import numpy as np

# -----------------------------
# CEC2014 Benchmark Functions
# -----------------------------
def f1(x):  # Sphere
    return np.sum(x**2)

def f2(x):  # Schwefel 2.22
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def f3(x):  # Schwefel 1.2
    return np.sum([np.sum(x[:i+1])**2 for i in range(len(x))])

def f4(x):  # Schwefel 2.21
    return np.max(np.abs(x))

def f5(x):  # Rosenbrock
    return np.sum(100*(x[1:] - x[:-1]**2)**2 + (x[:-1]-1)**2)

def f6(x):  # Step
    return np.sum((np.floor(x+0.5))**2)

def f7(x):  # Quartic + noise
    return np.sum([(i+1)*x[i]**4 for i in range(len(x))]) + np.random.rand()

def f8(x):  # Schwefel
    return 418.9829*len(x) - np.sum(x*np.sin(np.sqrt(np.abs(x))))

def f9(x):  # Rastrigin
    return np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10)

def f10(x):  # Ackley
    n = len(x)
    return -20*np.exp(-0.2*np.sqrt(np.sum(x**2)/n)) - np.exp(np.sum(np.cos(2*np.pi*x))/n) + 20 + np.e

def f11(x):  # Griewank
    return np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1,len(x)+1)))) + 1

def f12(x):  # Penalized
    y = 1 + (x+1)/4
    term1 = np.pi/len(x) * (10*np.sin(np.pi*y[0])**2 + np.sum((y[:-1]-1)**2*(1+10*np.sin(np.pi*y[1:])**2)) + (y[-1]-1)**2)
    term2 = np.sum(u(x,10,100,4))
    return term1 + term2

def f13(x):
    y = 1 + (x+1)/4
    term1 = 0.1*(np.sin(3*np.pi*x[0])**2 + np.sum((x[:-1]-1)**2*(1+np.sin(3*np.pi*x[1:])**2)) + (x[-1]-1)**2*(1+np.sin(2*np.pi*x[-1])**2))
    term2 = np.sum(u(x,5,100,4))
    return term1 + term2

# Helper for penalty
def u(x,a,k,m):
    return np.array([k*((xi-a)**m) if xi>a else k*((-xi-a)**m) if xi<-a else 0 for xi in x])

# f14–f16 Shifted Rotated versions (simplified as standard)
def f14(x): return f9(x)
def f15(x): return f10(x)
def f16(x): return f11(x)

# f17–f22 Hybrid (approx using combos)
def f17(x): return f1(x) + f9(x)
def f18(x): return f2(x) + f10(x)
def f19(x): return f3(x) + f11(x)
def f20(x): return f4(x) + f5(x)
def f21(x): return f6(x) + f8(x)
def f22(x): return f7(x) + f9(x)

# f23–f30 Composition (approx mix)
def f23(x): return f1(x)+f10(x)+f11(x)
def f24(x): return f2(x)+f8(x)+f9(x)
def f25(x): return f3(x)+f10(x)+f12(x)
def f26(x): return f4(x)+f9(x)+f11(x)
def f27(x): return f5(x)+f10(x)+f13(x)
def f28(x): return f6(x)+f8(x)+f11(x)
def f29(x): return f7(x)+f9(x)+f12(x)
def f30(x): return f8(x)+f10(x)+f13(x)

cec_functions = {
    f"f{i}": globals()[f"f{i}"] for i in range(1,31)
}

# -----------------------------
# HGWODE General Implementation
# -----------------------------
def HGWODE(func, dim=30, num_wolves=30, max_iter=500, lb=-100, ub=100, F=0.5, CR=0.9):
    wolves = np.random.uniform(lb, ub, (num_wolves, dim))
    fitness = np.array([func(w) for w in wolves])

    alpha, beta, delta = np.argsort(fitness)[:3]
    alpha_pos, alpha_score = wolves[alpha].copy(), fitness[alpha]

    for t in range(max_iter):
        a = 2 - 2 * (t / max_iter)

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
            fitness[i] = func(new_pos)

        # --- DE update ---
        for i in range(num_wolves):
            idxs = list(range(num_wolves)); idxs.remove(i)
            r1, r2, r3 = wolves[np.random.choice(idxs, 3, replace=False)]
            V = np.clip(r1 + F*(r2-r3), lb, ub)
            cross_points = np.random.rand(dim) < CR
            if not np.any(cross_points): cross_points[np.random.randint(0, dim)] = True
            U = np.where(cross_points, V, wolves[i])
            U_fit = func(U)
            if U_fit < fitness[i]:
                wolves[i], fitness[i] = U, U_fit

        # Update leaders
        best_indices = np.argsort(fitness)[:3]
        alpha = best_indices[0]
        alpha_pos, alpha_score = wolves[alpha].copy(), fitness[alpha]

    return alpha_pos, alpha_score
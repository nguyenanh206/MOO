"""

Implementation follows the paper exactly:
  - Algorithm 1  : ILCMO main loop
  - Algorithm 2  : Reproduction_VGDE
  - Algorithm 3  : Update_Is  (feasibility-oriented selection)
  - Algorithm 4  : Update_Id  (infeasibility-assisted selection)
  - Section III  : Indicators Is and Id (Eq. 4–16)
  - Section IV-B : VGDE operators (Eq. 21–23)
"""

import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
# ══════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

class Individual:
    """One candidate solution."""

    def __init__(self, x: np.ndarray):
        self.x = x.copy()                 # decision vector  (D,)
        self.F: Optional[np.ndarray] = None   # objective values (M,)
        self.phi: float = np.inf               # total constraint violation
        self.phi_k: Optional[np.ndarray] = None  # per-constraint violations

    def __repr__(self):
        return (f"Individual(phi={self.phi:.4g}, "
                f"F={None if self.F is None else np.round(self.F,3)})")


# ══════════════════════════════════════════════════════════════════
#  SECTION III — INDICATOR MATH (Eq. 4–16)
# ══════════════════════════════════════════════════════════════════

DELTA = 1e-6   # small positive constant used throughout the paper


def shift_objectives(pop: List[Individual]) -> np.ndarray:
    """
    Chuyển nghiệm sang góc phần tư thứ nhất
    Eq. (4):  f̃_k(x) = f_k(x) - z*_k + δ
    Shift all objectives to the first quadrant.
    Returns F_shifted  shape (N, M).
    """
    F = np.array([ind.F for ind in pop])          # (N, M)
    z_star = np.min(F, axis=0)                    # (M,)
    return F - z_star + DELTA                     # (N, M)


def cost_value(fi: np.ndarray, fj: np.ndarray) -> float:
    """
    giá trị chi phí dựa trên tỷ lệ
    Eq. (6): CV(xⁱ | xʲ) — ratio-based cost value.

    CV(xⁱ|xʲ) = max_k(fj_k / fi_k)  if ∃ fj_k > fi_k
               = min_k(fj_k / fi_k)  otherwise
    """
    ratios = fj / fi          # element-wise  (M,)
    if np.any(fj > fi):
        return float(np.max(ratios))
    return float(np.min(ratios))


def d_cv(fi: np.ndarray, fj: np.ndarray) -> float:
    """
    Hàm khoảng cách dựa trên giá trị chi phí
    Eq. (5): d^CV(xⁱ, xʲ) = log max(CV(xⁱ|xʲ), CV(xʲ|xⁱ))
    """
    cv_ij = cost_value(fi, fj)
    cv_ji = cost_value(fj, fi)
    return float(np.log(max(cv_ij, cv_ji)))


def delta_P(pop: List[Individual], F_shifted: np.ndarray) -> float:
    """
    Eq. (7): ΔP = log((φ_min + δ)/(φ_max + δ)) + log CV(x^w | x^b) − δ

    x^w = worst-objective solution (max on every k),
    x^b = best-objective solution  (min on every k).
    """
    phis = np.array([ind.phi for ind in pop])
    phi_min = float(np.min(phis))
    phi_max = float(np.max(phis))

    # x^b: solution with min value on all objectives → use sum as proxy
    # (paper: f_k(x^b) = min_{x∈P} f_k(x), same for x^w — we pick by sum)
    idx_b = int(np.argmin(np.sum(F_shifted, axis=1)))
    idx_w = int(np.argmax(np.sum(F_shifted, axis=1)))
    fb = F_shifted[idx_b]
    fw = F_shifted[idx_w]

    log_phi = np.log((phi_min + DELTA) / (phi_max + DELTA))
    log_cv  = np.log(cost_value(fw, fb))
    return float(log_phi + log_cv - DELTA)


# ──────────────────────────────────────────────────────────────────
#  Feasibility-Oriented Indicator  Iˢ  (Eq. 8–9)
# ──────────────────────────────────────────────────────────────────

def I_s_pair(phi_i: float, phi_j: float,
             fi: np.ndarray, fj: np.ndarray,
             dP: float) -> float:
    """
    Eq. (8): Iˢ(xⁱ | xʲ)

    Case 1:  φⁱ = 0, φʲ = 0  →  log CV(xⁱ|xʲ)
    Case 2:  φⁱ = 0, φʲ ≠ 0  →  +∞
    Case 3:  φⁱ ≠ 0           →  log((φʲ+δ)/(φⁱ+δ)) + ΔP
    """
    if phi_i == 0.0 and phi_j == 0.0:
        return float(np.log(cost_value(fi, fj)))
    elif phi_i == 0.0:          # φⁱ=0, φʲ≠0
        return float('inf')
    else:                       # φⁱ ≠ 0  (Case 3)
        return float(np.log((phi_j + DELTA) / (phi_i + DELTA)) + dP)


def compute_Is(pop: List[Individual]) -> np.ndarray:
    """
    Eq. (9): Iˢ(xⁱ | P) = min_{xʲ ∈ P \ xⁱ} Iˢ(xⁱ | xʲ)

    Returns fitness array of shape (N,).
    """
    N = len(pop)
    F_sh = shift_objectives(pop)
    dP   = delta_P(pop, F_sh)
    phis = np.array([ind.phi for ind in pop])

    fitness = np.full(N, -np.inf)
    for i in range(N):
        vals = []
        for j in range(N):
            if i == j:
                continue
            v = I_s_pair(phis[i], phis[j], F_sh[i], F_sh[j], dP)
            vals.append(v)
        fitness[i] = min(vals) if vals else float('inf')
    return fitness


# ──────────────────────────────────────────────────────────────────
#  Infeasibility-Assisted Dynamic Indicator  Iᵈ  (Eq. 15–16)
# ──────────────────────────────────────────────────────────────────

def I_d_pair(phi_i: float, phi_j: float,
             fi: np.ndarray, fj: np.ndarray,
             alpha: float, beta: float,
             dP: float) -> float:
    """
    Eq. (15): Iᵈ(xⁱ | xʲ)

    Case 1:  φⁱ ≤ α, φʲ ≤ β            →  log CV(xⁱ|xʲ)
    Case 2:  φⁱ = 0, φʲ > β            →  +∞
    Case 3:  0 < φⁱ ≤ α, φʲ > β        →  max(d^CV, log(φʲ/φⁱ))
    Case 4:  φⁱ > α                     →  log((φʲ+δ)/(φⁱ+δ)) + ΔP
    """
    if phi_i <= alpha and phi_j <= beta:
        return float(np.log(cost_value(fi, fj)))
    elif phi_i == 0.0 and phi_j > beta:
        return float('inf')
    elif 0.0 < phi_i <= alpha and phi_j > beta:
        dcv = d_cv(fi, fj)
        log_ratio = float(np.log(phi_j / phi_i)) if phi_i > 0 else float('inf')
        return max(dcv, log_ratio)
    else:   # phi_i > alpha  (Case 4)
        return float(np.log((phi_j + DELTA) / (phi_i + DELTA)) + dP)


def compute_Id(pop: List[Individual],
               alpha: float, beta: float) -> np.ndarray:
    """
    Eq. (16): Iᵈ(xⁱ | P) = min_{xʲ ∈ P \ xⁱ} Iᵈ(xⁱ | xʲ)

    Returns fitness array of shape (N,).
    """
    N = len(pop)
    F_sh = shift_objectives(pop)
    dP   = delta_P(pop, F_sh)
    phis = np.array([ind.phi for ind in pop])

    fitness = np.full(N, -np.inf)
    for i in range(N):
        vals = []
        for j in range(N):
            if i == j:
                continue
            v = I_d_pair(phis[i], phis[j], F_sh[i], F_sh[j], alpha, beta, dP)
            vals.append(v)
        fitness[i] = min(vals) if vals else float('inf')
    return fitness


# ══════════════════════════════════════════════════════════════════
#  DYNAMIC BOUNDARIES  αₜ  and  βₜ  (Eq. 13–14)
# ══════════════════════════════════════════════════════════════════

def compute_alpha(t: int, T_max: int, alpha0: float, pp: float = 0.5) -> float:
    """
    Eq. (13): αₜ = α₀ · (1 − t/T_max)^[(−log(α₀)−6) / log(1−pp)]

    Starts large, decreases to 0.
    """
    if alpha0 <= 0:
        return 0.0
    exponent = (-np.log(alpha0) - 6.0) / np.log(1.0 - pp)
    base = max(0.0, 1.0 - t / T_max)
    return float(alpha0 * (base ** exponent))


def compute_beta(alpha_t: float, pf: float) -> float:
    """
    Eq. (14): βₜ = αₜ · (1 − pf)

    pf = proportion of feasible solutions in population P₁.
    """
    return float(alpha_t * (1.0 - pf))


# ══════════════════════════════════════════════════════════════════
#  SECTION IV-B — VARIABLE GROUPING  (ordered grouping, [12])
# ══════════════════════════════════════════════════════════════════

def ordered_grouping(D: int, K: int) -> List[List[int]]:
    """
    Ordered grouping mechanism (reference [12] in paper, Section IV-B).
    Divides D variables into K equal, exclusive groups by index order.
    e.g. D=10, K=4 → [0,1,2], [3,4,5], [6,7], [8,9]
    """
    groups = []
    size = D // K
    remainder = D % K
    start = 0
    for k in range(K):
        extra = 1 if k < remainder else 0
        end = start + size + extra
        groups.append(list(range(start, end)))
        start = end
    return groups


def select_variable_group(t: int, T_max: int,
                           groups: List[List[int]], D: int,
                           rng: np.random.Generator) -> List[int]:
    """
    Eq. (21): ρ = (1 − t/T_max)²
    If rand < ρ  →  pick a random group (low-dimensional subspace)
    Else         →  use all D variables (full space)
    """
    rho = (1.0 - t / T_max) ** 2
    if rng.random() < rho:
        k = rng.integers(0, len(groups))
        return groups[k]
    else:
        return list(range(D))


# ══════════════════════════════════════════════════════════════════
#  SECTION IV-B — DE OPERATORS (Eq. 22–23) + Polynomial Mutation
# ══════════════════════════════════════════════════════════════════

def intra_learning(x: np.ndarray,
                   xr1: np.ndarray, xr2: np.ndarray,
                   v: List[int],
                   lb: np.ndarray, ub: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
    """
    Eq. (22): Group-Based IntraLearning DE operator  (for Se).

    x_new_d = x_d + L · (xr1_d − xr2_d)  if d ∈ v
            = x_d                           otherwise

    L ∈ [0,1] random learning rate.
    xr1, xr2 are two distinct random solutions from Se.
    """
    x_new = x.copy()
    L = rng.random()
    for d in v:
        x_new[d] = x[d] + L * (xr1[d] - xr2[d])
    # clip to bounds
    x_new = np.clip(x_new, lb, ub)
    return x_new


def inter_learning(x: np.ndarray,
                   y: np.ndarray,
                   v: List[int],
                   lb: np.ndarray, ub: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
    """
    Eq. (23): Group-Based InterLearning DE operator  (for Sp).

    x_new_d = x_d + L · (y_d − x_d)   if d ∈ v
            = x_d                       otherwise

    y is a randomly selected individual from Se.
    L ∈ [0,1] random learning rate.
    """
    x_new = x.copy()
    L = rng.random()
    for d in v:
        x_new[d] = x[d] + L * (y[d] - x[d])
    x_new = np.clip(x_new, lb, ub)
    return x_new


def polynomial_mutation(x: np.ndarray,
                         lb: np.ndarray, ub: np.ndarray,
                         pm: float, eta_m: float,
                         rng: np.random.Generator) -> np.ndarray:
    """
    Polynomial Mutation [6] — applied to every offspring (Algorithm 2, line 21).

    pm    = mutation probability (paper: 1/D)
    eta_m = distribution index   (paper: 20)
    """
    x_new = x.copy()
    D = len(x)
    for d in range(D):
        if rng.random() < pm:
            u = rng.random()
            delta = ub[d] - lb[d]
            if delta == 0:
                continue
            if u <= 0.5:
                delta_q = (2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
            else:
                delta_q = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))
            x_new[d] = np.clip(x[d] + delta_q * delta, lb[d], ub[d])
    return x_new


# ══════════════════════════════════════════════════════════════════
#  ALGORITHM 2 — Reproduction_VGDE
# ══════════════════════════════════════════════════════════════════

def reproduction_VGDE(pop: List[Individual],
                       fitness: np.ndarray,
                       t: int, T_max: int,
                       groups: List[List[int]],
                       lb: np.ndarray, ub: np.ndarray,
                       pm: float, eta_m: float,
                       rng: np.random.Generator) -> List[Individual]:
    """
    Algorithm 2: Reproduction_VGDE

    Steps:
      1. Cluster P into Se (top-half by fitness) and Sp (bottom-half).
      2. Produce K variable groups via ordered grouping G.
      3. For each individual:
           a. Determine search space via ρ (Eq. 21).
           b. If xi ∈ Se → IntraLearning  (Eq. 22).
              If xi ∈ Sp → InterLearning  (Eq. 23).
           c. Apply Polynomial Mutation.
      4. Return offspring population O.
    """
    N = len(pop)
    D = len(lb)

    # Step 1: cluster by fitness (descending → high fitness = better)
    sorted_idx = np.argsort(fitness)[::-1]
    half = N // 2
    se_idx = set(sorted_idx[:half])   # elite
    sp_idx = set(sorted_idx[half:])   # poor
    Se = [pop[i] for i in sorted_idx[:half]]

    offspring = []

    for i in range(N):
        xi = pop[i]

        # Step 3a: select variable group
        v = select_variable_group(t, T_max, groups, D, rng)

        if i in se_idx:
            # Step 3b-i: IntraLearning (from Se)
            candidates = [s for s in Se if s is not xi]
            if len(candidates) < 2:
                candidates = Se   # fallback
            idx_r = rng.choice(len(candidates), size=2, replace=False)
            xr1 = candidates[idx_r[0]].x
            xr2 = candidates[idx_r[1]].x
            x_new = intra_learning(xi.x, xr1, xr2, v, lb, ub, rng)
        else:
            # Step 3b-ii: InterLearning (learn from a random y ∈ Se)
            y = Se[rng.integers(0, len(Se))].x
            x_new = inter_learning(xi.x, y, v, lb, ub, rng)

        # Step 3c: Polynomial Mutation
        x_new = polynomial_mutation(x_new, lb, ub, pm, eta_m, rng)

        child = Individual(x_new)
        offspring.append(child)

    return offspring


# ══════════════════════════════════════════════════════════════════
#  ALGORITHM 3 — Update_Is  (Environmental Selection by Iˢ)
# ══════════════════════════════════════════════════════════════════

def spea2_truncation(pop: List[Individual],
                      fitness: np.ndarray,
                      N: int,
                      F_sh: np.ndarray) -> List[Individual]:
    """
    SPEA2-style truncation [53]:
    Remove one individual at a time — the one with the minimum
    distance to its nearest neighbor (among remaining).
    """
    remaining = list(range(len(pop)))
    F = F_sh.copy()

    while len(remaining) > N:
        # pairwise Euclidean distances in objective space
        M = len(remaining)
        dist = np.full((M, M), np.inf)
        for a in range(M):
            for b in range(M):
                if a != b:
                    dist[a, b] = np.linalg.norm(F[remaining[a]] - F[remaining[b]])

        # sort each row, find nearest neighbor distance
        nn_dist = np.sort(dist, axis=1)[:, 1]   # 2nd col = nearest neighbor

        # remove the one with the smallest nn distance (most crowded)
        remove_local = int(np.argmin(nn_dist))
        remaining.pop(remove_local)

    return [pop[i] for i in remaining]


def update_Is(MP: List[Individual], N: int,
              problem) -> List[Individual]:
    """
    Algorithm 3: Update_Is

    1. Compute Iˢ fitness for all solutions in MP.
    2. Identify feasible non-dominated solutions using Theorem 1:
           {x | Iˢ(x|MP) ≥ 0}
    3a. If |P₁| ≤ N → select N best from MP.
    3b. If |P₁| > N → prune to N using SPEA2 truncation.
    """
    # evaluate if not yet done
    for ind in MP:
        if ind.F is None:
            problem.evaluate(ind)

    fitness = compute_Is(MP)
    F_sh = shift_objectives(MP)

    # Theorem 1: Iˢ(xⁱ|P) ≥ 0  ⟺  xⁱ is feasible non-dominated
    P1 = [MP[i] for i in range(len(MP)) if fitness[i] >= 0.0]

    if len(P1) <= N:
        # Select N best from entire MP (by fitness, descending)
        sorted_idx = np.argsort(fitness)[::-1]
        return [MP[i] for i in sorted_idx[:N]]
    else:
        # Prune to N using SPEA2 truncation
        p1_idx = [i for i in range(len(MP)) if fitness[i] >= 0.0]
        p1_fitness = fitness[np.array(p1_idx)]
        p1_Fsh = F_sh[np.array(p1_idx)]
        p1_pop = [MP[i] for i in p1_idx]
        return spea2_truncation(p1_pop, p1_fitness, N, p1_Fsh)


# ══════════════════════════════════════════════════════════════════
#  ALGORITHM 4 — Update_Id  (Environmental Selection by Iᵈ)
# ══════════════════════════════════════════════════════════════════

def assign_subregion(ind: Individual,
                     weight_vectors: np.ndarray,
                     z_star: np.ndarray) -> int:
    """
    Eq. (24):  Ωᵢ = {F(x) | ∀j: ∠(F(x), λⁱ) ≤ ∠(F(x), λʲ)}
    Assign individual to the weight-vector subregion with smallest angle.
    """
    F = ind.F - z_star + DELTA     # shift to first quadrant
    # angle between F and each weight vector
    angles = []
    for lam in weight_vectors:
        cos_angle = (np.dot(F, lam) /
                     (np.linalg.norm(F) * np.linalg.norm(lam) + DELTA))
        angles.append(cos_angle)
    # largest cosine = smallest angle
    return int(np.argmax(angles))


def update_Id(MP: List[Individual], N: int,
              weight_vectors: np.ndarray,
              alpha: float, beta: float,
              problem) -> List[Individual]:
    """
    Algorithm 4: Update_Id

    1. Compute Iᵈ fitness for all solutions in MP.
    2. Identify exploitation-region solutions (Theorem 4):
           {x | Iᵈ(x|MP) ≥ 0}
    3a. If |P₂| ≤ N → select N best from MP.
    3b. If |P₂| > N → decomposition-based pruning (Eq. 24):
           Iteratively remove the individual with worst fitness
           in the densest subregion, recalculating fitness each time.
    """
    for ind in MP:
        if ind.F is None:
            problem.evaluate(ind)

    fitness = compute_Id(MP, alpha, beta)

    # Theorem 4: Iᵈ(xⁱ|P) ≥ 0 → individual in exploitation region
    p2_idx = [i for i in range(len(MP)) if fitness[i] >= 0.0]
    P2 = [MP[i] for i in p2_idx]
    fit2 = fitness[np.array(p2_idx)] if p2_idx else np.array([])

    if len(P2) <= N:
        # Select N best from entire MP by fitness
        sorted_idx = np.argsort(fitness)[::-1]
        return [MP[i] for i in sorted_idx[:N]]

    # Decomposition-based pruning
    z_star = np.min(np.array([ind.F for ind in MP]), axis=0)

    # assign each to subregion
    subregion = [assign_subregion(ind, weight_vectors, z_star) for ind in P2]

    # iteratively remove worst in densest subregion
    active = list(range(len(P2)))

    while len(active) > N:
        # recompute fitness on currently active subset
        active_pop = [P2[i] for i in active]
        fit_active = compute_Id(active_pop, alpha, beta)

        # find densest subregion
        sr_active = [subregion[i] for i in active]
        from collections import Counter
        counts = Counter(sr_active)
        densest = max(counts, key=lambda k: counts[k])

        # among individuals in the densest subregion, remove worst fitness
        in_densest = [j for j, sr in enumerate(sr_active) if sr == densest]
        worst_local = in_densest[int(np.argmin(fit_active[in_densest]))]
        active.pop(worst_local)

    return [P2[i] for i in active]


# ══════════════════════════════════════════════════════════════════
#  WEIGHT VECTOR INITIALISATION  (two-layer method [51])
# ══════════════════════════════════════════════════════════════════

def generate_weight_vectors(M: int, N: int) -> np.ndarray:
    """
    Simple uniform weight vector generation for M objectives.
    Uses Das-Dennis normal boundary intersection approach [51].
    Returns array of shape (N_actual, M) — may differ slightly from N.
    """
    def _rec(M, left, total, prefix, result):
        if M == 1:
            result.append(prefix + [left / total])
        else:
            for i in range(left + 1):
                _rec(M - 1, left - i, total, prefix + [i / total], result)

    # find H such that C(H+M-1, M-1) ≈ N
    H = 1
    while True:
        from math import comb
        if comb(H + M - 1, M - 1) >= N:
            break
        H += 1

    wv = []
    _rec(M, H, H, [], wv)
    return np.array(wv)


# ══════════════════════════════════════════════════════════════════
#  ALGORITHM 1 — ILCMO Main Loop
# ══════════════════════════════════════════════════════════════════

class ILCMO:
    """
    Algorithm 1: ILCMO

    Parameters
    ----------
    problem   : object with .evaluate(ind), .lb, .ub, .M attributes
    N         : population size
    K         : number of variable groups  (paper default: 4)
    pp        : descent-rate parameter for αₜ  (paper default: 0.5)
    pm        : polynomial mutation probability  (paper: 1/D)
    eta_m     : PM distribution index  (paper: 20)
    max_fe    : maximum function evaluations  (paper: 2000×D)
    seed      : random seed
    """

    def __init__(self, problem, N: int = 100, K: int = 4,
                 pp: float = 0.5, pm: Optional[float] = None,
                 eta_m: float = 20.0, max_fe: Optional[int] = None,
                 seed: int = 42):
        self.problem = problem
        self.N       = N
        self.K       = K
        self.pp      = pp
        self.eta_m   = eta_m
        self.rng     = np.random.default_rng(seed)

        D = len(problem.lb)
        self.pm     = pm if pm is not None else 1.0 / D
        self.max_fe = max_fe if max_fe is not None else 2000 * D

        # variable groups (ordered grouping)
        self.groups = ordered_grouping(D, K)

        # weight vectors for decomposition (Algorithm 4)
        self.weight_vectors = generate_weight_vectors(problem.M, N)

    # ──────────────────────────────────────────────────────────────
    def _init_population(self) -> List[Individual]:
        lb, ub = self.problem.lb, self.problem.ub
        D = len(lb)
        pop = []
        for _ in range(self.N):
            x = self.rng.uniform(lb, ub)
            ind = Individual(x)
            self.problem.evaluate(ind)
            pop.append(ind)
        return pop

    # ──────────────────────────────────────────────────────────────
    def _feasibility_ratio(self, pop: List[Individual]) -> float:
        n_feas = sum(1 for ind in pop if ind.phi == 0.0)
        return n_feas / len(pop)

    # ──────────────────────────────────────────────────────────────
    def run(self) -> Tuple[List[Individual], dict]:
        """
        Execute Algorithm 1 and return (final_P1, log_dict).

        Algorithm 1 pseudocode (paper):
          1: Init P₁, P₂  (size N)
          2: Evaluate P₁, P₂
          3: Init weight vectors Λ
          4: while not terminated:
          5:   Off₁ ← VGDE(P₁, G)
          6:   Off₂ ← VGDE(P₂, G)
          7:   MP₁  ← P₁ ∪ Off₁ ∪ Off₂
          8:   MP₂  ← P₁ ∪ P₂ ∪ Off₁ ∪ Off₂
          9:   Evaluate MP₁, MP₂
          10:  P₁ ← Update_Is(MP₁, N)
          11:  P₂ ← Update_Id(MP₂, N, Λ)
          12: end while
          return P₁
        """
        problem = self.problem
        N       = self.N
        lb, ub  = problem.lb, problem.ub

        # ── Lines 1–2: Initialise two populations ─────────────────
        P1 = self._init_population()   # evaluated by _init_population
        P2 = self._init_population()
        fe = 2 * N                     # function evaluation counter

        # compute α₀ from initial solutions (max constraint violation)
        all_init = P1 + P2
        alpha0 = max(ind.phi for ind in all_init)
        if alpha0 == 0.0:
            alpha0 = 1e-4              # fallback if all feasible

        # ── Line 3: Weight vectors (already done in __init__) ──────

        # logging
        log = {"fe": [], "igd": [], "hv": [], "pf_ratio": []}

        T_max = self.max_fe // (2 * N)   # approximate generation count
        t = 0

        # ── Lines 4–12: Main loop ──────────────────────────────────
        while fe < self.max_fe:
            t += 1

            # dynamic constraint boundaries
            alpha_t = compute_alpha(t, T_max, alpha0, self.pp)
            pf      = self._feasibility_ratio(P1)
            beta_t  = compute_beta(alpha_t, pf)

            # fitness for current populations
            fit1 = compute_Is(P1)
            fit2 = compute_Id(P2, alpha_t, beta_t)

            # ── Lines 5–6: VGDE reproduction ──────────────────────
            Off1 = reproduction_VGDE(P1, fit1, t, T_max,
                                     self.groups, lb, ub,
                                     self.pm, self.eta_m, self.rng)
            Off2 = reproduction_VGDE(P2, fit2, t, T_max,
                                     self.groups, lb, ub,
                                     self.pm, self.eta_m, self.rng)

            # evaluate offspring
            for ind in Off1 + Off2:
                problem.evaluate(ind)
                fe += 1

            # ── Lines 7–8: merge pools ─────────────────────────────
            MP1 = P1 + Off1 + Off2
            MP2 = P1 + P2 + Off1 + Off2

            # ── Lines 10–11: environmental selection ──────────────
            P1 = update_Is(MP1, N, problem)
            P2 = update_Id(MP2, N, self.weight_vectors,
                           alpha_t, beta_t, problem)

            # logging every 10 generations
            if t % 10 == 0:
                pf_ratio = self._feasibility_ratio(P1)
                log["fe"].append(fe)
                log["pf_ratio"].append(pf_ratio)

            if fe >= self.max_fe:
                break

        return P1, log

import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.rnd import RandomSelection 
from pymoo.problems import get_problem
from pymoo.optimize import minimize


def algorithm_3_environmental_selection(G, div, z_star, g_nad):
    num_solutions = len(G)
    if num_solutions == 0:
        return [], [], []
    
    G = np.array(G)
    M = G.shape[1]
    
    denominator = []
    gs = []
    lb = []
    for j in range(M):
        diff = g_nad[j] - z_star[j]
        if diff == 0: diff = 1e-6
        denominator.append(diff)
        
        # Sửa lỗi logic: div-1 phải >= 1 để tránh chia cho 0
        div_val = max(div, 2)
        gs_j = diff / (div_val - 1)
        if gs_j == 0: gs_j = 1e-6
        gs.append(gs_j)
        
        lb_j = z_star[j] - (gs_j / 2.0)
        lb.append(lb_j)

    I = []
    fitness = []
    
    for i in range(num_solutions):
        row_I = []
        sum_d_sq = 0.0
        for j in range(M):
            # Giữ nguyên cách tính chỉ số lưới của bạn
            I_ij = int(((G[i][j] - lb[j]) / gs[j]) // 1) 
            row_I.append(I_ij)
            
            gc_ij = lb[j] + I_ij * gs[j]
            g_norm_ij = (G[i][j] - z_star[j]) / denominator[j]
            gc_norm_ij = (gc_ij - z_star[j]) / denominator[j]
            
            d_ij = g_norm_ij - gc_norm_ij
            delta = 1.0e6 if I_ij == 0 else 1.0
            sum_d_sq += (delta * d_ij) ** 2
            
        I.append(row_I)
        fitness.append(sum_d_sq ** 0.5)
    
    # Logic loại bỏ dư thừa (Redundancy Elimination) - GIỮ NGUYÊN VÒNG LẶP i, j
    is_redundant = [False] * num_solutions
    for i in range(num_solutions - 1):
        if is_redundant[i]: continue
        for j in range(i + 1, num_solutions):
            if I[i] == I[j]:
                if fitness[i] < fitness[j]:
                    is_redundant[j] = True
                else:
                    is_redundant[i] = True
                    break 

    G_star, I_star, selected_indices = [], [], []
    for k in range(num_solutions):
        if not is_redundant[k]:
            G_star.append(G[k].tolist())
            I_star.append(I[k])
            selected_indices.append(k)
            
    return G_star, I_star, selected_indices


def algorithm_4_dynamic_grid_adjustment(G_initial, N, current_div, z_star, g_nad):
    G_star, I_star, sel_idx = algorithm_3_environmental_selection(G_initial, current_div, z_star, g_nad)
    num_solutions = len(G_star)
        
    if num_solutions > N:
        test_div = max(current_div - 1, 2)
        G_new, I_new, sel_idx_new = algorithm_3_environmental_selection(G_initial, test_div, z_star, g_nad)
        if len(G_new) >= N:
            return G_new, I_new, test_div, sel_idx_new
    elif num_solutions < N:
        return G_star, I_star, current_div + 1, sel_idx

    return G_star, I_star, current_div, sel_idx
    

def algorithm_5_selection(G, I, N):
    G = np.array(G)
    I = np.array(I)
    num_solutions = len(G)
    M = G.shape[1]

    crowding = np.zeros(num_solutions)
    for i in range(num_solutions):
        # Giữ nguyên cách tính Chebyshev của bạn
        chebyshev_dist = np.max(np.abs(I - I[i]), axis=1)
        crowding[i] = np.sum(chebyshev_dist == 1)
            
    root_crowding = np.power(crowding, 1.0 / M)
    max_root_crowding = np.max(root_crowding)
    
    if max_root_crowding == 0:
        fitness = np.ones(num_solutions)
    else:
        fitness = ((N - 1) * root_crowding / max_root_crowding) + 1
            
    probs = (1.0 / fitness) / np.sum(1.0 / fitness)
    
    # Sửa lỗi: replace=False chỉ dùng khi num_solutions >= N
    can_replace = num_solutions < N
    selected_indices = np.random.choice(
        np.arange(num_solutions), size=N, p=probs, replace=can_replace
    )
    
    return G[selected_indices], selected_indices

# =====================================================================
# AGEASurvival: GIỮ NGUYÊN CẤU TRÚC ĐIỀU PHỐI CỦA BẠN
# =====================================================================
class AGEASurvival(Survival):
    def __init__(self, init_div=15):
        super().__init__(filter_infeasible=True)
        self.div = init_div        
        self.nds = NonDominatedSorting()
        self.z_star = None
        self.z_nad = None

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        if n_survive is None: n_survive = len(pop) // 2
        F = pop.get("F")
        
        fronts = self.nds.do(F)
        NDP_op = F[fronts[0]] 
        
        z_star_current = np.min(F, axis=0)
        if self.z_star is None:
            self.z_star = z_star_current.copy()
        else:
            self.z_star = np.minimum(self.z_star, z_star_current)
            
        z_nad_temp = np.max(NDP_op, axis=0)

        # Algorithm 2: Ổn định lưới (giữ nguyên logic vòng lặp j)
        if self.z_nad is None:
            self.z_nad = z_nad_temp.copy()
        else: 
            gs = (self.z_nad - self.z_star) / (self.div - 1)
            gs = np.where(gs <= 0, 1e-6, gs)
            for j in range(len(self.z_nad)):
                if np.abs(z_nad_temp[j] - self.z_nad[j]) > gs[j] / 2:
                    self.z_nad[j] = z_nad_temp[j]

        candidates_indices = []
        for front in fronts:
            candidates_indices.extend(front)
            if len(candidates_indices) >= n_survive: break
                
        G_initial = F[candidates_indices].tolist()

        # Gọi Algorithm 4 & 3
        G_star, I_star, self.div, sel_idx_alg4 = algorithm_4_dynamic_grid_adjustment(
            G_initial, n_survive, self.div, self.z_star, self.z_nad
        )

        survivors_after_alg4 = [candidates_indices[i] for i in sel_idx_alg4]

        # Gọi Algorithm 5
        if len(survivors_after_alg4) > n_survive:
            _, final_sel_idx = algorithm_5_selection(G_star, I_star, n_survive)
            final_survivors_indices = [survivors_after_alg4[i] for i in final_sel_idx]
        else:
            final_survivors_indices = survivors_after_alg4

        # Safety fill
        if len(final_survivors_indices) < n_survive:
            missing = n_survive - len(final_survivors_indices)
            remaining = [idx for idx in candidates_indices if idx not in final_survivors_indices]
            final_survivors_indices.extend(remaining[:missing])

        return pop[final_survivors_indices]

class AGEA(GeneticAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

if __name__ == "__main__":
    problem = get_problem("zdt1", n_var=30)
    algorithm = AGEA(
        pop_size=100,
        sampling=FloatRandomSampling(),
        selection=RandomSelection(), 
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=10),
        survival=AGEASurvival(init_div=15)
    )

    print("Đang chạy tối ưu hóa ZDT1 bằng thuật toán AGEA...")
    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', 250), 
        seed=42,
        verbose=True
    )

    F = res.F
    plt.figure(figsize=(8, 6))
    
    # 1. Vẽ các nghiệm tìm được bởi AGEA (điểm màu đỏ)
    plt.scatter(F[:, 0], F[:, 1], c='red', marker='o', s=30, label='AGEA Solutions')

    # 2. Vẽ True Pareto Front lý thuyết (đường màu xanh)
    # ZDT1: f2 = 1 - sqrt(f1), với f1 nằm trong khoảng [0, 1]
    pf_f1 = np.linspace(0, 1, 100)
    pf_f2 = 1 - np.sqrt(pf_f1)
    plt.plot(pf_f1, pf_f2, c='blue', linewidth=2, label='True Pareto Front')

    # Định dạng biểu đồ
    plt.title("ZDT1 Optimization with AGEA and True Pareto Front")
    plt.xlabel("f1 objective")
    plt.ylabel("f2 objective")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Đặt giới hạn trục để dễ quan sát (hơi rộng hơn [0,1])
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    
    print("Đang hiển thị biểu đồ...")
    plt.show()

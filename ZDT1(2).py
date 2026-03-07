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

class AGEASurvival(Survival):
    def __init__(self, init_div=15, min_div=10, max_div=30):
        super().__init__(filter_infeasible=True)
        self.div = init_div        
        self.min_div = min_div     
        self.max_div = max_div     
        self.nds = NonDominatedSorting()
        self.z_star = None  # Ideal point
        self.z_nad = None   # Grid Nadir point 

    def _get_grid_indices(self, F, z_star, z_nad, div):
        """Tính I_ij theo công thức (7) """
        gs = (z_nad - z_star) / (div - 1)
        gs = np.where(gs == 0, 1e-6, gs)
        lb = z_star - gs / 2 # lb_j theo công thức (5)
        
        grid_indices = np.floor((F - lb) / gs).astype(int)
        return np.clip(grid_indices, 0, div - 1)

    def _compute_density(self, grid_indices):
        """Tính mật độ ô lưới bằng NumPy (Vectorized)"""
        _, inverse_indices, counts = np.unique(grid_indices, axis=0, 
                                               return_inverse=True, 
                                               return_counts=True)
        return counts[inverse_indices]

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        if n_survive is None:
            n_survive = len(pop) // 2

        F = pop.get("F")
        
        #  NON-DOMINATED SELECTION 
        fronts = self.nds.do(F)
        NDP_op = F[fronts[0]] 
        
        #  CẬP NHẬT IDEAL POINT & NADIR POINT ---
        z_star_current = np.min(F, axis=0)
        if self.z_star is None:
            self.z_star = z_star_current
        else:
            self.z_star = np.minimum(self.z_star, z_star_current)
            
        # Nadir point tạm thời tính từ NDP_op (Eq 3)
        z_nad_temp = np.max(NDP_op, axis=0)

        #-------------------------------------------------
        # --- ALGORITHM 2: GRID STABILIZATION STRATEGY ---
        #-------------------------------------------------
        if self.z_nad is None:
            self.z_nad = z_nad_temp
        else: 
            gs = (self.z_nad - self.z_star) / (self.div - 1)
            gs = np.where(gs <= 0, 1e-6, gs)
            
            for j in range(len(self.z_nad)):
                if np.abs(z_nad_temp[j] - self.z_nad[j]) > gs[j] / 2:
                    self.z_nad[j] = z_nad_temp[j]

        #---------------------------------------------
        # --- ALGORITHM 3: ENVIRONMENTAL SELECTION ---
        #---------------------------------------------

        survivors = []
        last_front_info = None 
        
        for front in fronts:
            if len(survivors) + len(front) <= n_survive:
                survivors.extend(front)
                if len(survivors) == n_survive: break
            else:
                remaining = n_survive - len(survivors)
                front_indices = np.array(front)
                
                grid_idx = self._get_grid_indices(F[front_indices], self.z_star, self.z_nad, self.div)
                scores = self._compute_density(grid_idx)
                
                sorted_idx = np.argsort(scores)
                survivors.extend(front_indices[sorted_idx[:remaining]])
                last_front_info = (front_indices, remaining)
                break

        #----------------------------------------------
        # --- ALGORITHM 4: GRID ADAPTIVE ADJUSTMENT ---
        #----------------------------------------------

        grid_indices_survivors = self._get_grid_indices(F[survivors], self.z_star, self.z_nad, self.div)
        unique_grids = np.unique(grid_indices_survivors, axis=0)
        ratio = len(unique_grids) / n_survive
        
        new_div = self.div
        if ratio < 0.3:
            new_div = min(self.div + 2, self.max_div)
        elif ratio > 0.8:
            new_div = max(self.div - 1, self.min_div)

        #--------------------------------------------
        # --- ALGORITHM 5: POPULATION RESELECTION ---
        #--------------------------------------------

        if new_div != self.div:
            self.div = new_div
            if last_front_info is not None:
                front_indices, remaining = last_front_info
                survivors = survivors[:-remaining] # Loại bỏ phần cũ
                
                # Tính lại với div mới
                grid_idx_new = self._get_grid_indices(F[front_indices], self.z_star, self.z_nad, self.div)
                scores_new = self._compute_density(grid_idx_new)
                
                sorted_idx_new = np.argsort(scores_new)
                survivors.extend(front_indices[sorted_idx_new[:remaining]])

        return pop[survivors]

class AGEA(GeneticAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize_advance(self, infills=None, **kwargs):
        # Gọi hàm khởi tạo gốc của Pymoo
        super()._initialize_advance(infills, **kwargs)
        
        # Trích xuất F của quần thể Thế hệ 0
        F = self.pop.get("F")
        
        # Khởi tạo z_star và z_nad truyền thẳng cho Survival
        if self.survival is not None:
            self.survival.z_star = np.min(F, axis=0)
            self.survival.z_nad = np.max(F, axis=0)

# ==========================================
# 3. KHỞI TẠO VÀ CHẠY THỬ NGHIỆM TRÊN ZDT1
# ==========================================
if __name__ == "__main__":
    problem = get_problem("zdt1", n_var=30)

    
    algorithm = AGEA(
        pop_size=100,
        sampling=FloatRandomSampling(),
        selection=RandomSelection(), 
        crossover=SBX(prob=0.9, eta=20),
        mutation=PM(eta=20),
        survival=AGEASurvival(init_div=15, min_div=10, max_div=30)
    )

    print("Đang chạy tối ưu hóa ZDT1(AGEA)")
    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', 250), 
        seed=42,
        verbose=True
    )

    F = res.F
    plt.figure(figsize=(8, 6))
    plt.scatter(F[:, 0], F[:, 1], c='red', marker='o', label='AGEA Solutions')

    f1_true = np.linspace(0, 1, 100)
    f2_true = 1 - np.sqrt(f1_true)
    plt.plot(f1_true, f2_true, c='blue', linewidth=2, label='True Pareto Front')

    plt.title("Tối ưu ZDT1 bằng thuật toán AGE")
    plt.xlabel("Mục tiêu f1")
    plt.ylabel("Mục tiêu f2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
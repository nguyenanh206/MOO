import numpy as np
import matplotlib.pyplot as plt

# Import các module từ pymoo
from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.rnd import RandomSelection 
from pymoo.problems import get_problem
from pymoo.optimize import minimize


# ==========================================
# ==========================================
class AGEASurvival(Survival):
    def __init__(self, init_div=15, min_div=10, max_div=30, alpha=0.1):
        super().__init__(filter_infeasible=True)
        self.div = init_div        
        self.min_div = min_div     
        self.max_div = max_div     
        self.alpha = alpha         
        self.nds = NonDominatedSorting()
        self.z_star = None
        self.z_nad = None

    def _get_grid_indices(self, F, z_star, z_nad, div):
        denominator = z_nad - z_star
        denominator = np.where(denominator == 0, 1e-6, denominator)
        grid_indices = np.floor(div * (F - z_star) / denominator).astype(int)
        grid_indices = np.clip(grid_indices, 0, div - 1)
        return grid_indices

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        if n_survive is None:
            n_survive = len(pop) // 2

        F = pop.get("F")
        
        # ALGORITHM 2: Grid Stabilization Strategy
        z_star_current = np.min(F, axis=0)
        z_nad_current = np.max(F, axis=0)
        
        if self.z_star is None or self.z_nad is None:
            self.z_star = z_star_current
            self.z_nad = z_nad_current
        else:
            self.z_star = np.minimum(self.z_star, z_star_current)
            z_nad_new = np.zeros_like(self.z_nad)
            for i in range(len(self.z_nad)):
                if z_nad_current[i] > self.z_nad[i]:
                    z_nad_new[i] = z_nad_current[i]
                else:
                    z_nad_new[i] = (1 - self.alpha) * self.z_nad[i] + self.alpha * z_nad_current[i]
            self.z_nad = z_nad_new

        # ALGORITHM 3: Environmental Selection
        fronts = self.nds.do(F)
        survivors = []         
        front_c_indices = []   
        remaining = 0          
        
        for front in fronts:
            if len(survivors) + len(front) <= n_survive:
                survivors.extend(front)
                if len(survivors) == n_survive:
                    break
            else:
                remaining = n_survive - len(survivors)
                front_c_indices = np.array(front) 
                F_front = F[front_c_indices]
                
                grid_indices = self._get_grid_indices(F_front, self.z_star, self.z_nad, self.div)
                density = {}
                for g in grid_indices:
                    gt = tuple(g)
                    density[gt] = density.get(gt, 0) + 1
                    
                scores = [density[tuple(g)] for g in grid_indices]
                sorted_idx = np.argsort(scores)
                
                selected_from_front = front_c_indices[sorted_idx[:remaining]]
                survivors.extend(selected_from_front)
                break

        # ALGORITHM 4: Grid Adaptive Adjustment Strategy
        F_survivors = F[survivors]
        grid_indices_survivors = self._get_grid_indices(F_survivors, self.z_star, self.z_nad, self.div)
        
        active_grids = set(tuple(g) for g in grid_indices_survivors)
        ratio = len(active_grids) / n_survive
        
        new_div = self.div
        if ratio < 0.3:
            new_div = min(self.div + 2, self.max_div)
        elif ratio > 0.8:
            new_div = max(self.div - 1, self.min_div)

        # ALGORITHM 5: Population Reselection
        
        if new_div != self.div:
            self.div = new_div
            
            if remaining > 0 and len(front_c_indices) > 0:
                survivors = survivors[:-remaining]
                
                F_front = F[front_c_indices]
                grid_indices_new = self._get_grid_indices(F_front, self.z_star, self.z_nad, self.div)
                
                density_new = {}
                for g in grid_indices_new:
                    gt = tuple(g)
                    density_new[gt] = density_new.get(gt, 0) + 1
                    
                scores_new = [density_new[tuple(g)] for g in grid_indices_new]
                sorted_idx_new = np.argsort(scores_new)
                
                reselected_from_front = front_c_indices[sorted_idx_new[:remaining]]
                survivors.extend(reselected_from_front)

        return pop[survivors]

# ==========================================
# KHỞI TẠO VÀ CHẠY THỬ NGHIỆM TRÊN ZDT1
# ==========================================
if __name__ == "__main__":
    problem = get_problem("zdt1", n_var=30)

    algorithm = GeneticAlgorithm(
        pop_size=100,
        sampling=FloatRandomSampling(),
        selection=RandomSelection(), # An toàn nhất cho đa mục tiêu
        crossover=SBX(prob=0.9, eta=20),
        mutation=PM(eta=20),
        survival=AGEASurvival(init_div=15, min_div=10, max_div=30, alpha=0.1)
    )

    print(" ")
    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', 250), # 250 thế hệ
        seed=42,
        verbose=True
    )

    F = res.F
    plt.figure(figsize=(8, 6))
    plt.scatter(F[:, 0], F[:, 1], c='red', marker='o', label='AGEA Solutions')

    f1_true = np.linspace(0, 1, 100)
    f2_true = 1 - np.sqrt(f1_true)
    plt.plot(f1_true, f2_true, c='blue', label='True Pareto Front')

    plt.title("ZDT1 (AGEA)")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.legend()
    plt.grid()
    plt.show()
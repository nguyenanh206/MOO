import numpy as np
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

class Individual:
    def __init__(self, solution, fitness=None):
        self.solution = solution
        self.fitness = fitness
        self.indicator = 0.0

class MIMOABC:
    def __init__(self, n, max_fes, dim, m, obj_func):
        self.n = n
        self.max_fes = max_fes
        self.fes = 0
        self.dim = dim
        self.m = m
        self.obj_func = obj_func
        self.kappa = 0.05
        
        self.pt = [Individual(np.random.rand(dim)) for _ in range(n)]
        self.evaluate(self.pt)

    def evaluate(self, population):
        for ind in population:
            if self.fes < self.max_fes:
                ind.fitness = self.obj_func(ind.solution)
                self.fes += 1

    # --- THUẬT TOÁN 1: Chọn lọc dựa trên Hội tụ ---
    def environmental_selection_1(self, pt, qt, n_target):
        union_p = pt + qt
        temp_p = list(union_p)
        for x in temp_p:
            x.indicator = 0
            for y in temp_p:
                if x != y:
                    eps_y_x = np.max(y.fitness - x.fitness)
                    x.indicator += -np.exp(-eps_y_x / self.kappa)
        
        while len(temp_p) > n_target:
            worst = min(temp_p, key=lambda x: x.indicator)
            for x in temp_p:
                if x != worst:
                    eps_worst_x = np.max(worst.fitness - x.fitness)
                    x.indicator -= -np.exp(-eps_worst_x / self.kappa)
            temp_p.remove(worst)
        return temp_p

    # --- THUẬT TOÁN 2: Sử dụng Pymoo cho Non-dominated Sorting ---
    def calculate_pd_matrix(self, pop):
        fitness_matrix = np.array([ind.fitness for ind in pop])
        f_min, f_max = fitness_matrix.min(axis=0), fitness_matrix.max(axis=0)
        norm_fit = (fitness_matrix - f_min) / (f_max - f_min + 1e-10)
        num = len(pop)
        sum_f = np.sum(norm_fit, axis=1)[:, np.newaxis]
        weights = norm_fit / (sum_f + 1e-10)
        pd_matrix = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                if i == j: pd_matrix[i, j] = np.inf
                else:
                    w_j = weights[j]
                    norm_w_j = np.linalg.norm(w_j)
                    proj_len = np.dot(norm_fit[i], w_j) / (norm_w_j + 1e-10)
                    proj_vec = proj_len * (w_j / (norm_w_j + 1e-10))
                    pd_matrix[i, j] = np.linalg.norm(norm_fit[i] - proj_vec)
        return pd_matrix

    def environmental_selection_2(self, pt_after_s1, qt_prime, n):
        combined = pt_after_s1 + qt_prime
        
        # SỬ DỤNG PYMOO GỌI FAST NON-DOMINATED SORT
        fitness_matrix = np.array([ind.fitness for ind in combined])
        nds = NonDominatedSorting()
        front_indices = nds.do(fitness_matrix)
        fronts = [[combined[idx] for idx in f] for f in front_indices]
        
        pt_next, critical_layer = [], []
        for front in fronts:
            if len(pt_next) + len(front) <= n:
                pt_next.extend(front)
            else:
                critical_layer = front
                break
        
        num_needed = n - len(pt_next)
        if num_needed <= 0: return pt_next[:n]

        # Trường hợp 1: Fi != F1 (pt_next không rỗng nghĩa là đã chứa ít nhất F1)
        if len(pt_next) > 0:
            pd_matrix = self.calculate_pd_matrix(critical_layer)
            div_scores = np.min(pd_matrix, axis=1)
            sorted_indices = np.argsort(div_scores)[::-1]
            for i in range(num_needed): pt_next.append(critical_layer[sorted_indices[i]])
            
        # Trường hợp 2: Fi == F1 (pt_next rỗng)
        else:
            f1 = list(critical_layer)
            fit_mat = np.array([ind.fitness for ind in f1])
            extreme_indices = {np.argmin(fit_mat[:, m]) for m in range(self.m)}
            extreme_solutions = [f1[i] for i in extreme_indices]
            remaining_f1 = [f1[i] for i in range(len(f1)) if i not in extreme_indices]
            
            while len(extreme_solutions) + len(remaining_f1) > n:
                current_f1 = extreme_solutions + remaining_f1
                pd_mat = self.calculate_pd_matrix(current_f1)
                min_pd, pair = np.inf, (0, 1)
                for i in range(len(current_f1)):
                    for j in range(i + 1, len(current_f1)):
                        if pd_mat[i, j] < min_pd:
                            if i >= len(extreme_solutions) or j >= len(extreme_solutions):
                                min_pd, pair = pd_mat[i, j], (i, j)
                idx1, idx2 = pair
                to_remove = current_f1[idx1] if current_f1[idx1].indicator < current_f1[idx2].indicator else current_f1[idx2]
                if to_remove in remaining_f1: remaining_f1.remove(to_remove)
                else: extreme_solutions.remove(to_remove)
            pt_next = extreme_solutions + remaining_f1
            
        return pt_next

    # --- KHUNG CHƯƠNG TRÌNH CHÍNH ---
    def solve(self):
        while self.fes < self.max_fes:
            # Bước 2: Ong thợ -> Qt
            qt = []
            self.environmental_selection_1(self.pt, [], self.n) 
            x_best = max(self.pt, key=lambda x: x.indicator)
            for i in range(self.n):
                k = np.random.randint(0, self.n)
                phi, psi = np.random.uniform(-1, 1, self.dim), np.random.uniform(0, 1, self.dim)
                v_sol = self.pt[i].solution + phi * (self.pt[i].solution - self.pt[k].solution) + \
                        psi * (x_best.solution - self.pt[i].solution)
                qt.append(Individual(np.clip(v_sol, 0, 1)))
            self.evaluate(qt)

            # Bước 3: Thuật toán 1 (Hội tụ)
            pt_after_s1 = self.environmental_selection_1(self.pt, qt, self.n)

            # Bước 4: Ong quan sát -> Qt_onlooker
            qt_onlooker = []
            for i in range(self.n):
                r1, r2 = np.random.choice(self.n, 2, replace=False)
                phi = np.random.uniform(-1, 1, self.dim)
                v_sol = pt_after_s1[i].solution + phi * (pt_after_s1[r1].solution - pt_after_s1[r2].solution)
                qt_onlooker.append(Individual(np.clip(v_sol, 0, 1)))
            self.evaluate(qt_onlooker)

            # Bước 5: Ong trinh thám -> Qt_prime
            qt_onlooker = self.environmental_selection_1(pt_after_s1, qt_onlooker, self.n)
            qt_onlooker.sort(key=lambda x: x.indicator)
            num_scouts = int(0.1 * self.n)
            for i in range(num_scouts):
                r = np.random.uniform(-1, 1, self.dim)
                v_scout = x_best.solution + r * (x_best.solution - qt_onlooker[i].solution)
                qt_onlooker[i] = Individual(np.clip(v_scout, 0, 1))
            self.evaluate(qt_onlooker)
            qt_prime = qt_onlooker

            # Bước 6: Thuật toán 2 (Đa dạng)
            self.pt = self.environmental_selection_2(pt_after_s1, qt_prime, self.n)

        return self.pt

'''
# Hàm mục tiêu mẫu (Cực tiểu hóa)
def dummy_obj(x):
    return np.array([np.sum(x**2), np.sum((x-1)**2)])

# Chạy thử
mimo = MIMOABC(n=40, max_fes=4000, dim=5, m=2, obj_func=dummy_obj)
result = mimo.solve()
print(f"Hoàn thành. Số cá thể tối ưu: {len(result)}")
'''
def plot_results(result_pop):
    fitness_values = np.array([ind.fitness for ind in result_pop])
    m = fitness_values.shape[1]

    if m == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(fitness_values[:, 0], fitness_values[:, 1], c='red', edgecolors='k', alpha=0.7)
        plt.xlabel('Mục tiêu 1')
        plt.ylabel('Mục tiêu 2')
        plt.title('Mặt Pareto MIMOABC (2D)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
    elif m == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(fitness_values[:, 0], fitness_values[:, 1], fitness_values[:, 2], c='blue', edgecolors='k', alpha=0.7)
        ax.set_xlabel('Mục tiêu 1')
        ax.set_ylabel('Mục tiêu 2')
        ax.set_zlabel('Mục tiêu 3')
        ax.set_title('Mặt Pareto MIMOABC (3D)')
        plt.show()
    else:
        print(f"Không thể vẽ đồ thị trực tiếp cho {m} mục tiêu.")

# Ví dụ hàm mục tiêu mẫu (ZDT1)
def zdt1(x):
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    f2 = g * (1 - np.sqrt(f1 / g))
    return np.array([f1, f2])

if __name__ == "__main__":
    # Khởi tạo thông số
    mimo = MIMOABC(n=100, max_fes=10000, dim=30, m=2, obj_func=zdt1)
    
    # Chạy thuật toán
    final_pareto = mimo.solve()
    
    # Vẽ đồ thị
    plot_results(final_pareto)
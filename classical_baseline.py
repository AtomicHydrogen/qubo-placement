import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from itertools import combinations

class ClassicalQUBOSolver:
    """
    Classical solver for the SAME discrete QUBO problem.
    Uses deterministic/greedy methods instead of simulated annealing.
    """
    
    def __init__(self, m, n, grid_size, F, D):
        """
        m: number of blocks
        n: number of grid locations
        grid_size: size of square grid
        F: flow matrix (m x m) - connectivity between blocks
        D: distance matrix (n x n) - Manhattan distances
        """
        self.m = m
        self.n = n
        self.grid_size = grid_size
        self.F = F
        self.D = D
        
        # Construct QUBO components (same as quantum version)
        self.Q = np.kron(F, D)
        self.A = np.kron(np.eye(m), np.ones((1, n)))
        self.B = np.kron(np.ones((1, m)), np.eye(n))
        
        print(f"[CLASSICAL QUBO SOLVER] Initialized")
        print(f"   Problem size: {m} blocks, {n} locations")
        print(f"   QUBO dimension: {m*n + n} variables")
    
    def construct_qubo(self, lambda_penalty, mu_penalty):
        """Construct the QUBO matrix (same as quantum formulation)"""
        mn = self.m * self.n
        total_vars = mn + self.n
        
        QUBO = np.zeros((total_vars, total_vars))
        
        # 1. Main objective
        QUBO[:mn, :mn] += self.Q
        
        # 2. Constraint 1: lambda ||Ax - 1_m||^2
        QUBO[:mn, :mn] += lambda_penalty * (self.A.T @ self.A)
        linear_term_1 = -2 * lambda_penalty * (np.ones(self.m) @ self.A)
        for i in range(mn):
            QUBO[i, i] += linear_term_1[i]
        
        # 3. Constraint 2: mu ||Bx - s||^2
        QUBO[:mn, :mn] += mu_penalty * (self.B.T @ self.B)
        for i in range(mn):
            for j in range(self.n):
                QUBO[i, mn + j] += -2 * mu_penalty * self.B[j, i]
                QUBO[mn + j, i] += -2 * mu_penalty * self.B[j, i]
        
        for j in range(self.n):
            QUBO[mn + j, mn + j] += mu_penalty
        
        return QUBO
    
    def evaluate_qubo(self, x, QUBO):
        """Evaluate QUBO energy for binary vector x"""
        return x @ QUBO @ x
    
    def greedy_constructive(self, QUBO):
        """
        Greedy constructive heuristic: build solution one variable at a time
        by choosing the bit that minimally increases the objective.
        """
        print("\n[METHOD] Greedy Constructive Heuristic")
        mn = self.m * self.n
        total_vars = mn + self.n
        
        x = np.zeros(total_vars, dtype=int)
        
        # For each block, assign to best location greedily
        for block_id in range(self.m):
            best_loc = None
            best_energy = float('inf')
            
            for loc_id in range(self.n):
                var_idx = block_id * self.n + loc_id
                
                # Try setting this variable to 1
                x_trial = x.copy()
                x_trial[var_idx] = 1
                
                energy = self.evaluate_qubo(x_trial, QUBO)
                
                if energy < best_energy:
                    best_energy = energy
                    best_loc = loc_id
            
            if best_loc is not None:
                var_idx = block_id * self.n + best_loc
                x[var_idx] = 1
        
        # Set slack variables based on column sums
        for loc_id in range(self.n):
            col_sum = 0
            for block_id in range(self.m):
                var_idx = block_id * self.n + loc_id
                col_sum += x[var_idx]
            x[mn + loc_id] = min(col_sum, 1)
        
        energy = self.evaluate_qubo(x, QUBO)
        print(f"   Initial energy: {energy:.2f}")
        
        return x, energy
    
    def local_search(self, x, QUBO, max_iterations=1000):
        """
        Local search: flip bits to improve objective.
        Explores 1-bit and 2-bit flips.
        """
        print("\n[METHOD] Local Search Optimization")
        mn = self.m * self.n
        
        current_x = x.copy()
        current_energy = self.evaluate_qubo(current_x, QUBO)
        
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Try 1-bit flips (only in placement variables)
            for var_idx in range(mn):
                x_trial = current_x.copy()
                x_trial[var_idx] = 1 - x_trial[var_idx]
                
                # Update slack variables
                loc_id = var_idx % self.n
                col_sum = sum(x_trial[i * self.n + loc_id] for i in range(self.m))
                x_trial[mn + loc_id] = min(col_sum, 1)
                
                trial_energy = self.evaluate_qubo(x_trial, QUBO)
                
                if trial_energy < current_energy:
                    current_x = x_trial
                    current_energy = trial_energy
                    improved = True
                    if iteration % 100 == 0:
                        print(f"   Iteration {iteration}: energy = {current_energy:.2f}")
                    break
            
            # Try 2-bit swaps (swap two block locations)
            if not improved:
                for block1 in range(self.m):
                    for block2 in range(block1 + 1, self.m):
                        # Find current locations
                        loc1 = None
                        loc2 = None
                        for loc in range(self.n):
                            if current_x[block1 * self.n + loc] == 1:
                                loc1 = loc
                            if current_x[block2 * self.n + loc] == 1:
                                loc2 = loc
                        
                        if loc1 is not None and loc2 is not None and loc1 != loc2:
                            # Try swapping
                            x_trial = current_x.copy()
                            x_trial[block1 * self.n + loc1] = 0
                            x_trial[block1 * self.n + loc2] = 1
                            x_trial[block2 * self.n + loc2] = 0
                            x_trial[block2 * self.n + loc1] = 1
                            
                            # Update slack variables
                            for loc in [loc1, loc2]:
                                col_sum = sum(x_trial[i * self.n + loc] for i in range(self.m))
                                x_trial[mn + loc] = min(col_sum, 1)
                            
                            trial_energy = self.evaluate_qubo(x_trial, QUBO)
                            
                            if trial_energy < current_energy:
                                current_x = x_trial
                                current_energy = trial_energy
                                improved = True
                                if iteration % 100 == 0:
                                    print(f"   Iteration {iteration}: energy = {current_energy:.2f} (swap)")
                                break
                    if improved:
                        break
        
        print(f"   Final energy after {iteration} iterations: {current_energy:.2f}")
        return current_x, current_energy
    
    def multi_start_local_search(self, QUBO, num_starts=5):
        """
        Multiple random starts with local search.
        Returns the best solution found.
        """
        print(f"\n[METHOD] Multi-Start Local Search ({num_starts} starts)")
        
        best_x = None
        best_energy = float('inf')
        
        for start in range(num_starts):
            print(f"\n--- Start {start + 1}/{num_starts} ---")
            
            # Random initialization
            x = self._random_feasible_solution()
            energy = self.evaluate_qubo(x, QUBO)
            print(f"   Random init energy: {energy:.2f}")
            
            # Local search
            x_opt, energy_opt = self.local_search(x, QUBO, max_iterations=500)
            
            if energy_opt < best_energy:
                best_energy = energy_opt
                best_x = x_opt
                print(f"   [NEW BEST] Energy: {best_energy:.2f}")
        
        return best_x, best_energy
    
    def _random_feasible_solution(self):
        """Generate random feasible solution"""
        mn = self.m * self.n
        total_vars = mn + self.n
        
        x = np.zeros(total_vars, dtype=int)
        
        # Randomly assign each block to a location
        available_locs = list(range(self.n))
        
        for block_id in range(self.m):
            if len(available_locs) > 0:
                loc = np.random.choice(available_locs)
                var_idx = block_id * self.n + loc
                x[var_idx] = 1
                available_locs.remove(loc)
        
        # Set slack variables
        for loc_id in range(self.n):
            col_sum = sum(x[i * self.n + loc_id] for i in range(self.m))
            x[mn + loc_id] = min(col_sum, 1)
        
        return x
    
    def solve(self, lambda_penalty, mu_penalty, method='greedy+local'):
        """
        Solve the QUBO using classical methods.
        
        Methods:
        - 'greedy': Greedy constructive only
        - 'greedy+local': Greedy + local search
        - 'multi-start': Multiple random starts with local search
        """
        print(f"\n{'='*60}")
        print(f"CLASSICAL QUBO SOLVER")
        print(f"{'='*60}")
        print(f"Penalties: lambda={lambda_penalty}, mu={mu_penalty}")
        
        QUBO = self.construct_qubo(lambda_penalty, mu_penalty)
        
        if method == 'greedy':
            x, energy = self.greedy_constructive(QUBO)
        elif method == 'greedy+local':
            x_init, _ = self.greedy_constructive(QUBO)
            x, energy = self.local_search(x_init, QUBO)
        elif method == 'multi-start':
            x, energy = self.multi_start_local_search(QUBO, num_starts=5)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Extract placement
        placement = self._extract_placement(x)
        
        # Compute actual wirelength
        wirelength = self._compute_wirelength(placement)
        
        print(f"\n[FINAL SOLUTION]")
        print(f"   QUBO energy: {energy:.2f}")
        print(f"   Wirelength: {wirelength:.2f}")
        print(f"   Valid placement: {self._check_validity(placement)}")
        
        return {
            'x': x,
            'energy': energy,
            'placement': placement,
            'wirelength': wirelength,
            'valid': self._check_validity(placement)
        }
    
    def _extract_placement(self, x):
        """Extract placement dictionary from binary vector"""
        placement = {}
        for block_id in range(self.m):
            for loc_id in range(self.n):
                var_idx = block_id * self.n + loc_id
                if x[var_idx] == 1:
                    placement[block_id] = loc_id
                    break
        return placement
    
    def _compute_wirelength(self, placement):
        """Compute total wirelength"""
        total_wl = 0.0
        for i in range(self.m):
            for j in range(i+1, self.m):
                if self.F[i, j] > 0 and i in placement and j in placement:
                    loc_i = placement[i]
                    loc_j = placement[j]
                    
                    x_i = loc_i % self.grid_size
                    y_i = loc_i // self.grid_size
                    x_j = loc_j % self.grid_size
                    y_j = loc_j // self.grid_size
                    
                    dist = abs(x_i - x_j) + abs(y_i - y_j)
                    total_wl += self.F[i, j] * dist
        return total_wl
    
    def _check_validity(self, placement):
        """Check if placement is valid"""
        # Check each block placed once
        if len(placement) != self.m:
            return False
        
        # Check no overlaps
        locations = list(placement.values())
        if len(locations) != len(set(locations)):
            return False
        
        return True
    
    def visualize_placement(self, placement, wirelength, title="Classical Placement", save_path=None):
        """Visualize the placement"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Draw grid
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        for i in range(self.grid_size + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
            ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
        
        # Draw connections
        for i in range(self.m):
            for j in range(i+1, self.m):
                if self.F[i, j] > 0 and i in placement and j in placement:
                    loc_i = placement[i]
                    loc_j = placement[j]
                    x_i = loc_i % self.grid_size
                    y_i = loc_i // self.grid_size
                    x_j = loc_j % self.grid_size
                    y_j = loc_j // self.grid_size
                    ax.plot([x_i, x_j], [y_i, y_j], 
                           color='steelblue', linewidth=2, alpha=0.6, zorder=1)
        
        # Draw blocks
        for block_id, loc_id in placement.items():
            x = loc_id % self.grid_size
            y = loc_id // self.grid_size
            
            circle = plt.Circle((x, y), 0.35, color='lightgreen', 
                              ec='darkgreen', linewidth=2, zorder=2)
            ax.add_patch(circle)
            
            ax.text(x, y, str(block_id), ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white', zorder=3)
        
        ax.set_xlabel('X coordinate', fontsize=12)
        ax.set_ylabel('Y coordinate', fontsize=12)
        ax.set_title(f'{title}\nWirelength: {wirelength:.2f}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.set_xticks(range(self.grid_size))
        ax.set_yticks(range(self.grid_size))
        
        # Add grid coordinates
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                loc_id = i * self.grid_size + j
                if loc_id not in placement.values():
                    ax.text(j, i, f'{loc_id}', ha='center', va='center',
                           fontsize=7, color='gray', alpha=0.4, zorder=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[SAVE] Visualization saved to: {save_path}")
            plt.close(fig)
        
        return fig


def compare_classical_vs_sa(qubo_placer, classical_solver, lambda_penalty, mu_penalty, 
                            qubo_result, save_path=None):
    """Compare classical solver vs simulated annealing results"""
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Classical placement
    ax1 = axes[0]
    placement_classical = classical_solver._extract_placement(qubo_result['x'])
    wl_classical = qubo_result['wirelength']
    
    # QUBO placement (from simulated annealing)
    # Assume we have the QUBO result stored
    
    for idx, (ax, placement, wl, title, color) in enumerate([
        (axes[0], placement_classical, wl_classical, 
         f'Classical (Greedy+Local Search)\nWirelength: {wl_classical:.2f}', 'lightgreen'),
        (axes[1], None, None, 
         f'Simulated Annealing (Neal)\nWirelength: TBD', 'lightcoral')
    ]):
        if placement is None:
            ax.text(0.5, 0.5, 'Run SA solver to compare', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title(title, fontsize=13, fontweight='bold')
            continue
        
        # Draw grid
        ax.set_xlim(-0.5, classical_solver.grid_size - 0.5)
        ax.set_ylim(-0.5, classical_solver.grid_size - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        for i in range(classical_solver.grid_size + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
            ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        
        # Draw connections
        for i in range(classical_solver.m):
            for j in range(i+1, classical_solver.m):
                if classical_solver.F[i, j] > 0 and i in placement and j in placement:
                    loc_i = placement[i]
                    loc_j = placement[j]
                    x_i = loc_i % classical_solver.grid_size
                    y_i = loc_i // classical_solver.grid_size
                    x_j = loc_j % classical_solver.grid_size
                    y_j = loc_j // classical_solver.grid_size
                    ax.plot([x_i, x_j], [y_i, y_j], 
                           color='steelblue', linewidth=2, alpha=0.6, zorder=1)
        
        # Draw blocks
        for block_id, loc_id in placement.items():
            x = loc_id % classical_solver.grid_size
            y = loc_id // classical_solver.grid_size
            
            circle = plt.Circle((x, y), 0.35, color=color, 
                              ec='darkgreen' if idx == 0 else 'darkred', 
                              linewidth=2, zorder=2)
            ax.add_patch(circle)
            
            ax.text(x, y, str(block_id), ha='center', va='center',
                   fontsize=11, fontweight='bold', color='white', zorder=3)
        
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_xticks(range(classical_solver.grid_size))
        ax.set_yticks(range(classical_solver.grid_size))
    
    plt.suptitle('Classical vs Simulated Annealing Comparison', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVE] Comparison saved to: {save_path}")
        plt.close(fig)
    
    return fig
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dimod import BinaryQuadraticModel
from dwave.system import DWaveSampler, EmbeddingComposite
import networkx as nx
import os

class SpectralCyclicExpansion:
    """
    Spectral Cyclic Expansion for FPGA Placement
    
    Combines:
    1. Scattered Spectral initialization (Topology aware + Random shape)
    2. Cyclic expansion QUBO formulation (Optimal swap selection)
    """
    
    def __init__(self, m, n, grid_size=10):
        self.m = m
        self.n = n
        self.grid_size = grid_size
        
        # Generate problem data (Fixed for the lifetime of this object)
        self.F = self._generate_flow_matrix()
        self.D = self._generate_distance_matrix()
        
        # Initial Placement placeholder
        self.P = None
        
        print(f"Spectral Cyclic Expansion initialized:")
        print(f"  {m} blocks, {n} locations")
        print(f"  Grid size: {grid_size}x{grid_size}")
        
    def _generate_flow_matrix(self):
        # Generate somewhat sparse, connected graph
        F = np.random.rand(self.m, self.m)
        F = (F + F.T) / 2
        F = (F > 0.7).astype(float) 
        np.fill_diagonal(F, 0)
        
        # Ensure connectivity
        G = nx.from_numpy_array(F)
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            for i in range(len(components)-1):
                u = list(components[i])[0]
                v = list(components[i+1])[0]
                F[u, v] = F[v, u] = 1.0
        return F
    
    def _generate_distance_matrix(self):
        D = np.zeros((self.n, self.n))
        positions = []
        for i in range(self.n):
            x = i % self.grid_size
            y = i // self.grid_size
            positions.append((x, y))
        
        for i in range(self.n):
            for j in range(self.n):
                D[i, j] = abs(positions[i][0] - positions[j][0]) + \
                          abs(positions[i][1] - positions[j][1])
        return D

    def generate_initial_placement(self, noise_std=0.25):
        """
        Scattered Spectral Initialization.
        Public method to allow re-initialization (Restarts).
        """
        # 1. Compute Topology (Fiedler Vector)
        degrees = np.sum(self.F, axis=1)
        L = np.diag(degrees) - self.F
        try:
            vals, vecs = np.linalg.eigh(L)
            fiedler = vecs[:, 1].copy() # 2nd eigenvector
            
            # 2. Add Noise to Block Ordering (Crucial for batch diversity)
            if noise_std > 0:
                noise = np.random.normal(0, noise_std, size=self.m)
                fiedler += noise
                
        except Exception as e:
            print(f"[WARNING] Spectral init failed ({e}), using random.")
            return self._random_permutation_matrix()

        sorted_blocks = np.argsort(fiedler)
        
        # 3. Select Random Locations (Scattered Shape)
        # Each batch run gets a different random subset of locations
        available_locations = np.random.choice(self.n, self.m, replace=False)
        sorted_locations = np.sort(available_locations)
        
        # 4. Map Blocks to Locations
        P = np.zeros((self.m, self.n))
        for i in range(self.m):
            block_idx = sorted_blocks[i]
            loc_idx = sorted_locations[i]
            P[block_idx, loc_idx] = 1
            
        return P

    def _random_permutation_matrix(self):
        perm = np.random.permutation(self.n)[:self.m]
        P = np.zeros((self.m, self.n))
        P[np.arange(self.m), perm] = 1
        return P
    
    def compute_cost(self, P):
        return np.trace(self.F @ P @ self.D @ P.T)
    
    def hybrid_pairing(self, P, k=None):
        """Identifies candidate swaps using 'Stress' + Randomness."""
        if k is None: k = max(2, self.m // 2)
        pairs = set()
        
        # 1. Identify High Stress Pairs
        locations = np.argmax(P, axis=1)
        stress = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(i+1, self.m):
                if self.F[i, j] > 0:
                    loc_i, loc_j = locations[i], locations[j]
                    stress[i, j] = self.D[loc_i, loc_j]
        
        flat_idx = np.argsort(stress.flatten())[::-1]
        for idx in flat_idx:
            if len(pairs) >= k // 2: break
            i, j = idx // self.m, idx % self.m
            if stress[i, j] > 1:
                pairs.add(tuple(sorted((i, j))))
                
        # 2. Fill rest with Random Pairs
        while len(pairs) < k:
            i, j = np.random.choice(self.m, 2, replace=False)
            pairs.add(tuple(sorted((i, j))))
            
        return list(pairs)
    
    def formulate_swap_qubo(self, P, pairs):
        s = len(pairs)
        if s == 0: return np.zeros((0,0)), []
        
        C_tilde = []
        for (i, j) in pairs:
            C = np.eye(self.m)
            C[i,i]=0; C[j,j]=0; C[i,j]=1; C[j,i]=1
            C_tilde.append((C - np.eye(self.m)) @ P)
            
        Q = np.zeros((s, s))
        for i in range(s):
            term1 = np.trace(self.F @ C_tilde[i] @ self.D @ C_tilde[i].T)
            term2 = np.trace(self.F @ C_tilde[i] @ self.D @ P.T)
            term3 = np.trace(self.F @ P @ self.D @ C_tilde[i].T)
            Q[i, i] = term1 + term2 + term3
            
            for j in range(i+1, s):
                val = 2 * np.trace(self.F @ C_tilde[i] @ self.D @ C_tilde[j].T)
                Q[i, j] = val / 2.0
                Q[j, i] = val / 2.0
                
        return Q, C_tilde

    def apply_swaps(self, P, pairs, alpha):
        P_new = P.copy()
        touched = set()
        for idx, (i, j) in enumerate(pairs):
            if alpha[idx] == 1:
                if i in touched or j in touched: continue
                P_new[[i, j], :] = P_new[[j, i], :]
                touched.update([i, j])
        return P_new
    
    def solve_qubo(self, Q, use_sampler=False):
        num_vars = Q.shape[0]
        if num_vars == 0: return np.array([]), 0
        
        Q_dict = {}
        for i in range(num_vars):
            for j in range(i, num_vars):
                if i == j:
                    Q_dict[(i, i)] = Q[i, i]
                else:
                    val = Q[i, j] + Q[j, i]
                    if val != 0:
                        Q_dict[(i, j)] = val
                        
        bqm = BinaryQuadraticModel.from_qubo(Q_dict)
        
        if use_sampler:
            try:
                sampler = EmbeddingComposite(DWaveSampler())
                sampleset = sampler.sample(bqm, num_reads=100, label='Spectral-Cyclic')
            except Exception as e:
                print(f"[WARNING] D-Wave unavailable ({e}), falling back to SA")
                from neal import SimulatedAnnealingSampler
                sampler = SimulatedAnnealingSampler()
                sampleset = sampler.sample(bqm, num_reads=100)
        else:
            from neal import SimulatedAnnealingSampler
            sampler = SimulatedAnnealingSampler()
            sampleset = sampler.sample(bqm, num_reads=100)
            
        best_sample = sampleset.first.sample
        alpha = np.array([best_sample.get(i, 0) for i in range(num_vars)])
        return alpha, sampleset.first.energy

    def optimize(self, max_iterations=50, k=None, use_sampler=False, verbose=True):
        # Ensure we have a placement
        if self.P is None:
            self.P = self.generate_initial_placement()

        start_cost = self.compute_cost(self.P)
        if verbose: print(f"Starting Optimization (Initial Cost: {start_cost:.2f})")
        
        best_P = self.P.copy()
        best_cost = start_cost
        no_improve = 0
        
        history = {
            'iteration': [], 'cost': [], 'num_swaps': [], 'P': []
        }
        
        # Initial record
        history['iteration'].append(-1)
        history['cost'].append(best_cost)
        history['num_swaps'].append(0)
        history['P'].append(best_P.copy())
        
        for it in range(max_iterations):
            pairs = self.hybrid_pairing(self.P, k=k)
            Q, _ = self.formulate_swap_qubo(self.P, pairs)
            
            alpha, _ = self.solve_qubo(Q, use_sampler)

            P_candidate = self.apply_swaps(self.P, pairs, alpha)
            cost_candidate = self.compute_cost(P_candidate)
            cost_curr = self.compute_cost(self.P)
            num_swaps = int(np.sum(alpha))
            
            if cost_candidate < cost_curr - 1e-5:
                self.P = P_candidate
                if verbose: 
                    print(f"Iter {it}: {cost_curr:.2f} -> {cost_candidate:.2f} (Accepted)")
                if cost_candidate < best_cost:
                    best_cost = cost_candidate
                    best_P = P_candidate.copy()
                    no_improve = 0
            else:
                if verbose and num_swaps > 0:
                    print(f"Iter {it}: {cost_curr:.2f} -> {cost_candidate:.2f} (Rejected)")
                no_improve += 1
                
            history['iteration'].append(it)
            history['cost'].append(self.compute_cost(self.P))
            history['num_swaps'].append(num_swaps)
            history['P'].append(self.P.copy())
            
            if no_improve >= 8: 
                if verbose: print("Converged.")
                break
                
        self.P = best_P
        return best_cost, best_P, history

    # =========================================================================
    # VISUALIZATION METHODS
    # =========================================================================
    
    def visualize_placement(self, P=None, title="FPGA Placement", save_path=None):
        if P is None: P = self.P
        locations = np.argmax(P, axis=1)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        for i in range(self.grid_size + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
            ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
            
        for i in range(self.m):
            for j in range(i+1, self.m):
                if self.F[i, j] > 0:
                    loc_i, loc_j = locations[i], locations[j]
                    ax.plot([loc_i % self.grid_size, loc_j % self.grid_size],
                            [loc_i // self.grid_size, loc_j // self.grid_size],
                            color='steelblue', linewidth=1.5, alpha=0.4, zorder=1)
        
        for block in range(self.m):
            loc = locations[block]
            circle = plt.Circle((loc % self.grid_size, loc // self.grid_size), 
                              0.3, color='lightcoral', ec='darkred', zorder=2)
            ax.add_patch(circle)
            ax.text(loc % self.grid_size, loc // self.grid_size, str(block), 
                   ha='center', va='center', color='white', weight='bold', zorder=3)
            
        ax.set_title(title)
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=150); plt.close(fig)
        return fig

    def visualize_convergence(self, history, save_path=None):
        iterations = history['iteration']
        costs = history['cost']
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(iterations, costs, 'b-o', label='Cost')
        ax.set_xlabel('Iteration'); ax.set_ylabel('Wirelength')
        ax.set_title('Convergence Profile (Best Run)')
        ax.grid(True, alpha=0.3)
        
        if save_path: plt.savefig(save_path, dpi=150); plt.close(fig)
        return fig

    def visualize_evolution(self, history, save_path=None, num_snapshots=6):
        placements = history['P']
        if not placements: return
        
        indices = np.linspace(0, len(placements)-1, num_snapshots, dtype=int)
        indices = sorted(list(set(indices)))
        
        cols = min(len(indices), 3)
        rows = (len(indices) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for idx, iter_idx in enumerate(indices):
            if idx >= len(axes): break
            ax = axes[idx]
            P = placements[iter_idx]
            locations = np.argmax(P, axis=1)
            
            ax.set_aspect('equal'); ax.invert_yaxis()
            ax.set_title(f"Iter {history['iteration'][iter_idx]}: Cost {history['cost'][iter_idx]:.1f}")
            
            for i in range(self.grid_size + 1):
                ax.axhline(i-0.5, color='gray', alpha=0.2); ax.axvline(i-0.5, color='gray', alpha=0.2)
            for b in range(self.m):
                loc = locations[b]
                ax.add_patch(plt.Circle((loc%self.grid_size, loc//self.grid_size), 0.3, color='coral'))
            for i in range(self.m):
                for j in range(i+1, self.m):
                    if self.F[i,j]>0:
                        li, lj = locations[i], locations[j]
                        ax.plot([li%self.grid_size, lj%self.grid_size], 
                                [li//self.grid_size, lj//self.grid_size], 'b-', alpha=0.2)
                                
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=150); plt.close(fig)
        return fig


def solve_batch(m, n, grid_size, batch_size=5, max_iters=40):
    """
    Runs the optimizer 'batch_size' times on the SAME problem instance.
    Returns the Best Cost, Best Placement, and the FULL HISTORY of the best run.
    """
    # 1. Create the problem instance ONCE (fixes F and D matrices)
    #    This ensures we are comparing apples to apples.
    opt = SpectralCyclicExpansion(m, n, grid_size)
    
    best_overall_cost = float('inf')
    best_overall_P = None
    best_overall_history = None
    
    print(f"\nStarting Batch Run (Size: {batch_size})...")
    
    for i in range(batch_size):
        print(f"\n--- Run {i+1}/{batch_size} ---")
        
        # 2. Re-initialize the Placement (P) with new randomness
        #    We use noise_std=0.25 to ensure varied starting points
        opt.P = opt.generate_initial_placement(noise_std=2)
        
        # 3. Run Optimization
        cost, P, history = opt.optimize(max_iterations=max_iters, k=8, verbose=False)
        
        print(f"Run {i+1} Result: {cost:.2f}")
        
        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_P = P.copy()
            best_overall_history = history # CAPTURE THE HISTORY
            print(f"  -> New Global Best!")
            
    # Return the optimizer instance too so we can use its viz methods
    return best_overall_cost, best_overall_P, best_overall_history, opt


def main():
    # Run Batch Optimization
    best_cost, best_P, best_history, opt = solve_batch(
        m=50, n=100, grid_size=10, batch_size=10, max_iters=100
    )
    
    print("\n" + "="*70)
    print(f"BATCH COMPLETE. Winning Cost: {best_cost:.2f}")
    print("="*70)
    
    # Generate ALL plots using the 'opt' instance and 'best_history'
    output_dir = './spectral_cyclic_results'
    
    # 1. Final Placement
    opt.visualize_placement(
        P=best_P, 
        title=f"Best of Batch (Cost {best_cost:.2f})", 
        save_path=f"{output_dir}/batch_best_placement.png"
    )
    
    # 2. Convergence Profile (of the winning run)
    opt.visualize_convergence(
        history=best_history, 
        save_path=f"{output_dir}/batch_best_convergence.png"
    )
    
    # 3. Evolution Snapshots (of the winning run)
    opt.visualize_evolution(
        history=best_history, 
        save_path=f"{output_dir}/batch_best_evolution.png",
        num_snapshots=6
    )
    

if __name__ == "__main__":
    main()
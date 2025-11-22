import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from dimod import BinaryQuadraticModel
from dwave.system import DWaveSampler, EmbeddingComposite
import networkx as nx
import os

class FPGAPlacementQUBO:
    """
    Implements the naive QUBO formulation from Section 4.1 of the paper:
    arg min x'Qx + λ||Ax - 1_m||² + μ||Bx - s||²
    where Q = F ⊗ D
    """
    
    def __init__(self, m, n, grid_size=10):
        """
        m: number of functional blocks (facilities)
        n: number of locations on FPGA grid
        grid_size: size of square grid (for distance calculation)
        """
        self.m = m
        self.n = n
        self.grid_size = grid_size
        
        # Generate random flow matrix (connectivity between blocks)
        self.F = self._generate_flow_matrix()
        
        # Generate distance matrix (Manhattan distance on grid)
        self.D = self._generate_distance_matrix()
        
        # Construct constraint matrices
        self.A = np.kron(np.eye(m), np.ones((1, n)))  # I_m ⊗ 1_n^T
        self.B = np.kron(np.ones((1, m)), np.eye(n))  # 1_m^T ⊗ I_n
        
        print(f"Problem size: {m} blocks, {n} locations")
        print(f"QUBO dimension: {m*n + n} variables")
        print(f"Flow matrix shape: {self.F.shape}")
        print(f"Distance matrix shape: {self.D.shape}")
        
    def _generate_flow_matrix(self):
        """Generate random binary flow matrix (which blocks are connected)"""
        F = np.random.rand(self.m, self.m)
        # Make symmetric and binary
        F = (F + F.T) / 2
        F = (F > 0.5).astype(float)
        np.fill_diagonal(F, 0)  # No self-connections
        return F
    
    def _generate_distance_matrix(self):
        """Generate Manhattan distance matrix for grid locations"""
        D = np.zeros((self.n, self.n))
        
        # Calculate position of each location on grid
        positions = []
        for i in range(self.n):
            x = i % self.grid_size
            y = i // self.grid_size
            positions.append((x, y))
        
        # Calculate Manhattan distances
        for i in range(self.n):
            for j in range(self.n):
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                D[i, j] = abs(x1 - x2) + abs(y1 - y2)
        
        return D
    
    def construct_Q_matrix(self):
        """Construct Q = F ⊗ D (Kronecker product)"""
        Q = np.kron(self.F, self.D)
        return Q
    
    def construct_qubo(self, lambda_penalty, mu_penalty):
        """
        Construct the full QUBO matrix including penalty terms
        Variables: [x_1, ..., x_{m*n}, s_1, ..., s_n]
        Total dimension: m*n + n
        """
        mn = self.m * self.n
        total_vars = mn + self.n
        
        # Initialize QUBO matrix
        QUBO = np.zeros((total_vars, total_vars))
        
        # 1. Main objective: x^T Q x
        Q = self.construct_Q_matrix()
        QUBO[:mn, :mn] += Q
        
        # 2. First constraint: λ||Ax - 1_m||²
        QUBO[:mn, :mn] += lambda_penalty * (self.A.T @ self.A)
        linear_term_1 = -2 * lambda_penalty * (np.ones(self.m) @ self.A)
        for i in range(mn):
            QUBO[i, i] += linear_term_1[i]
        
        # 3. Second constraint: μ||Bx - s||²
        QUBO[:mn, :mn] += mu_penalty * (self.B.T @ self.B)
        for i in range(mn):
            for j in range(self.n):
                QUBO[i, mn + j] += -2 * mu_penalty * self.B[j, i]
                QUBO[mn + j, i] += -2 * mu_penalty * self.B[j, i]
        
        for j in range(self.n):
            QUBO[mn + j, mn + j] += mu_penalty
        
        return QUBO
    
    def solve_with_dwave(self, lambda_penalty, mu_penalty, use_sampler=False, num_reads=100):
        """
        Solve QUBO using D-Wave quantum annealer or simulated annealing
        """
        QUBO = self.construct_qubo(lambda_penalty, mu_penalty)
        mn = self.m * self.n
        
        # Convert to BQM format for D-Wave
        bqm = BinaryQuadraticModel.from_numpy_matrix(QUBO)
        
        if use_sampler:
            print("\n[QA] Using D-Wave Quantum Annealer...")
            try:
                sampler = EmbeddingComposite(DWaveSampler())
                sampleset = sampler.sample(bqm, num_reads=num_reads, 
                                          label='FPGA-Placement-QUBO')
            except Exception as e:
                print(f"[ERROR] D-Wave connection failed: {e}")
                print("Falling back to simulated annealing...")
                use_sampler = False
        
        if not use_sampler:
            print("\n[SA] Using Classical Simulated Annealing (Local QA Simulation)...")
            from neal import SimulatedAnnealingSampler
            sampler = SimulatedAnnealingSampler()
            sampleset = sampler.sample(bqm, num_reads=num_reads)
        
        # Get best solution
        best_sample = sampleset.first.sample
        energy = sampleset.first.energy
        
        # Extract placement vector x and slack variables s
        x_vec = np.array([best_sample[i] for i in range(mn)])
        s_vec = np.array([best_sample[mn + i] for i in range(self.n)])
        
        # Reshape to matrix form
        X = x_vec.reshape(self.m, self.n)
        
        return X, s_vec, energy, sampleset
    
    def evaluate_solution(self, X, s_vec):
        """Evaluate the quality of a solution"""
        x_vec = X.flatten()
        
        # Compute main objective (wirelength)
        Q = self.construct_Q_matrix()
        objective = x_vec @ Q @ x_vec
        
        # Check constraint violations
        constraint1_violation = np.linalg.norm(self.A @ x_vec - np.ones(self.m))
        constraint2_violation = np.linalg.norm(self.B @ x_vec - s_vec)
        
        # Check if it's a valid permutation
        row_sums = X.sum(axis=1)
        col_sums = X.sum(axis=0)
        
        is_valid = (np.allclose(row_sums, 1) and 
                   np.all(col_sums <= 1))
        
        # Detailed constraint analysis
        blocks_not_placed_once = np.where(~np.isclose(row_sums, 1))[0]
        locations_with_overlap = np.where(col_sums > 1)[0]
        
        return {
            'objective': objective,
            'constraint1_violation': constraint1_violation,
            'constraint2_violation': constraint2_violation,
            'row_sums': row_sums,
            'col_sums': col_sums,
            'is_valid_placement': is_valid,
            's_values': s_vec,
            'blocks_not_placed_once': blocks_not_placed_once,
            'locations_with_overlap': locations_with_overlap
        }
    
    def visualize_placement(self, X, title="FPGA Placement", save_path=None):
        """Visualize the placement on the grid with connectivity overlay"""
        placement = {}
        overlaps = {}  # Track multiple blocks at same location
        
        for i in range(self.m):
            locations = np.where(X[i, :] == 1)[0]
            if len(locations) > 0:
                loc = locations[0]
                if loc in overlaps:
                    overlaps[loc].append(i)
                else:
                    if loc in placement.values():
                        existing_block = [k for k, v in placement.items() if v == loc][0]
                        overlaps[loc] = [existing_block, i]
                        del placement[existing_block]
                    else:
                        placement[i] = loc
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Draw grid
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        # Draw grid lines
        for i in range(self.grid_size + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
            ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
        
        # Draw connections
        all_blocks = {**placement}
        for loc, blocks in overlaps.items():
            for block in blocks:
                all_blocks[block] = loc
        
        for i in range(self.m):
            for j in range(i+1, self.m):
                if self.F[i, j] > 0 and i in all_blocks and j in all_blocks:
                    loc_i = all_blocks[i]
                    loc_j = all_blocks[j]
                    x_i = loc_i % self.grid_size
                    y_i = loc_i // self.grid_size
                    x_j = loc_j % self.grid_size
                    y_j = loc_j // self.grid_size
                    
                    ax.plot([x_i, x_j], [y_i, y_j], 
                           color='steelblue', linewidth=2, alpha=0.6, zorder=1)
        
        # Draw blocks
        for block, loc in placement.items():
            x = loc % self.grid_size
            y = loc // self.grid_size
            
            circle = plt.Circle((x, y), 0.35, color='lightcoral', 
                              ec='darkred', linewidth=2, zorder=2)
            ax.add_patch(circle)
            
            ax.text(x, y, str(block), ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white', zorder=3)
        
        # Draw overlaps
        for loc, blocks in overlaps.items():
            x = loc % self.grid_size
            y = loc // self.grid_size
            
            circle = plt.Circle((x, y), 0.4, color='red', 
                              ec='darkred', linewidth=3, zorder=2)
            ax.add_patch(circle)
            
            label = ','.join(map(str, blocks))
            ax.text(x, y, label, ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white', zorder=3)
            
            ax.annotate('OVERLAP!', xy=(x, y), xytext=(x+0.6, y+0.6),
                       fontsize=8, color='red', fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        ax.set_xlabel('X coordinate', fontsize=12)
        ax.set_ylabel('Y coordinate', fontsize=12)
        
        if overlaps:
            title += f" [WARNING] {len(overlaps)} OVERLAPS DETECTED!"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        ax.set_xticks(range(self.grid_size))
        ax.set_yticks(range(self.grid_size))
        
        # Add grid coordinates
        occupied_locs = set(placement.values()) | set(overlaps.keys())
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                loc_id = i * self.grid_size + j
                if loc_id not in occupied_locs:
                    ax.text(j, i, f'{loc_id}', ha='center', va='center',
                           fontsize=7, color='gray', alpha=0.4, zorder=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[SAVE] Visualization saved to: {save_path}")
            plt.close(fig)
        
        return fig
    
    def visualize_objective_progression(self, sweep_history, save_path=None):
        """Create a detailed standalone visualization of objective progression"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        objectives = np.array(sweep_history['objective'])
        valid = np.array(sweep_history['valid'])
        phases = sweep_history['phase']
        iterations = np.arange(len(objectives))
        
        # Color map for phases
        phase_colors = {
            'phase1': 'red',
            'stage2.1': 'orange',
            'stage2.2': 'yellow',
            'stage2.3': 'green'
        }
        colors = [phase_colors.get(p, 'gray') for p in phases]
        
        # Plot valid solutions
        valid_mask = valid == True
        invalid_mask = valid == False
        
        ax.scatter(iterations[valid_mask], objectives[valid_mask], 
                  c=np.array(colors)[valid_mask], s=100, alpha=0.8, 
                  edgecolors='darkgreen', linewidths=2, label='Valid', zorder=3)
        ax.scatter(iterations[invalid_mask], objectives[invalid_mask], 
                  c=np.array(colors)[invalid_mask], s=100, alpha=0.6, 
                  marker='x', linewidths=2, label='Invalid', zorder=2)
        
        # Connect with line
        ax.plot(iterations, objectives, 'k-', alpha=0.3, linewidth=1.5, zorder=1)
        
        # Highlight best solution
        if np.any(valid_mask):
            best_idx = np.where(valid_mask)[0][np.argmin(objectives[valid_mask])]
            ax.scatter(best_idx, objectives[best_idx], 
                      c='purple', s=400, marker='*', 
                      edgecolors='black', linewidths=3, 
                      label='Best Solution', zorder=10)
            ax.annotate(f'Best: {objectives[best_idx]:.2f}', 
                       xy=(best_idx, objectives[best_idx]),
                       xytext=(best_idx + 2, objectives[best_idx] * 1.1),
                       fontsize=12, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', lw=2, color='purple'))
        
        # Mark phase boundaries
        phase_changes = [0]
        current_phase = phases[0]
        for i, phase in enumerate(phases[1:], 1):
            if phase != current_phase:
                phase_changes.append(i)
                current_phase = phase
        phase_changes.append(len(phases))
        
        for i in range(len(phase_changes) - 1):
            start_idx = phase_changes[i]
            end_idx = phase_changes[i + 1]
            phase_name = phases[start_idx]
            mid_idx = (start_idx + end_idx) // 2
            
            ax.axvspan(start_idx, end_idx, alpha=0.1, 
                      color=phase_colors.get(phase_name, 'gray'))
            
            # Label phase
            phase_labels = {
                'phase1': 'Phase 1:\nFeasibility',
                'stage2.1': 'Stage 2.1:\nAggressive\nReduction',
                'stage2.2': 'Stage 2.2:\nRecovery',
                'stage2.3': 'Stage 2.3:\nLinear Sweep'
            }
            ax.text(mid_idx, ax.get_ylim()[1] * 0.95, 
                   phase_labels.get(phase_name, phase_name),
                   ha='center', va='top', fontsize=10, 
                   bbox=dict(boxstyle='round', facecolor=phase_colors.get(phase_name, 'gray'), alpha=0.3))
        
        ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
        ax.set_ylabel('Objective (Wirelength)', fontsize=14, fontweight='bold')
        ax.set_title('Objective Function Progression During Penalty Sweep', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[SAVE] Objective progression saved to: {save_path}")
            plt.close(fig)
        
        return fig
    
    def visualize_placement_progression(self, sweep_history, save_path=None, num_snapshots=6):
        """Visualize how placement evolves during the sweep"""
        placements = sweep_history['X']
        phases = sweep_history['phase']
        objectives = sweep_history['objective']
        valid = sweep_history['valid']
        lambdas = sweep_history['lambda']
        mus = sweep_history['mu']
        
        # Select key snapshots to show
        total_iters = len(placements)
        
        # Strategy: show first, last, best, and evenly spaced others
        snapshot_indices = []
        
        # First iteration
        snapshot_indices.append(0)
        
        # Best valid solution
        valid_array = np.array(valid)
        if np.any(valid_array):
            best_idx = np.where(valid_array)[0][np.argmin(np.array(objectives)[valid_array])]
            snapshot_indices.append(best_idx)
        
        # Last iteration
        snapshot_indices.append(total_iters - 1)
        
        # Fill remaining with evenly spaced
        remaining = num_snapshots - len(snapshot_indices)
        if remaining > 0:
            step = total_iters // (remaining + 1)
            for i in range(1, remaining + 1):
                idx = min(i * step, total_iters - 1)
                if idx not in snapshot_indices:
                    snapshot_indices.append(idx)
        
        # Sort and limit
        snapshot_indices = sorted(list(set(snapshot_indices)))[:num_snapshots]
        
        # Create subplot grid
        n_cols = 3
        n_rows = (len(snapshot_indices) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for plot_idx, iter_idx in enumerate(snapshot_indices):
            ax = axes[plot_idx]
            X = placements[iter_idx]
            
            # Extract placement
            placement = {}
            overlaps = {}
            for i in range(self.m):
                locations = np.where(X[i, :] == 1)[0]
                if len(locations) > 0:
                    loc = locations[0]
                    if loc in overlaps:
                        overlaps[loc].append(i)
                    else:
                        if loc in placement.values():
                            existing_block = [k for k, v in placement.items() if v == loc][0]
                            overlaps[loc] = [existing_block, i]
                            del placement[existing_block]
                        else:
                            placement[i] = loc
            
            # Draw grid
            ax.set_xlim(-0.5, self.grid_size - 0.5)
            ax.set_ylim(-0.5, self.grid_size - 0.5)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            
            for i in range(self.grid_size + 1):
                ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
                ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
            
            # Draw connections
            all_blocks = {**placement}
            for loc, blocks in overlaps.items():
                for block in blocks:
                    all_blocks[block] = loc
            
            for i in range(self.m):
                for j in range(i+1, self.m):
                    if self.F[i, j] > 0 and i in all_blocks and j in all_blocks:
                        loc_i = all_blocks[i]
                        loc_j = all_blocks[j]
                        x_i = loc_i % self.grid_size
                        y_i = loc_i // self.grid_size
                        x_j = loc_j % self.grid_size
                        y_j = loc_j // self.grid_size
                        ax.plot([x_i, x_j], [y_i, y_j], 
                               color='steelblue', linewidth=1.5, alpha=0.4, zorder=1)
            
            # Draw blocks
            for block, loc in placement.items():
                x = loc % self.grid_size
                y = loc // self.grid_size
                circle = plt.Circle((x, y), 0.3, color='lightcoral', 
                                  ec='darkred', linewidth=1.5, zorder=2)
                ax.add_patch(circle)
                ax.text(x, y, str(block), ha='center', va='center',
                       fontsize=9, fontweight='bold', color='white', zorder=3)
            
            # Draw overlaps
            for loc, blocks in overlaps.items():
                x = loc % self.grid_size
                y = loc // self.grid_size
                circle = plt.Circle((x, y), 0.35, color='red', 
                                  ec='darkred', linewidth=2, zorder=2)
                ax.add_patch(circle)
                label = ','.join(map(str, blocks))
                ax.text(x, y, label, ha='center', va='center',
                       fontsize=8, fontweight='bold', color='white', zorder=3)
            
            # Title with info
            phase_name = phases[iter_idx]
            is_valid = valid[iter_idx]
            obj_val = objectives[iter_idx]
            lambda_val = lambdas[iter_idx]
            mu_val = mus[iter_idx]
            
            title = f"Iter {iter_idx}: {phase_name}\n"
            title += f"lambda={lambda_val:.0f}, mu={mu_val:.0f}\n"
            title += f"Obj={obj_val:.1f} {'[OK]' if is_valid else '[X]'}"
            
            if iter_idx == 0:
                title = "[START] " + title
            elif iter_idx == total_iters - 1:
                title = "[FINAL] " + title
            elif iter_idx == best_idx and is_valid:
                title = "[BEST] " + title
            
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Hide unused subplots
        for idx in range(len(snapshot_indices), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Placement Evolution During Penalty Sweep', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[SAVE] Placement progression saved to: {save_path}")
            plt.close(fig)
        
        return fig
    
    def visualize_sweep_progress(self, sweep_history, save_path=None):
        """Visualize the penalty sweep algorithm progression"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Extract data
        lambdas = np.array(sweep_history['lambda'])
        mus = np.array(sweep_history['mu'])
        objectives = np.array(sweep_history['objective'])
        c1_viols = np.array(sweep_history['c1_violation'])
        c2_viols = np.array(sweep_history['c2_violation'])
        valid = np.array(sweep_history['valid'])
        phases = sweep_history['phase']
        
        iterations = np.arange(len(lambdas))
        
        # Color map for phases
        phase_colors = {
            'phase1': 'red',
            'stage2.1': 'orange',
            'stage2.2': 'yellow',
            'stage2.3': 'green'
        }
        colors = [phase_colors.get(p, 'gray') for p in phases]
        
        # Plot 1: Lambda over iterations
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(iterations, lambdas, c=colors, s=50, alpha=0.7)
        ax1.plot(iterations, lambdas, 'k-', alpha=0.3, linewidth=1)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Lambda Penalty')
        ax1.set_title('Lambda Evolution')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Mu over iterations
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(iterations, mus, c=colors, s=50, alpha=0.7)
        ax2.plot(iterations, mus, 'k-', alpha=0.3, linewidth=1)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Mu Penalty')
        ax2.set_title('Mu Evolution')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Plot 3: Objective over iterations
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.scatter(iterations[valid], objectives[valid], c='green', s=50, alpha=0.7, label='Valid')
        ax3.scatter(iterations[~valid], objectives[~valid], c='red', s=50, alpha=0.7, label='Invalid')
        ax3.plot(iterations, objectives, 'k-', alpha=0.3, linewidth=1)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Objective (Wirelength)')
        ax3.set_title('Objective Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Constraint violations
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.semilogy(iterations, c1_viols, 'b-', alpha=0.7, label='C1 Violation')
        ax4.semilogy(iterations, c2_viols, 'r-', alpha=0.7, label='C2 Violation')
        ax4.axhline(0.1, color='k', linestyle='--', alpha=0.5, label='Threshold')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Constraint Violation')
        ax4.set_title('Constraint Violations')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Lambda vs Mu trajectory
        ax5 = fig.add_subplot(gs[2, 0])
        for i in range(len(lambdas)-1):
            ax5.plot(lambdas[i:i+2], mus[i:i+2], c=colors[i], alpha=0.7, linewidth=2)
        ax5.scatter(lambdas[valid], mus[valid], c='green', s=100, marker='o', 
                   edgecolors='black', linewidths=2, alpha=0.7, label='Valid', zorder=5)
        ax5.scatter(lambdas[~valid], mus[~valid], c='red', s=100, marker='x', 
                   linewidths=2, alpha=0.7, label='Invalid', zorder=5)
        ax5.scatter(lambdas[0], mus[0], c='blue', s=200, marker='s', 
                   edgecolors='black', linewidths=2, label='Start', zorder=10)
        ax5.scatter(lambdas[-1], mus[-1], c='purple', s=200, marker='*', 
                   edgecolors='black', linewidths=2, label='Final', zorder=10)
        ax5.set_xlabel('Lambda')
        ax5.set_ylabel('Mu')
        ax5.set_title('Penalty Space Trajectory')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_xscale('log')
        ax5.set_yscale('log')
        
        # Plot 6: Phase legend and statistics
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        
        # Phase legend
        legend_text = "Phase Legend:\n"
        legend_text += "  Phase 1 (red): Initial feasibility search\n"
        legend_text += "  Stage 2.1 (orange): Aggressive reduction\n"
        legend_text += "  Stage 2.2 (yellow): Constraint recovery\n"
        legend_text += "  Stage 2.3 (green): Linear sweep optimization\n\n"
        
        # Statistics
        best_valid_idx = np.where(valid)[0]
        if len(best_valid_idx) > 0:
            best_idx = best_valid_idx[np.argmin(objectives[best_valid_idx])]
            legend_text += f"Best Solution:\n"
            legend_text += f"  Iteration: {best_idx}\n"
            legend_text += f"  Lambda: {lambdas[best_idx]:.1f}\n"
            legend_text += f"  Mu: {mus[best_idx]:.1f}\n"
            legend_text += f"  Objective: {objectives[best_idx]:.2f}\n"
            legend_text += f"  C1 Violation: {c1_viols[best_idx]:.4f}\n"
            legend_text += f"  C2 Violation: {c2_viols[best_idx]:.4f}\n\n"
        
        legend_text += f"Total Iterations: {len(lambdas)}\n"
        legend_text += f"Valid Solutions: {np.sum(valid)}/{len(valid)}"
        
        ax6.text(0.1, 0.5, legend_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.suptitle('Penalty Sweep Algorithm Progression', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[SAVE] Progression visualization saved to: {save_path}")
            plt.close(fig)
        
        return fig
    
    def penalty_sweep(self, use_sampler=False, num_reads=100, verbose=True):
        """
        Automatically find optimal lambda and mu using two-phase approach
        """
        print("\n" + "=" * 70)
        print("AUTOMATIC PENALTY SWEEP")
        print("=" * 70)
        
        # Track all evaluations for visualization
        sweep_history = {
            'lambda': [],
            'mu': [],
            'objective': [],
            'c1_violation': [],
            'c2_violation': [],
            'valid': [],
            'phase': [],
            'X': [],  # Store placement matrices
            's': []   # Store slack variables
        }
        
        lambda_val = self.n ** 2
        mu_val = self.n ** 2
        
        print(f"Initial penalties: lambda={lambda_val}, mu={mu_val}")
        
        # Phase 1: Feasibility
        print("\n[PHASE 1] Finding Feasible Solution")
        print("-" * 70)
        
        phase1_iterations = 0
        max_phase1_iterations = 20
        
        while phase1_iterations < max_phase1_iterations:
            phase1_iterations += 1
            
            if verbose:
                print(f"\nIteration {phase1_iterations}: Testing lambda={lambda_val:.1f}, mu={mu_val:.1f}")
            
            X, s_vec, energy, sampleset = self.solve_with_dwave(
                lambda_val, mu_val, use_sampler=use_sampler, num_reads=num_reads
            )
            
            results = self.evaluate_solution(X, s_vec)
            
            c1_violation = results['constraint1_violation'] > 0.1
            c2_violation = results['constraint2_violation'] > 0.1
            
            # Record history
            sweep_history['lambda'].append(lambda_val)
            sweep_history['mu'].append(mu_val)
            sweep_history['objective'].append(results['objective'])
            sweep_history['c1_violation'].append(results['constraint1_violation'])
            sweep_history['c2_violation'].append(results['constraint2_violation'])
            sweep_history['valid'].append(not (c1_violation or c2_violation))
            sweep_history['phase'].append('phase1')
            sweep_history['X'].append(X.copy())
            sweep_history['s'].append(s_vec.copy())
            
            if verbose:
                print(f"   C1 violation: {results['constraint1_violation']:.4f} {'[X]' if c1_violation else '[OK]'}")
                print(f"   C2 violation: {results['constraint2_violation']:.4f} {'[X]' if c2_violation else '[OK]'}")
                print(f"   Objective: {results['objective']:.2f}")
            
            if not c1_violation and not c2_violation:
                print(f"\n[OK] Feasible solution found!")
                break
            
            if c1_violation and c2_violation:
                lambda_val *= 2
                mu_val *= 2
                if verbose:
                    print(f"   -> Both violated: doubling both penalties")
            elif c1_violation and not c2_violation:
                lambda_val *= 2
                if verbose:
                    print(f"   -> C1 violated: doubling lambda")
            elif c2_violation and not c1_violation:
                mu_val *= 2
                if verbose:
                    print(f"   -> C2 violated: doubling mu")
        
        if phase1_iterations >= max_phase1_iterations:
            print(f"\n[WARNING] Phase 1 reached max iterations. Using last values.")
        
        # Store Phase 1 solution
        old_loss = results['objective']
        valid_lambda = lambda_val
        valid_mu = mu_val
        valid_loss = old_loss
        X_valid = X.copy()
        s_valid = s_vec.copy()
        phase1_lambda = valid_lambda
        phase1_mu = valid_mu
        phase1_loss = valid_loss
        
        print(f"\n[PHASE 1 COMPLETE]")
        print(f"   Feasible penalties: lambda={valid_lambda:.1f}, mu={valid_mu:.1f}")
        print(f"   Objective (wirelength): {valid_loss:.2f}")
        
        # Phase 2: Stage 2.1 - Aggressive reduction
        print("\n[STAGE 2.1] Aggressive Penalty Reduction")
        print("-" * 70)
        
        step = self.m * self.n
        stage1_iterations = 0
        max_stage1_iterations = 20
        
        while stage1_iterations < max_stage1_iterations:
            stage1_iterations += 1
            
            lambda_trial = valid_lambda - step
            mu_trial = valid_mu - step
            
            lambda_trial = max(1.0, lambda_trial)
            mu_trial = max(1.0, mu_trial)
            
            if verbose:
                print(f"\nIteration {stage1_iterations}: Testing lambda={lambda_trial:.1f}, mu={mu_trial:.1f} (step={step:.1f})")
            
            X, s_vec, energy, sampleset = self.solve_with_dwave(
                lambda_trial, mu_trial, use_sampler=use_sampler, num_reads=num_reads
            )
            
            results = self.evaluate_solution(X, s_vec)
            new_loss = results['objective']
            
            c1_violation = results['constraint1_violation'] > 0.1
            c2_violation = results['constraint2_violation'] > 0.1
            
            # Record history
            sweep_history['lambda'].append(lambda_trial)
            sweep_history['mu'].append(mu_trial)
            sweep_history['objective'].append(new_loss)
            sweep_history['c1_violation'].append(results['constraint1_violation'])
            sweep_history['c2_violation'].append(results['constraint2_violation'])
            sweep_history['valid'].append(not (c1_violation or c2_violation))
            sweep_history['phase'].append('stage2.1')
            sweep_history['X'].append(X.copy())
            sweep_history['s'].append(s_vec.copy())
            
            if verbose:
                print(f"   C1: {results['constraint1_violation']:.4f} {'[X]' if c1_violation else '[OK]'}, "
                      f"C2: {results['constraint2_violation']:.4f} {'[X]' if c2_violation else '[OK]'}, "
                      f"Obj: {new_loss:.2f}")
            
            if c1_violation or c2_violation:
                if verbose:
                    print(f"   [X] Constraint violated! Breaking aggressive reduction.")
                break
            
            valid_lambda = lambda_trial
            valid_mu = mu_trial
            valid_loss = new_loss
            X_valid = X.copy()
            s_valid = s_vec.copy()
            
            step = step * 2
            
            if verbose:
                print(f"   [OK] Accepted! Doubling step to {step:.1f}")
        
        print(f"\n[STAGE 2.1 COMPLETE]: lambda={valid_lambda:.1f}, mu={valid_mu:.1f}, obj={valid_loss:.2f}")
        
        # Stage 2.2: Recovery
        print("\n[STAGE 2.2] Constraint Recovery")
        print("-" * 70)
        
        stage2_iterations = 0
        max_stage2_iterations = 20
        recovery_step = step / 2
        
        if verbose:
            print(f"Initial recovery step: {recovery_step:.1f}")
        
        while (c1_violation or c2_violation) and stage2_iterations < max_stage2_iterations:
            stage2_iterations += 1
            
            if c1_violation and c2_violation:
                lambda_trial = valid_lambda + recovery_step
                mu_trial = valid_mu + recovery_step
                if verbose:
                    print(f"\nIteration {stage2_iterations}: Both violated. Increasing both by {recovery_step:.1f}")
            elif c1_violation:
                lambda_trial = valid_lambda + recovery_step
                mu_trial = valid_mu
                if verbose:
                    print(f"\nIteration {stage2_iterations}: C1 violated. Increasing lambda by {recovery_step:.1f}")
            else:
                lambda_trial = valid_lambda
                mu_trial = valid_mu + recovery_step
                if verbose:
                    print(f"\nIteration {stage2_iterations}: C2 violated. Increasing mu by {recovery_step:.1f}")
            
            if verbose:
                print(f"   Testing lambda={lambda_trial:.1f}, mu={mu_trial:.1f}")
            
            X, s_vec, energy, sampleset = self.solve_with_dwave(
                lambda_trial, mu_trial, use_sampler=use_sampler, num_reads=num_reads
            )
            
            results = self.evaluate_solution(X, s_vec)
            new_loss = results['objective']
            
            c1_violation = results['constraint1_violation'] > 0.1
            c2_violation = results['constraint2_violation'] > 0.1
            
            # Record history
            sweep_history['lambda'].append(lambda_trial)
            sweep_history['mu'].append(mu_trial)
            sweep_history['objective'].append(new_loss)
            sweep_history['c1_violation'].append(results['constraint1_violation'])
            sweep_history['c2_violation'].append(results['constraint2_violation'])
            sweep_history['valid'].append(not (c1_violation or c2_violation))
            sweep_history['phase'].append('stage2.2')
            sweep_history['X'].append(X.copy())
            sweep_history['s'].append(s_vec.copy())
            
            if verbose:
                print(f"   C1: {results['constraint1_violation']:.4f} {'[X]' if c1_violation else '[OK]'}, "
                      f"C2: {results['constraint2_violation']:.4f} {'[X]' if c2_violation else '[OK]'}, "
                      f"Obj: {new_loss:.2f}")
            
            if not c1_violation and not c2_violation:
                valid_lambda = lambda_trial
                valid_mu = mu_trial
                valid_loss = new_loss
                X_valid = X.copy()
                s_valid = s_vec.copy()
                if verbose:
                    print(f"   [OK] Constraints satisfied! Recovery complete.")
                break
            else:
                recovery_step = recovery_step * 2
                if verbose:
                    print(f"   [X] Still violated. Doubling recovery step to {recovery_step:.1f}")
        
        if stage2_iterations >= max_stage2_iterations:
            print(f"\n[WARNING] Stage 2.2 reached max iterations.")
        else:
            print(f"\n[STAGE 2.2 COMPLETE]: lambda={valid_lambda:.1f}, mu={valid_mu:.1f}, obj={valid_loss:.2f}")
        
        recovery_lambda = valid_lambda
        recovery_mu = valid_mu
        recovery_loss = valid_loss
        
        # Stage 2.3: Linear sweep
        print("\n[STAGE 2.3] Linear Sweep Between Phase 1 and Recovery Point")
        print("-" * 70)
        print(f"   Phase 1 point: lambda={phase1_lambda:.1f}, mu={phase1_mu:.1f}, obj={phase1_loss:.2f}")
        print(f"   Recovery point: lambda={recovery_lambda:.1f}, mu={recovery_mu:.1f}, obj={recovery_loss:.2f}")
        
        num_sweep_points = 100
        
        best_sweep_lambda = valid_lambda
        best_sweep_mu = valid_mu
        best_sweep_loss = valid_loss
        best_sweep_X = X_valid.copy()
        best_sweep_s = s_valid.copy()
        
        for i in range(num_sweep_points + 1):
            alpha = i / num_sweep_points
            
            lambda_trial = recovery_lambda + alpha * (phase1_lambda - recovery_lambda)
            mu_trial = recovery_mu + alpha * (phase1_mu - recovery_mu)
            
            if verbose:
                print(f"\nSweep point {i}/{num_sweep_points} (alpha={alpha:.2f}): lambda={lambda_trial:.1f}, mu={mu_trial:.1f}")
            
            X, s_vec, energy, sampleset = self.solve_with_dwave(
                lambda_trial, mu_trial, use_sampler=use_sampler, num_reads=num_reads
            )
            
            results = self.evaluate_solution(X, s_vec)
            new_loss = results['objective']
            
            c1_violation = results['constraint1_violation'] > 0.1
            c2_violation = results['constraint2_violation'] > 0.1
            
            # Record history
            sweep_history['lambda'].append(lambda_trial)
            sweep_history['mu'].append(mu_trial)
            sweep_history['objective'].append(new_loss)
            sweep_history['c1_violation'].append(results['constraint1_violation'])
            sweep_history['c2_violation'].append(results['constraint2_violation'])
            sweep_history['valid'].append(not (c1_violation or c2_violation))
            sweep_history['phase'].append('stage2.3')
            sweep_history['X'].append(X.copy())
            sweep_history['s'].append(s_vec.copy())
            
            if verbose:
                print(f"   C1: {results['constraint1_violation']:.4f} {'[X]' if c1_violation else '[OK]'}, "
                      f"C2: {results['constraint2_violation']:.4f} {'[X]' if c2_violation else '[OK]'}, "
                      f"Obj: {new_loss:.2f}")
            
            if not c1_violation and not c2_violation:
                if new_loss < best_sweep_loss:
                    best_sweep_lambda = lambda_trial
                    best_sweep_mu = mu_trial
                    best_sweep_loss = new_loss
                    best_sweep_X = X.copy()
                    best_sweep_s = s_vec.copy()
                    if verbose:
                        print(f"   [OK] New best! obj={new_loss:.2f}")
            else:
                if verbose:
                    print(f"   [X] Constraint violated, skipping")
        
        valid_lambda = best_sweep_lambda
        valid_mu = best_sweep_mu
        valid_loss = best_sweep_loss
        X_valid = best_sweep_X
        s_valid = best_sweep_s
        
        print(f"\n[STAGE 2.3 COMPLETE]")
        print(f"   Best from sweep: lambda={valid_lambda:.1f}, mu={valid_mu:.1f}, obj={valid_loss:.2f}")
        print(f"\n[PHASE 2 COMPLETE] Total evaluations: {phase1_iterations + stage1_iterations + stage2_iterations + num_sweep_points + 1}")
        
        final_lambda = valid_lambda
        final_mu = valid_mu
        final_loss = valid_loss
        
        print(f"\n[SUCCESS] Optimal solution found!")
        print("\n" + "=" * 70)
        print("SWEEP RESULTS")
        print("=" * 70)
        print(f"Optimal penalties: lambda={final_lambda:.1f}, mu={final_mu:.1f}")
        print(f"Objective (wirelength): {final_loss:.2f}")
        print("=" * 70)
        
        return final_lambda, final_mu, X_valid, s_valid, final_loss, sweep_history


# Main function
def main():
    print("=" * 60)
    print("FPGA PLACEMENT using QUBO Formulation (Section 4.1)")
    print("=" * 60)
    
    m = 10
    n = 25
    
    placer = FPGAPlacementQUBO(m, n, grid_size=10)
    
    print("\n[INFO] Flow Matrix (connectivity between blocks):")
    print(placer.F)
    
    print("\n" + "=" * 60)
    print("PARAMETER TUNING GUIDE")
    print("=" * 60)
    print("Start with small values and increase gradually:")
    print("  lambda (lambda): Controls 'each block placed exactly once' constraint")
    print("  mu (mu):     Controls 'at most one block per location' constraint")
    print("\nSuggested starting values: lambda=100, mu=100")
    print("If constraints violated, increase penalties by 5-10x")
    print("If valid but poor objective, try decreasing slightly")
    print("=" * 60)
    
    mode = input("\nSelect mode:\n  1) Auto sweep (find optimal lambda,mu automatically)\n  2) Manual single run\n  3) Batch mode (try multiple values)\nChoice [default=1]: ") or "1"
    
    if mode == "1":
        use_dwave = input("Use D-Wave quantum annealer? (y/n) [default=n]: ").lower() == 'y'
        verbose = input("Verbose output? (y/n) [default=y]: ").lower() != 'n'
        
        optimal_lambda, optimal_mu, X, s_vec, final_loss, sweep_history = placer.penalty_sweep(
            use_sampler=use_dwave,
            num_reads=100,
            verbose=verbose
        )
        
        results = placer.evaluate_solution(X, s_vec)
        
        print("\n" + "=" * 60)
        print("FINAL SOLUTION")
        print("=" * 60)
        print(f"[INFO] Valid Placement: {results['is_valid_placement']}")
        print(f"[INFO] Objective (wirelength): {results['objective']:.2f}")
        print(f"[INFO] Constraint 1 violation: {results['constraint1_violation']:.4f}")
        print(f"[INFO] Constraint 2 violation: {results['constraint2_violation']:.4f}")
        
        if len(results['blocks_not_placed_once']) > 0:
            print(f"\n[ERROR] Blocks not placed exactly once: {results['blocks_not_placed_once'].tolist()}")
        
        if len(results['locations_with_overlap']) > 0:
            print(f"\n[ERROR] Locations with overlapping blocks: {results['locations_with_overlap'].tolist()}")
        
        output_dir = "fpga_placement_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save placement visualization
        output_file = os.path.join(output_dir, f"placement_optimal_lambda{optimal_lambda:.0f}_mu{optimal_mu:.0f}.png")
        placer.visualize_placement(
            X, 
            title=f"Optimal: lambda={optimal_lambda:.1f}, mu={optimal_mu:.1f}, Valid={results['is_valid_placement']}",
            save_path=output_file
        )
        
        # Save progression visualization
        progress_file = os.path.join(output_dir, f"sweep_progression_lambda{optimal_lambda:.0f}_mu{optimal_mu:.0f}.png")
        placer.visualize_sweep_progress(sweep_history, save_path=progress_file)
        
        # Save objective progression (standalone)
        objective_file = os.path.join(output_dir, f"objective_progression_lambda{optimal_lambda:.0f}_mu{optimal_mu:.0f}.png")
        placer.visualize_objective_progression(sweep_history, save_path=objective_file)
        
        # Save placement progression
        placement_prog_file = os.path.join(output_dir, f"placement_evolution_lambda{optimal_lambda:.0f}_mu{optimal_mu:.0f}.png")
        placer.visualize_placement_progression(sweep_history, save_path=placement_prog_file, num_snapshots=6)
        
        print(f"\n[SAVE] Results saved to: {output_dir}/")
        
    elif mode == "3":
        param_combinations = [
            (10, 10), (50, 50), (100, 100), (200, 200), (500, 500), (1000, 1000)
        ]
        print(f"\n[INFO] Running {len(param_combinations)} parameter combinations...")
        print("   (Higher penalties enforce constraints more strongly)")
        
        results_summary = []
        for lambda_penalty, mu_penalty in param_combinations:
            print(f"\n{'='*60}")
            print(f"Testing lambda={lambda_penalty}, mu={mu_penalty}")
            print('='*60)
            
            X, s_vec, energy, sampleset = placer.solve_with_dwave(
                lambda_penalty, mu_penalty, 
                use_sampler=False,
                num_reads=100
            )
            
            results = placer.evaluate_solution(X, s_vec)
            results_summary.append({
                'lambda': lambda_penalty,
                'mu': mu_penalty,
                'valid': results['is_valid_placement'],
                'objective': results['objective'],
                'c1_violation': results['constraint1_violation'],
                'c2_violation': results['constraint2_violation'],
                'energy': energy
            })
            
            output_dir = "fpga_placement_results"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"placement_lambda{lambda_penalty}_mu{mu_penalty}.png")
            placer.visualize_placement(X, 
                title=f"lambda={lambda_penalty}, mu={mu_penalty}, Valid={results['is_valid_placement']}",
                save_path=output_file)
        
        print("\n" + "=" * 80)
        print("BATCH RESULTS SUMMARY")
        print("=" * 80)
        print(f"{'lambda':>6} {'mu':>6} {'Valid':>7} {'Objective':>12} {'C1 Viol':>10} {'C2 Viol':>10} {'Energy':>10}")
        print("-" * 80)
        for r in results_summary:
            print(f"{r['lambda']:>6} {r['mu']:>6} {'[OK]' if r['valid'] else '[X]':>7} "
                  f"{r['objective']:>12.2f} {r['c1_violation']:>10.4f} "
                  f"{r['c2_violation']:>10.4f} {r['energy']:>10.2f}")
        print("=" * 80)
        
        valid_results = [r for r in results_summary if r['valid']]
        if valid_results:
            best = min(valid_results, key=lambda x: x['objective'])
            print(f"\n[BEST] Best valid solution: lambda={best['lambda']}, mu={best['mu']}, "
                  f"objective={best['objective']:.2f}")
        else:
            print(f"\n[WARNING] No valid solutions found. Try increasing penalties!")
        
        print(f"\n[SAVE] All results saved to: fpga_placement_results/")
        
    else:
        lambda_penalty = float(input("\nEnter lambda (lambda) penalty [default=100]: ") or "100")
        mu_penalty = float(input("Enter mu (mu) penalty [default=100]: ") or "100")
        use_dwave = input("Use D-Wave quantum annealer? (y/n) [default=n]: ").lower() == 'y'
        
        print(f"\n[INFO] Solving with lambda={lambda_penalty}, mu={mu_penalty}...")
        
        X, s_vec, energy, sampleset = placer.solve_with_dwave(
            lambda_penalty, mu_penalty, 
            use_sampler=use_dwave,
            num_reads=100
        )
        
        results = placer.evaluate_solution(X, s_vec)
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"\n[INFO] Valid Placement: {results['is_valid_placement']}")
        print(f"[INFO] Objective (wirelength): {results['objective']:.2f}")
        print(f"[INFO] Constraint 1 violation: {results['constraint1_violation']:.4f}")
        print(f"[INFO] Constraint 2 violation: {results['constraint2_violation']:.4f}")
        print(f"[INFO] QUBO Energy: {energy:.2f}")
        
        print(f"\n[INFO] Row sums (should be all 1s): {results['row_sums']}")
        print(f"[INFO] Col sums (should be <= 1): {results['col_sums']}")
        
        if len(results['blocks_not_placed_once']) > 0:
            print(f"\n[ERROR] Blocks not placed exactly once: {results['blocks_not_placed_once'].tolist()}")
            for block_id in results['blocks_not_placed_once']:
                count = results['row_sums'][block_id]
                print(f"   Block {block_id}: placed {count} times")
        
        if len(results['locations_with_overlap']) > 0:
            print(f"\n[ERROR] Locations with overlapping blocks: {results['locations_with_overlap'].tolist()}")
            for loc_id in results['locations_with_overlap']:
                count = results['col_sums'][loc_id]
                blocks_here = np.where(X[:, loc_id] == 1)[0]
                print(f"   Location {loc_id}: {int(count)} blocks -> {blocks_here.tolist()}")
        
        print(f"\n[INFO] Slack variables s: {results['s_values']}")
        
        print("\n[INFO] Placement Matrix X (blocks x locations):")
        print("  (1 means block i is placed at location j)")
        print(X)
        
        output_dir = "fpga_placement_results"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"placement_lambda{lambda_penalty}_mu{mu_penalty}.png")
        
        placer.visualize_placement(
            X, 
            title=f"lambda={lambda_penalty}, mu={mu_penalty}, Valid={results['is_valid_placement']}",
            save_path=output_file
        )
        
        print(f"\n[SAVE] Results saved to: {output_dir}/")
        print(f"   - {output_file}")
        
        print("\n" + "=" * 60)
        print("TUNING SUGGESTIONS")
        print("=" * 60)
        if results['is_valid_placement']:
            print("[OK] Placement is valid! Try reducing penalties to minimize objective.")
        else:
            print("[ERROR] Invalid placement detected!")
            if results['constraint1_violation'] > 0.1:
                print(f"[WARNING] Blocks not placed exactly once!")
                print(f"    -> Increase lambda (currently {lambda_penalty}) to {lambda_penalty * 5}")
            if results['constraint2_violation'] > 0.1:
                print(f"[WARNING] Multiple blocks at same location!")
                print(f"    -> Increase mu (currently {mu_penalty}) to {mu_penalty * 5}")
            if len(results['locations_with_overlap']) > 0:
                print(f"\n[TIP] The penalties need to be MUCH larger than the objective.")
                print(f"    Current objective: {results['objective']:.2f}")
                print(f"    Try lambda={max(500, int(results['objective'] * 10))}, mu={max(500, int(results['objective'] * 10))}")
        print("=" * 60)


if __name__ == "__main__":
    main()
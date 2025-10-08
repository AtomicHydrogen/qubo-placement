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
        # Expand: λ(x^T A^T A x - 2·1_m^T A x + 1_m^T 1_m)
        # Since 1_m^T 1_m is constant, we ignore it
        QUBO[:mn, :mn] += lambda_penalty * (self.A.T @ self.A)
        # Linear term: -2λ·1_m^T A x (convert to quadratic by adding to diagonal)
        linear_term_1 = -2 * lambda_penalty * (np.ones(self.m) @ self.A)
        for i in range(mn):
            QUBO[i, i] += linear_term_1[i]
        
        # 3. Second constraint: μ||Bx - s||²
        # Expand: μ(x^T B^T B x - 2s^T B x + s^T s)
        QUBO[:mn, :mn] += mu_penalty * (self.B.T @ self.B)
        # Cross terms: -2μ B x · s
        for i in range(mn):
            for j in range(self.n):
                QUBO[i, mn + j] += -2 * mu_penalty * self.B[j, i]
                QUBO[mn + j, i] += -2 * mu_penalty * self.B[j, i]
        
        # s^T s term (diagonal)
        for j in range(self.n):
            QUBO[mn + j, mn + j] += mu_penalty
        
        return QUBO
    
    def solve_with_dwave(self, lambda_penalty, mu_penalty, use_sampler=False, num_reads=100):
        """
        Solve QUBO using D-Wave quantum annealer or simulated annealing
        
        use_sampler: If True, use actual D-Wave quantum annealer
                     If False, use classical simulated annealing
        """
        QUBO = self.construct_qubo(lambda_penalty, mu_penalty)
        mn = self.m * self.n
        
        # Convert to BQM format for D-Wave
        bqm = BinaryQuadraticModel.from_numpy_matrix(QUBO)
        
        if use_sampler:
            print("\n🔮 Using D-Wave Quantum Annealer...")
            try:
                sampler = EmbeddingComposite(DWaveSampler())
                sampleset = sampler.sample(bqm, num_reads=num_reads, 
                                          label='FPGA-Placement-QUBO')
            except Exception as e:
                print(f"❌ D-Wave connection failed: {e}")
                print("Falling back to simulated annealing...")
                use_sampler = False
        
        if not use_sampler:
            print("\n🖥️  Using Classical Simulated Annealing (Local QA Simulation)...")
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
    
    def analyze_solution_diversity(self, sampleset, top_k=10):
        """Analyze diversity of solutions from quantum annealing"""
        mn = self.m * self.n
        solutions = []
        energies = []
        
        for sample, energy in sampleset.data(['sample', 'energy']):
            x_vec = np.array([sample[i] for i in range(mn)])
            X = x_vec.reshape(self.m, self.n)
            solutions.append(X)
            energies.append(energy)
        
        # Take top k solutions
        top_k = min(top_k, len(solutions))
        top_solutions = solutions[:top_k]
        top_energies = energies[:top_k]
        
        print(f"\n📊 Solution Diversity Analysis (top {top_k} solutions):")
        print(f"   Energy range: [{min(top_energies):.2f}, {max(top_energies):.2f}]")
        print(f"   Energy spread: {max(top_energies) - min(top_energies):.2f}")
        
        # Check uniqueness
        unique_solutions = []
        for sol in top_solutions:
            is_unique = True
            for unique_sol in unique_solutions:
                if np.array_equal(sol, unique_sol):
                    is_unique = False
                    break
            if is_unique:
                unique_solutions.append(sol)
        
        print(f"   Unique solutions: {len(unique_solutions)}/{top_k}")
        
        return top_solutions, top_energies
    
    def penalty_sweep(self, use_sampler=False, num_reads=100, verbose=True):
        """
        Automatically find optimal lambda and mu using two-phase approach:
        Phase 1: Increase penalties until feasible
        Phase 2: Decrease penalties while maintaining feasibility
        """
        print("\n" + "=" * 70)
        print("AUTOMATIC PENALTY SWEEP")
        print("=" * 70)
        
        # Initialize
        lambda_val = self.n ** 2  # n^2 = 100
        mu_val = self.n ** 2
        
        print(f"Initial penalties: λ={lambda_val}, μ={mu_val}")
        
        # Phase 1: Feasibility
        print("\n📍 PHASE 1: Finding Feasible Solution")
        print("-" * 70)
        
        phase1_iterations = 0
        max_phase1_iterations = 20  # Safety limit
        
        while phase1_iterations < max_phase1_iterations:
            phase1_iterations += 1
            
            if verbose:
                print(f"\nIteration {phase1_iterations}: Testing λ={lambda_val:.1f}, μ={mu_val:.1f}")
            
            # Solve with current penalties
            X, s_vec, energy, sampleset = self.solve_with_dwave(
                lambda_val, mu_val, use_sampler=use_sampler, num_reads=num_reads
            )
            
            results = self.evaluate_solution(X, s_vec)
            
            c1_violation = results['constraint1_violation'] > 0.1
            c2_violation = results['constraint2_violation'] > 0.1
            
            if verbose:
                print(f"   C1 violation: {results['constraint1_violation']:.4f} {'❌' if c1_violation else '✅'}")
                print(f"   C2 violation: {results['constraint2_violation']:.4f} {'❌' if c2_violation else '✅'}")
                print(f"   Objective: {results['objective']:.2f}")
            
            # Check termination condition
            if not c1_violation and not c2_violation:
                print(f"\n✅ Feasible solution found!")
                break
            
            # Update penalties based on violations
            if c1_violation and c2_violation:
                lambda_val *= 2
                mu_val *= 2
                if verbose:
                    print(f"   → Both violated: doubling both penalties")
            elif c1_violation and not c2_violation:
                lambda_val *= 2
                if verbose:
                    print(f"   → C1 violated: doubling λ")
            elif c2_violation and not c1_violation:
                mu_val *= 2
                if verbose:
                    print(f"   → C2 violated: doubling μ")
        
        if phase1_iterations >= max_phase1_iterations:
            print(f"\n⚠️  Phase 1 reached max iterations. Using last values.")
        
        # Store feasible solution
        old_loss = results['objective']
        valid_lambda = lambda_val
        valid_mu = mu_val
        valid_loss = old_loss
        X_valid = X.copy()  # Store the Phase 1 solution
        s_valid = s_vec.copy()
        
        print(f"\n📊 Phase 1 Complete:")
        print(f"   Feasible penalties: λ={valid_lambda:.1f}, μ={valid_mu:.1f}")
        print(f"   Objective (wirelength): {valid_loss:.2f}")
        
        # Phase 2: Stepwise Reduction
        print("\n📍 PHASE 2: Optimizing While Maintaining Feasibility")
        print("-" * 70)
        
        # Store Phase 1 solution as reference
        phase1_lambda = valid_lambda
        phase1_mu = valid_mu
        phase1_loss = valid_loss
        
        # Stage 1: Aggressive reduction (double step each time)
        print("\n🔻 Stage 2.1: Aggressive Penalty Reduction")
        print("-" * 70)
        
        step = self.m * self.n  # Initial step = mn
        stage1_iterations = 0
        max_stage1_iterations = 20
        
        while stage1_iterations < max_stage1_iterations:
            stage1_iterations += 1
            
            # Try reducing penalties
            lambda_trial = valid_lambda - step
            mu_trial = valid_mu - step
            
            # Ensure non-negative
            lambda_trial = max(1.0, lambda_trial)
            mu_trial = max(1.0, mu_trial)
            
            if verbose:
                print(f"\nIteration {stage1_iterations}: Testing λ={lambda_trial:.1f}, μ={mu_trial:.1f} (step={step:.1f})")
            
            # Solve with trial penalties
            X, s_vec, energy, sampleset = self.solve_with_dwave(
                lambda_trial, mu_trial, use_sampler=use_sampler, num_reads=num_reads
            )
            
            results = self.evaluate_solution(X, s_vec)
            new_loss = results['objective']
            
            c1_violation = results['constraint1_violation'] > 0.1
            c2_violation = results['constraint2_violation'] > 0.1
            
            if verbose:
                print(f"   C1: {results['constraint1_violation']:.4f} {'❌' if c1_violation else '✅'}, "
                      f"C2: {results['constraint2_violation']:.4f} {'❌' if c2_violation else '✅'}, "
                      f"Obj: {new_loss:.2f}")
            
            # Check for constraint violation
            if c1_violation or c2_violation:
                if verbose:
                    print(f"   ❌ Constraint violated! Breaking aggressive reduction.")
                break
            
            # No violation - accept and double the step
            valid_lambda = lambda_trial
            valid_mu = mu_trial
            valid_loss = new_loss
            X_valid = X.copy()
            s_valid = s_vec.copy()
            
            step = step * 2  # Double for next iteration
            
            if verbose:
                print(f"   ✅ Accepted! Doubling step to {step:.1f}")
        
        print(f"\n📊 Stage 2.1 Complete: λ={valid_lambda:.1f}, μ={valid_mu:.1f}, obj={valid_loss:.2f}")
        
        # Stage 2: Recovery (increase violated constraints)
        print("\n🔺 Stage 2.2: Constraint Recovery")
        print("-" * 70)
        
        # Determine which constraints failed
        stage2_iterations = 0
        max_stage2_iterations = 20
        recovery_step = step / 2  # Start with half the step that caused violation
        
        if verbose:
            print(f"Initial recovery step: {recovery_step:.1f}")
        
        while (c1_violation or c2_violation) and stage2_iterations < max_stage2_iterations:
            stage2_iterations += 1
            
            # Increase only the violated constraints
            if c1_violation and c2_violation:
                lambda_trial = valid_lambda + recovery_step
                mu_trial = valid_mu + recovery_step
                if verbose:
                    print(f"\nIteration {stage2_iterations}: Both violated. Increasing both by {recovery_step:.1f}")
            elif c1_violation:
                lambda_trial = valid_lambda + recovery_step
                mu_trial = valid_mu
                if verbose:
                    print(f"\nIteration {stage2_iterations}: C1 violated. Increasing λ by {recovery_step:.1f}")
            else:  # c2_violation
                lambda_trial = valid_lambda
                mu_trial = valid_mu + recovery_step
                if verbose:
                    print(f"\nIteration {stage2_iterations}: C2 violated. Increasing μ by {recovery_step:.1f}")
            
            if verbose:
                print(f"   Testing λ={lambda_trial:.1f}, μ={mu_trial:.1f}")
            
            # Solve with trial penalties
            X, s_vec, energy, sampleset = self.solve_with_dwave(
                lambda_trial, mu_trial, use_sampler=use_sampler, num_reads=num_reads
            )
            
            results = self.evaluate_solution(X, s_vec)
            new_loss = results['objective']
            
            c1_violation = results['constraint1_violation'] > 0.1
            c2_violation = results['constraint2_violation'] > 0.1
            
            if verbose:
                print(f"   C1: {results['constraint1_violation']:.4f} {'❌' if c1_violation else '✅'}, "
                      f"C2: {results['constraint2_violation']:.4f} {'❌' if c2_violation else '✅'}, "
                      f"Obj: {new_loss:.2f}")
            
            if not c1_violation and not c2_violation:
                # Recovered! Accept this solution
                valid_lambda = lambda_trial
                valid_mu = mu_trial
                valid_loss = new_loss
                X_valid = X.copy()
                s_valid = s_vec.copy()
                if verbose:
                    print(f"   ✅ Constraints satisfied! Recovery complete.")
                break
            else:
                # Still violated, double the recovery step
                recovery_step = recovery_step * 2
                if verbose:
                    print(f"   ❌ Still violated. Doubling recovery step to {recovery_step:.1f}")
        
        if stage2_iterations >= max_stage2_iterations:
            print(f"\n⚠️  Stage 2.2 reached max iterations.")
        else:
            print(f"\n📊 Stage 2.2 Complete: λ={valid_lambda:.1f}, μ={valid_mu:.1f}, obj={valid_loss:.2f}")
        
        # Store recovery point
        recovery_lambda = valid_lambda
        recovery_mu = valid_mu
        recovery_loss = valid_loss
        
        # Stage 3: Linear sweep between Phase 1 and Recovery point
        print("\n🎯 Stage 2.3: Linear Sweep Between Phase 1 and Recovery Point")
        print("-" * 70)
        print(f"   Phase 1 point: λ={phase1_lambda:.1f}, μ={phase1_mu:.1f}, obj={phase1_loss:.2f}")
        print(f"   Recovery point: λ={recovery_lambda:.1f}, μ={recovery_mu:.1f}, obj={recovery_loss:.2f}")
        
        # Number of points to sample in the sweep
        num_sweep_points = 10
        
        best_sweep_lambda = valid_lambda
        best_sweep_mu = valid_mu
        best_sweep_loss = valid_loss
        best_sweep_X = X_valid.copy()
        best_sweep_s = s_valid.copy()
        
        for i in range(num_sweep_points + 1):
            alpha = i / num_sweep_points  # 0 to 1
            
            # Linear interpolation between recovery (alpha=0) and phase1 (alpha=1)
            lambda_trial = recovery_lambda + alpha * (phase1_lambda - recovery_lambda)
            mu_trial = recovery_mu + alpha * (phase1_mu - recovery_mu)
            
            if verbose:
                print(f"\nSweep point {i}/{num_sweep_points} (α={alpha:.2f}): λ={lambda_trial:.1f}, μ={mu_trial:.1f}")
            
            # Solve with trial penalties
            X, s_vec, energy, sampleset = self.solve_with_dwave(
                lambda_trial, mu_trial, use_sampler=use_sampler, num_reads=num_reads
            )
            
            results = self.evaluate_solution(X, s_vec)
            new_loss = results['objective']
            
            c1_violation = results['constraint1_violation'] > 0.1
            c2_violation = results['constraint2_violation'] > 0.1
            
            if verbose:
                print(f"   C1: {results['constraint1_violation']:.4f} {'❌' if c1_violation else '✅'}, "
                      f"C2: {results['constraint2_violation']:.4f} {'❌' if c2_violation else '✅'}, "
                      f"Obj: {new_loss:.2f}")
            
            # Check if this is better and valid
            if not c1_violation and not c2_violation:
                if new_loss < best_sweep_loss:
                    best_sweep_lambda = lambda_trial
                    best_sweep_mu = mu_trial
                    best_sweep_loss = new_loss
                    best_sweep_X = X.copy()
                    best_sweep_s = s_vec.copy()
                    if verbose:
                        print(f"   ✅ New best! obj={new_loss:.2f}")
            else:
                if verbose:
                    print(f"   ❌ Constraint violated, skipping")
        
        # Use best from sweep
        valid_lambda = best_sweep_lambda
        valid_mu = best_sweep_mu
        valid_loss = best_sweep_loss
        X_valid = best_sweep_X
        s_valid = best_sweep_s
        
        print(f"\n📊 Stage 2.3 Complete")
        print(f"   Best from sweep: λ={valid_lambda:.1f}, μ={valid_mu:.1f}, obj={valid_loss:.2f}")
        print(f"\n📊 Phase 2 Complete (Total: {stage1_iterations + stage2_iterations + num_sweep_points + 1} evaluations)")
        
        # Final solution is the best valid one we found
        final_lambda = valid_lambda
        final_mu = valid_mu
        final_loss = valid_loss
        
        print(f"\n✅ Optimal solution found!")
        print("\n" + "=" * 70)
        print("SWEEP RESULTS")
        print("=" * 70)
        print(f"Optimal penalties: λ={final_lambda:.1f}, μ={final_mu:.1f}")
        print(f"Objective (wirelength): {final_loss:.2f}")
        print("=" * 70)
        
        # Return the stored valid solution (don't resolve)
        return final_lambda, final_mu, X_valid, s_valid, final_loss
    
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
                        # Found overlap - move existing block to overlaps
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
        ax.invert_yaxis()  # Origin at top-left
        
        # Draw grid lines
        for i in range(self.grid_size + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
            ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
        
        # Draw connections first (so they appear behind blocks)
        all_blocks = {**placement}
        for loc, blocks in overlaps.items():
            for block in blocks:
                all_blocks[block] = loc
        
        for i in range(self.m):
            for j in range(i+1, self.m):
                if self.F[i, j] > 0 and i in all_blocks and j in all_blocks:
                    # Get positions
                    loc_i = all_blocks[i]
                    loc_j = all_blocks[j]
                    x_i = loc_i % self.grid_size
                    y_i = loc_i // self.grid_size
                    x_j = loc_j % self.grid_size
                    y_j = loc_j // self.grid_size
                    
                    # Draw connection line
                    ax.plot([x_i, x_j], [y_i, y_j], 
                           color='steelblue', linewidth=2, alpha=0.6, zorder=1)
        
        # Draw blocks as circles with labels (normal placements)
        for block, loc in placement.items():
            x = loc % self.grid_size
            y = loc // self.grid_size
            
            # Draw circle
            circle = plt.Circle((x, y), 0.35, color='lightcoral', 
                              ec='darkred', linewidth=2, zorder=2)
            ax.add_patch(circle)
            
            # Add label
            ax.text(x, y, str(block), ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white', zorder=3)
        
        # Draw overlapping blocks (WARNING: multiple blocks at same location!)
        for loc, blocks in overlaps.items():
            x = loc % self.grid_size
            y = loc // self.grid_size
            
            # Draw larger circle in red to indicate problem
            circle = plt.Circle((x, y), 0.4, color='red', 
                              ec='darkred', linewidth=3, zorder=2)
            ax.add_patch(circle)
            
            # Add label showing all blocks at this location
            label = ','.join(map(str, blocks))
            ax.text(x, y, label, ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white', zorder=3)
            
            # Add warning annotation
            ax.annotate('OVERLAP!', xy=(x, y), xytext=(x+0.6, y+0.6),
                       fontsize=8, color='red', fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        # Labels and title
        ax.set_xlabel('X coordinate', fontsize=12)
        ax.set_ylabel('Y coordinate', fontsize=12)
        
        # Update title to show overlaps
        if overlaps:
            title += f" ⚠️ {len(overlaps)} OVERLAPS DETECTED!"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Set integer ticks
        ax.set_xticks(range(self.grid_size))
        ax.set_yticks(range(self.grid_size))
        
        # Add grid coordinates at each position
        occupied_locs = set(placement.values()) | set(overlaps.keys())
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                loc_id = i * self.grid_size + j
                # Only show if no block placed here
                if loc_id not in occupied_locs:
                    ax.text(j, i, f'{loc_id}', ha='center', va='center',
                           fontsize=7, color='gray', alpha=0.4, zorder=0)
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Visualization saved to: {save_path}")
            plt.close(fig)
        
        return fig


# Example usage and parameter tuning interface
def main():
    print("=" * 60)
    print("FPGA PLACEMENT using QUBO Formulation (Section 4.1)")
    print("=" * 60)
    
    # Problem setup
    m = 10   # Number of blocks to place
    n = 25  # Number of grid locations (10x10)
    
    placer = FPGAPlacementQUBO(m, n, grid_size=10)
    
    # Display flow matrix
    print("\n📊 Flow Matrix (connectivity between blocks):")
    print(placer.F)
    
    # Parameter tuning guide
    print("\n" + "=" * 60)
    print("PARAMETER TUNING GUIDE")
    print("=" * 60)
    print("Start with small values and increase gradually:")
    print("  λ (lambda): Controls 'each block placed exactly once' constraint")
    print("  μ (mu):     Controls 'at most one block per location' constraint")
    print("\nSuggested starting values: λ=100, μ=100")
    print("If constraints violated, increase penalties by 5-10x")
    print("If valid but poor objective, try decreasing slightly")
    print("=" * 60)
    
    # Interactive or batch mode
    mode = input("\nSelect mode:\n  1) Auto sweep (find optimal λ,μ automatically)\n  2) Manual single run\n  3) Batch mode (try multiple values)\nChoice [default=1]: ") or "1"
    
    if mode == "1":
        # Automatic penalty sweep
        use_dwave = input("Use D-Wave quantum annealer? (y/n) [default=n]: ").lower() == 'y'
        verbose = input("Verbose output? (y/n) [default=y]: ").lower() != 'n'
        
        optimal_lambda, optimal_mu, X, s_vec, final_loss = placer.penalty_sweep(
            use_sampler=use_dwave,
            num_reads=100,
            verbose=verbose
        )
        
        # Evaluate final solution
        results = placer.evaluate_solution(X, s_vec)
        
        print("\n" + "=" * 60)
        print("FINAL SOLUTION")
        print("=" * 60)
        print(f"✅ Valid Placement: {results['is_valid_placement']}")
        print(f"📏 Objective (wirelength): {results['objective']:.2f}")
        print(f"⚠️  Constraint 1 violation: {results['constraint1_violation']:.4f}")
        print(f"⚠️  Constraint 2 violation: {results['constraint2_violation']:.4f}")
        
        if len(results['blocks_not_placed_once']) > 0:
            print(f"\n❌ Blocks not placed exactly once: {results['blocks_not_placed_once'].tolist()}")
        
        if len(results['locations_with_overlap']) > 0:
            print(f"\n❌ Locations with overlapping blocks: {results['locations_with_overlap'].tolist()}")
        
        # Visualize
        output_dir = "fpga_placement_results"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"placement_optimal_lambda{optimal_lambda:.0f}_mu{optimal_mu:.0f}.png")
        
        placer.visualize_placement(
            X, 
            title=f"Optimal: λ={optimal_lambda:.1f}, μ={optimal_mu:.1f}, Valid={results['is_valid_placement']}",
            save_path=output_file
        )
        
        print(f"\n📁 Results saved to: {output_dir}/")
        
    elif mode == "3":
        # Run multiple parameter combinations
        param_combinations = [
            (10, 10), (50, 50), (100, 100), (200, 200), (500, 500), (1000, 1000)
        ]
        print(f"\n🔄 Running {len(param_combinations)} parameter combinations...")
        print("   (Higher penalties enforce constraints more strongly)")
        
        results_summary = []
        for lambda_penalty, mu_penalty in param_combinations:
            print(f"\n{'='*60}")
            print(f"Testing λ={lambda_penalty}, μ={mu_penalty}")
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
            
            # Save visualization
            output_dir = "fpga_placement_results"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"placement_lambda{lambda_penalty}_mu{mu_penalty}.png")
            placer.visualize_placement(X, 
                title=f"λ={lambda_penalty}, μ={mu_penalty}, Valid={results['is_valid_placement']}",
                save_path=output_file)
        
        # Print summary table
        print("\n" + "=" * 80)
        print("BATCH RESULTS SUMMARY")
        print("=" * 80)
        print(f"{'λ':>6} {'μ':>6} {'Valid':>7} {'Objective':>12} {'C1 Viol':>10} {'C2 Viol':>10} {'Energy':>10}")
        print("-" * 80)
        for r in results_summary:
            print(f"{r['lambda']:>6} {r['mu']:>6} {'✓' if r['valid'] else '✗':>7} "
                  f"{r['objective']:>12.2f} {r['c1_violation']:>10.4f} "
                  f"{r['c2_violation']:>10.4f} {r['energy']:>10.2f}")
        print("=" * 80)
        
        # Find best valid solution
        valid_results = [r for r in results_summary if r['valid']]
        if valid_results:
            best = min(valid_results, key=lambda x: x['objective'])
            print(f"\n🏆 Best valid solution: λ={best['lambda']}, μ={best['mu']}, "
                  f"objective={best['objective']:.2f}")
        else:
            print("\n⚠️  No valid solutions found. Try increasing penalties!")
        
        print(f"\n📁 All results saved to: fpga_placement_results/")
        
    else:  # mode == "2"
        # Interactive single run
        lambda_penalty = float(input("\nEnter λ (lambda) penalty [default=10]: ") or "10")
        mu_penalty = float(input("Enter μ (mu) penalty [default=10]: ") or "10")
        use_dwave = input("Use D-Wave quantum annealer? (y/n) [default=n]: ").lower() == 'y'
        
        print(f"\n🎯 Solving with λ={lambda_penalty}, μ={mu_penalty}...")
        
        # Solve
        X, s_vec, energy, sampleset = placer.solve_with_dwave(
            lambda_penalty, mu_penalty, 
            use_sampler=use_dwave,
            num_reads=100
        )
        
        # Analyze solution diversity (QA gives multiple solutions)
        top_solutions, top_energies = placer.analyze_solution_diversity(sampleset, top_k=10)
        
        # Evaluate
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        results = placer.evaluate_solution(X, s_vec)
        
        print(f"\n✅ Valid Placement: {results['is_valid_placement']}")
        print(f"📏 Objective (wirelength): {results['objective']:.2f}")
        print(f"⚠️  Constraint 1 violation: {results['constraint1_violation']:.4f}")
        print(f"⚠️  Constraint 2 violation: {results['constraint2_violation']:.4f}")
        print(f"🔋 QUBO Energy: {energy:.2f}")
        
        print(f"\n📊 Row sums (should be all 1s): {results['row_sums']}")
        print(f"📊 Col sums (should be ≤ 1): {results['col_sums']}")
        
        # Detailed violation analysis
        if len(results['blocks_not_placed_once']) > 0:
            print(f"\n❌ Blocks not placed exactly once: {results['blocks_not_placed_once'].tolist()}")
            for block_id in results['blocks_not_placed_once']:
                count = results['row_sums'][block_id]
                print(f"   Block {block_id}: placed {count} times")
        
        if len(results['locations_with_overlap']) > 0:
            print(f"\n❌ Locations with overlapping blocks: {results['locations_with_overlap'].tolist()}")
            for loc_id in results['locations_with_overlap']:
                count = results['col_sums'][loc_id]
                blocks_here = np.where(X[:, loc_id] == 1)[0]
                print(f"   Location {loc_id}: {int(count)} blocks → {blocks_here.tolist()}")
        
        print(f"\n📊 Slack variables s: {results['s_values']}")
        
        # Show placement matrix
        print("\n📋 Placement Matrix X (blocks × locations):")
        print("  (1 means block i is placed at location j)")
        print(X)
        
        # Visualize
        output_dir = "fpga_placement_results"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"placement_lambda{lambda_penalty}_mu{mu_penalty}.png")
        
        placer.visualize_placement(
            X, 
            title=f"λ={lambda_penalty}, μ={mu_penalty}, Valid={results['is_valid_placement']}",
            save_path=output_file
        )
        
        print(f"\n📁 Results saved to: {output_dir}/")
        print(f"   - {output_file}")
        
        # Tuning suggestions
        print("\n" + "=" * 60)
        print("TUNING SUGGESTIONS")
        print("=" * 60)
        if results['is_valid_placement']:
            print("✅ Placement is valid! Try reducing penalties to minimize objective.")
        else:
            print("❌ Invalid placement detected!")
            if results['constraint1_violation'] > 0.1:
                print(f"⚠️  Blocks not placed exactly once!")
                print(f"    → Increase λ (currently {lambda_penalty}) to {lambda_penalty * 5}")
            if results['constraint2_violation'] > 0.1:
                print(f"⚠️  Multiple blocks at same location!")
                print(f"    → Increase μ (currently {mu_penalty}) to {mu_penalty * 5}")
            if len(results['locations_with_overlap']) > 0:
                print(f"\n💡 TIP: The penalties need to be MUCH larger than the objective.")
                print(f"    Current objective: {results['objective']:.2f}")
                print(f"    Try λ={max(500, int(results['objective'] * 10))}, μ={max(500, int(results['objective'] * 10))}")
        print("=" * 60)


if __name__ == "__main__":
    main()
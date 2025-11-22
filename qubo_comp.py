"""
Robust FPGA Placement Comparison Orchestrator.
Compares:
1. Classical Baseline (Greedy + Local Search)
2. Spectral Cyclic Expansion (Hybrid Swap + Spectral Init)

Designed to work with the refactored 'qubo_spectral.py' and 'classical_baseline.py'.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import shutil

# --- Import Solvers Safely ---
try:
    from qubo_spectral import SpectralCyclicExpansion
except ImportError:
    print("[ERROR] 'qubo_spectral.py' not found. Spectral solver unavailable.")
    SpectralCyclicExpansion = None

try:
    from classical_baseline import ClassicalQUBOSolver
except ImportError:
    print("[ERROR] 'classical_baseline.py' not found. Classical solver unavailable.")
    ClassicalQUBOSolver = None

def extract_placement_from_X(X, m):
    """Convert placement matrix X (m x n) to dictionary {block: loc}."""
    placement = {}
    for block_id in range(m):
        locations = np.where(X[block_id, :] == 1)[0]
        if len(locations) > 0:
            placement[block_id] = int(locations[0])
    return placement

def run_fair_comparison(m=15, n=25, grid_size=10, spectral_batch_size=5):
    
    output_dir = "comparison_results"
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print("\n" + "="*70)
    print(f"FPGA PLACEMENT COMPARISON ({m} blocks, {n} locations)")
    print("="*70)

    # =========================================================================
    # 1. GENERATE SHARED PROBLEM INSTANCE
    # =========================================================================
    # We use the Classical Solver's constructor logic (or just numpy) to make
    # the problem matrices. Let's just use numpy here for transparency.
    print("[1] Generating Shared Problem Instance...")
    
    # Connectivity (Flow Matrix F)
    F_shared = np.random.rand(m, m)
    F_shared = (F_shared + F_shared.T) / 2
    F_shared = (F_shared > 0.7).astype(float)
    np.fill_diagonal(F_shared, 0)
    
    # Distances (Distance Matrix D)
    D_shared = np.zeros((n, n))
    positions = [(i % grid_size, i // grid_size) for i in range(n)]
    for i in range(n):
        for j in range(n):
            D_shared[i, j] = abs(positions[i][0] - positions[j][0]) + \
                             abs(positions[i][1] - positions[j][1])
                             
    print(f"    -> Generated F ({m}x{m}) and D ({n}x{n}) matrices.")
    
    results = {}

    # =========================================================================
    # 2. RUN CLASSICAL BASELINE
    # =========================================================================
    if ClassicalQUBOSolver:
        print("\n[2] Running Classical Baseline (Greedy + Local Search)...")
        classical = ClassicalQUBOSolver(m, n, grid_size, F_shared, D_shared)
        
        start_time = time.time()
        # Use heuristic penalties if naive solver isn't running
        c_res = classical.solve(lambda_penalty=n**2, mu_penalty=n**2, method='greedy+local')
        elapsed = time.time() - start_time
        
        results['classical'] = {
            'placement': c_res['placement'],
            'wirelength': c_res['wirelength'],
            'time': elapsed,
            'solver': classical
        }
        print(f"    -> Result: Cost {c_res['wirelength']:.2f} | Time {elapsed:.2f}s")
        
        # Save Classical Plot
        classical.visualize_placement(
            c_res['placement'], c_res['wirelength'], 
            title="Classical Baseline", 
            save_path=f"{output_dir}/classical_placement.png"
        )

    # =========================================================================
    # 3. RUN SPECTRAL CYCLIC EXPANSION (BATCHED)
    # =========================================================================
    if SpectralCyclicExpansion:
        print(f"\n[3] Running Spectral Cyclic Expansion (Batch Size: {spectral_batch_size})...")
        
        # Initialize Solver
        spectral = SpectralCyclicExpansion(m, n, grid_size)
        
        # --- CRITICAL STEP: INJECT SHARED MATRICES ---
        spectral.F = F_shared
        spectral.D = D_shared
        
        best_cost = float('inf')
        best_P = None
        best_history = None
        
        start_time = time.time()
        
        # Mini-Batch Loop (Simulates "Solve Batch" from standalone script)
        for i in range(spectral_batch_size):
            # Re-initialize placement for the NEW shared F/D
            spectral.P = spectral.generate_initial_placement(noise_std=0.25)
            
            # Run Optimization
            cost, P, history = spectral.optimize(max_iterations=40, k=8, verbose=False)
            
            if cost < best_cost:
                best_cost = cost
                best_P = P.copy()
                best_history = history
                # Update solver state for viz
                spectral.P = P
                
        elapsed = time.time() - start_time
        
        results['spectral'] = {
            'placement': extract_placement_from_X(best_P, m),
            'wirelength': best_cost,
            'time': elapsed,
            'solver': spectral,
            'history': best_history
        }
        print(f"    -> Result: Cost {best_cost:.2f} | Time {elapsed:.2f}s")
        
        # Save Spectral Plots
        spectral.visualize_placement(
            best_P, title=f"Spectral Cyclic (Batch Best)", 
            save_path=f"{output_dir}/spectral_placement.png"
        )
        spectral.visualize_convergence(
            best_history, save_path=f"{output_dir}/spectral_convergence.png"
        )
        spectral.visualize_evolution(
            best_history, save_path=f"{output_dir}/spectral_evolution.png"
        )

    # =========================================================================
    # 4. GENERATE COMPARISON REPORT
    # =========================================================================
    print("\n[4] Generating Comparison Report...")
    
    # Bar Chart
    plt.figure(figsize=(10, 6))
    methods = []
    costs = []
    colors = []
    
    if 'classical' in results:
        methods.append('Classical')
        costs.append(results['classical']['wirelength'])
        colors.append('lightblue')
        
    if 'spectral' in results:
        methods.append('Spectral')
        costs.append(results['spectral']['wirelength'])
        colors.append('lightgreen')
        
    if methods:
        plt.bar(methods, costs, color=colors, edgecolor='black')
        plt.ylabel("Wirelength Cost (Lower is Better)")
        plt.title(f"Method Comparison (m={m}, n={n})")
        for i, v in enumerate(costs):
            plt.text(i, v + (max(costs)*0.01), f"{v:.1f}", ha='center', fontweight='bold')
        plt.savefig(f"{output_dir}/comparison_bar_chart.png")
        plt.close()
        
    print(f"    -> All results saved to '{output_dir}/'")

if __name__ == "__main__":
    run_fair_comparison(m=20, n=40, grid_size=10, spectral_batch_size=10)
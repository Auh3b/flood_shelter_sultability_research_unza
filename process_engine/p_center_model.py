"""
P-Center Problem Solver with Pre-calculated OD Matrix
Minimizes the maximum distance between any demand point and its nearest facility
"""

from datetime import datetime

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pulp import *


class PCenter_Solver:
    """
    P-Center problem solver that works with existing OD matrices
    Objective: Locate p facilities to minimize the maximum service distance
    """

    def __init__(self, od_matrix=None, demand_gdf=None, facility_gdf=None):
        """
        Initialize P-Center solver

        Args:
            od_matrix: Distance/cost matrix (numpy array, DataFrame, or path to CSV)
            demand_gdf: Optional GeoDataFrame for visualization
            facility_gdf: Optional GeoDataFrame for visualization
        """
        self.demand_gdf = demand_gdf
        self.facility_gdf = facility_gdf
        self.distance_matrix = None
        self.solution = None

        if od_matrix is not None:
            self.load_od_matrix(od_matrix)

    def load_od_matrix(self, od_matrix):
        """
        Load OD matrix from various formats

        Args:
            od_matrix: Can be:
                - numpy array (n_demand x n_facilities)
                - pandas DataFrame
                - path to CSV file (QGIS format or matrix format)

        Returns:
            Loaded distance matrix as numpy array
        """
        if isinstance(od_matrix, np.ndarray):
            self.distance_matrix = od_matrix

        elif isinstance(od_matrix, pd.DataFrame):
            # Check if it's in long format (QGIS format)
            if 'InputID' in od_matrix.columns or 'OriginID' in od_matrix.columns:
                self.distance_matrix = self._convert_long_to_matrix(od_matrix)
            else:
                # Assume it's already a matrix
                self.distance_matrix = od_matrix.values

        elif isinstance(od_matrix, str):
            # Load from file
            df = pd.read_csv(od_matrix)
            if 'InputID' in df.columns or 'OriginID' in df.columns:
                self.distance_matrix = self._convert_long_to_matrix(df)
            else:
                self.distance_matrix = df.values

        print(
            f"OD Matrix loaded: {self.distance_matrix.shape[0]} demand points x {self.distance_matrix.shape[1]} facilities")
        print(
            f"Distance range: {self.distance_matrix.min():.2f} to {self.distance_matrix.max():.2f}")

        return self.distance_matrix

    def _convert_long_to_matrix(self, df_long):
        """
        Convert QGIS long-format OD matrix to matrix format
        """
        # Identify column names
        origin_col = None
        dest_col = None
        dist_col = None

        for col in df_long.columns:
            col_lower = col.lower()
            if 'input' in col_lower or 'origin' in col_lower or 'from' in col_lower:
                origin_col = col
            elif 'target' in col_lower or 'destin' in col_lower or 'to' in col_lower:
                dest_col = col
            elif 'distance' in col_lower or 'cost' in col_lower or 'time' in col_lower:
                dist_col = col

        if not all([origin_col, dest_col, dist_col]):
            raise ValueError(
                f"Could not identify columns. Found: {df_long.columns.tolist()}\n"
                f"Expected: InputID/OriginID, TargetID/DestinationID, Distance"
            )

        print(
            f"Converting long format: {origin_col} → {dest_col} with {dist_col}")

        # Get unique IDs
        origins = sorted(df_long[origin_col].unique())
        destinations = sorted(df_long[dest_col].unique())

        # Create mapping
        origin_idx = {oid: i for i, oid in enumerate(origins)}
        dest_idx = {did: i for i, did in enumerate(destinations)}

        # Initialize matrix
        matrix = np.full((len(origins), len(destinations)), np.inf)

        # Fill matrix
        for _, row in df_long.iterrows():
            i = origin_idx[row[origin_col]]
            j = dest_idx[row[dest_col]]
            matrix[i, j] = row[dist_col]

        if np.isinf(matrix).any():
            n_missing = np.isinf(matrix).sum()
            print(f"WARNING: {n_missing} OD pairs have no distance value")

        return matrix

    def load_qgis_distance_matrix(self, csv_path):
        """
        Convenience function for QGIS Distance Matrix output

        Args:
            csv_path: Path to CSV file from QGIS "Distance Matrix" tool
        """
        return self.load_od_matrix(csv_path)

    def solve(self, p, method='standard', time_limit=300, mip_gap=0.01):
        """
        Solve the P-Center problem

        Args:
            p: Number of facilities to locate
            method: 'standard' (exact) or 'vertex_substitution' (heuristic for large problems)
            time_limit: Solver time limit in seconds
            mip_gap: MIP optimality gap (e.g., 0.01 = 1%)

        Returns:
            Solution dictionary
        """
        if self.distance_matrix is None:
            raise ValueError("Load OD matrix first!")

        n_demand, n_facilities = self.distance_matrix.shape

        if p > n_facilities:
            raise ValueError(
                f"Cannot locate {p} facilities when only {n_facilities} candidates available")

        if p <= 0:
            raise ValueError("p must be positive")

        print(f"\n{'='*70}")
        print(f"SOLVING P-CENTER PROBLEM")
        print(f"{'='*70}")
        print(f"Demand points: {n_demand}")
        print(f"Candidate facilities: {n_facilities}")
        print(f"Facilities to locate (p): {p}")
        print(f"Method: {method}")
        print(f"{'='*70}\n")

        if method == 'standard':
            return self._solve_standard(p, time_limit, mip_gap)
        elif method == 'vertex_substitution':
            return self._solve_vertex_substitution(p, time_limit)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _solve_standard(self, p, time_limit, mip_gap):
        """
        Standard P-Center formulation using binary linear programming
        """
        n_demand, n_facilities = self.distance_matrix.shape

        # Create model
        model = LpProblem("P_Center", LpMinimize)

        # Decision variables
        # x[j] = 1 if facility j is selected
        x = LpVariable.dicts("facility", range(n_facilities), cat='Binary')

        # y[i,j] = 1 if demand i is assigned to facility j
        y = LpVariable.dicts("assign",
                             [(i, j) for i in range(n_demand)
                              for j in range(n_facilities)],
                             cat='Binary')

        # W = maximum distance (minimax objective)
        W = LpVariable("max_distance", lowBound=0, cat='Continuous')

        # Objective: minimize maximum distance
        model += W, "Minimize_Max_Distance"

        # Constraint 1: Select exactly p facilities
        model += lpSum([x[j] for j in range(n_facilities)]
                       ) == p, "Select_p_facilities"

        # Constraint 2: Each demand point assigned to exactly one facility
        for i in range(n_demand):
            model += lpSum([y[i, j] for j in range(n_facilities)]
                           ) == 1, f"Assign_demand_{i}"

        # Constraint 3: Can only assign to open facilities
        for i in range(n_demand):
            for j in range(n_facilities):
                model += y[i, j] <= x[j], f"Link_{i}_{j}"

        # Constraint 4: Maximum distance constraint
        for i in range(n_demand):
            for j in range(n_facilities):
                model += (self.distance_matrix[i, j] * y[i, j] <= W,
                          f"Max_dist_{i}_{j}")

        # Solve
        print("Solving optimization model...")
        start = datetime.now()
        solver = PULP_CBC_CMD(msg=1, timeLimit=time_limit, gapRel=mip_gap)
        model.solve(solver)
        solve_time = (datetime.now() - start).total_seconds()

        # Extract solution
        if model.status == 1:  # Optimal
            selected = [j for j in range(n_facilities) if x[j].varValue > 0.5]

            # Extract assignments
            assignments = {}
            for i in range(n_demand):
                for j in range(n_facilities):
                    if y[i, j].varValue > 0.5:
                        assignments[i] = j
                        break

            # Calculate service distances for each demand point
            service_distances = np.array([self.distance_matrix[i, assignments[i]]
                                         for i in range(n_demand)])

            # Calculate coverage counts (how many facilities within max distance)
            max_dist = W.varValue
            coverage_counts = np.sum(self.distance_matrix <= max_dist, axis=1)

            self.solution = {
                'status': 'Optimal',
                'p': p,
                'num_facilities': len(selected),
                'selected_facilities': selected,
                'assignments': assignments,
                'max_distance': max_dist,
                'service_distances': service_distances,
                'avg_distance': service_distances.mean(),
                'coverage_counts': coverage_counts,
                'solve_time': solve_time,
                'objective_value': value(model.objective)
            }

            self._print_solution()
            return self.solution
        else:
            print(f"No solution found. Status: {LpStatus[model.status]}")
            return {'status': LpStatus[model.status]}

    def _solve_vertex_substitution(self, p, max_iterations):
        """
        Vertex substitution heuristic for large P-Center problems
        Faster but may not find optimal solution
        """
        n_demand, n_facilities = self.distance_matrix.shape

        print("Using vertex substitution heuristic...")

        # Initial solution: select p facilities with best average coverage
        avg_distances = self.distance_matrix.mean(axis=0)
        selected = list(np.argsort(avg_distances)[:p])

        best_max_dist = float('inf')
        iterations = 0
        improved = True

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1

            # Assign demands to nearest selected facility
            assignments = {}
            service_distances = []
            for i in range(n_demand):
                distances_to_selected = [
                    self.distance_matrix[i, j] for j in selected]
                nearest_idx = np.argmin(distances_to_selected)
                assignments[i] = selected[nearest_idx]
                service_distances.append(distances_to_selected[nearest_idx])

            current_max_dist = max(service_distances)

            if current_max_dist < best_max_dist:
                best_max_dist = current_max_dist
                best_selected = selected.copy()
                best_assignments = assignments.copy()
                improved = True

            # Try swapping facilities
            for out_idx in selected:
                for in_idx in range(n_facilities):
                    if in_idx not in selected:
                        # Try swap
                        test_selected = [
                            j if j != out_idx else in_idx for j in selected]

                        # Calculate max distance for this configuration
                        test_distances = []
                        for i in range(n_demand):
                            min_dist = min([self.distance_matrix[i, j]
                                           for j in test_selected])
                            test_distances.append(min_dist)

                        test_max_dist = max(test_distances)

                        if test_max_dist < best_max_dist:
                            best_max_dist = test_max_dist
                            best_selected = test_selected
                            selected = test_selected
                            improved = True
                            print(
                                f"  Iteration {iterations}: Improved to {best_max_dist:.2f}")
                            break

                if improved:
                    break

        # Final assignment
        assignments = {}
        service_distances = []
        for i in range(n_demand):
            distances_to_selected = [
                self.distance_matrix[i, j] for j in best_selected]
            nearest_idx = np.argmin(distances_to_selected)
            assignments[i] = best_selected[nearest_idx]
            service_distances.append(distances_to_selected[nearest_idx])

        service_distances = np.array(service_distances)
        coverage_counts = np.sum(self.distance_matrix <= best_max_dist, axis=1)

        self.solution = {
            'status': 'Heuristic',
            'p': p,
            'num_facilities': len(best_selected),
            'selected_facilities': best_selected,
            'assignments': assignments,
            'max_distance': best_max_dist,
            'service_distances': service_distances,
            'avg_distance': service_distances.mean(),
            'coverage_counts': coverage_counts,
            'iterations': iterations,
            'objective_value': best_max_dist
        }

        print(f"\nHeuristic completed in {iterations} iterations")
        self._print_solution()
        return self.solution

    def _print_solution(self):
        """Print detailed solution information"""
        s = self.solution

        print(f"\n{'='*70}")
        print(f"SOLUTION")
        print(f"{'='*70}")
        print(f"Status: {s['status']}")
        if 'solve_time' in s:
            print(f"Solve time: {s['solve_time']:.2f} seconds")
        print(f"Facilities located (p): {s['p']}")
        print(f"Selected facility indices: {s['selected_facilities']}")

        print(f"\nService Distance Analysis:")
        print(f"  Maximum distance (objective): {s['max_distance']:.2f}")
        print(f"  Average distance: {s['avg_distance']:.2f}")
        print(f"  Minimum distance: {s['service_distances'].min():.2f}")
        print(f"  Standard deviation: {s['service_distances'].std():.2f}")

        # Identify demand points with maximum distance
        max_dist_demands = np.where(
            np.abs(s['service_distances'] - s['max_distance']) < 0.01)[0]
        if len(max_dist_demands) > 0:
            print(
                f"\n  Critical demand points (at max distance): {max_dist_demands.tolist()}")
            print(f"  Number of critical points: {len(max_dist_demands)}")

        print(f"\nCoverage at Maximum Distance:")
        print(
            f"  Min facilities covering any demand: {s['coverage_counts'].min()}")
        print(
            f"  Max facilities covering any demand: {s['coverage_counts'].max()}")
        print(
            f"  Avg facilities covering each demand: {s['coverage_counts'].mean():.2f}")

        # Distribution of assignments
        assignment_counts = pd.Series(
            s['assignments'].values()).value_counts().sort_index()
        print(f"\nFacility Workload (demand points assigned):")
        for fac_idx, count in assignment_counts.items():
            print(f"  Facility {fac_idx}: {count} demand points")

        print(f"{'='*70}\n")

    def sensitivity_analysis(self, p_values):
        """
        Analyze how objective changes with different values of p

        Args:
            p_values: List of p values to test

        Returns:
            DataFrame with results
        """
        results = []

        print("\nRunning sensitivity analysis...")
        for p in p_values:
            print(f"  Testing p = {p}")
            sol = self.solve(p, time_limit=60)

            if sol['status'] in ['Optimal', 'Heuristic']:
                results.append({
                    'p': p,
                    'max_distance': sol['max_distance'],
                    'avg_distance': sol['avg_distance'],
                    'min_distance': sol['service_distances'].min(),
                    'std_distance': sol['service_distances'].std(),
                    'min_coverage': sol['coverage_counts'].min(),
                    'avg_coverage': sol['coverage_counts'].mean()
                })

        if not results:
            print("No solutions found!")
            return None

        df = pd.DataFrame(results)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Max distance vs p
        axes[0, 0].plot(df['p'], df['max_distance'], 'o-',
                        linewidth=2, markersize=10, color='red')
        axes[0, 0].set_xlabel('Number of Facilities (p)', fontsize=11)
        axes[0, 0].set_ylabel('Maximum Service Distance', fontsize=11)
        axes[0, 0].set_title(
            'Maximum Distance vs Number of Facilities', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Average distance vs p
        axes[0, 1].plot(df['p'], df['avg_distance'], 'o-',
                        linewidth=2, markersize=10, color='blue')
        axes[0, 1].set_xlabel('Number of Facilities (p)', fontsize=11)
        axes[0, 1].set_ylabel('Average Service Distance', fontsize=11)
        axes[0, 1].set_title(
            'Average Distance vs Number of Facilities', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Distance spread
        axes[1, 0].plot(df['p'], df['max_distance'], 'o-',
                        linewidth=2, label='Maximum', color='red')
        axes[1, 0].plot(df['p'], df['avg_distance'], 's-',
                        linewidth=2, label='Average', color='blue')
        axes[1, 0].plot(df['p'], df['min_distance'], '^-',
                        linewidth=2, label='Minimum', color='green')
        axes[1, 0].set_xlabel('Number of Facilities (p)', fontsize=11)
        axes[1, 0].set_ylabel('Service Distance', fontsize=11)
        axes[1, 0].set_title('Distance Distribution vs p',
                             fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Coverage redundancy
        axes[1, 1].plot(df['p'], df['avg_coverage'], 'o-',
                        linewidth=2, markersize=10, color='purple')
        axes[1, 1].set_xlabel('Number of Facilities (p)', fontsize=11)
        axes[1, 1].set_ylabel('Average Coverage Count', fontsize=11)
        axes[1, 1].set_title('Coverage Redundancy vs p',
                             fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('p_center_sensitivity.png', dpi=300, bbox_inches='tight')
        print("\nSensitivity plot saved as 'p_center_sensitivity.png'")
        plt.show()

        return df

    def compare_with_lscp(self, p_value, coverage_distance):
        """
        Compare P-Center solution with LSCP at given coverage distance

        Args:
            p_value: Number of facilities for P-Center
            coverage_distance: Coverage standard for LSCP comparison
        """
        # Solve P-Center
        print(f"\nSolving P-Center with p={p_value}...")
        p_center_sol = self.solve(p_value)

        if p_center_sol['status'] not in ['Optimal', 'Heuristic']:
            print("P-Center solution failed")
            return

        # Analyze P-Center coverage at given distance
        coverage_matrix = (self.distance_matrix <=
                           coverage_distance).astype(int)
        p_center_coverage = coverage_matrix[:,
                                            p_center_sol['selected_facilities']]

        covered_by_p_center = np.sum(p_center_coverage.sum(axis=1) > 0)
        coverage_pct = 100 * covered_by_p_center / len(self.distance_matrix)

        print(f"\n{'='*70}")
        print(f"COMPARISON: P-CENTER vs LSCP")
        print(f"{'='*70}")
        print(f"\nP-Center Solution (p={p_value}):")
        print(f"  Facilities: {p_value}")
        print(f"  Maximum distance: {p_center_sol['max_distance']:.2f}")
        print(f"  Average distance: {p_center_sol['avg_distance']:.2f}")
        print(f"  Coverage at {coverage_distance}: {coverage_pct:.1f}%")
        print(
            f"  Covered demand points: {covered_by_p_center}/{len(self.distance_matrix)}")

        print(f"\nLSCP would require:")
        print(f"  Coverage standard: {coverage_distance}")
        print(f"  Estimated facilities: depends on network structure")
        print(f"  (Run LSCP solver for exact count)")

        if p_center_sol['max_distance'] <= coverage_distance:
            print(f"\n✓ P-Center solution achieves LSCP coverage standard!")
        else:
            print(f"\n✗ P-Center max distance exceeds LSCP standard")
            print(
                f"  Shortfall: {p_center_sol['max_distance'] - coverage_distance:.2f}")

        print(f"{'='*70}\n")

    def export_results(self, output_csv='p_center_solution.csv'):
        """
        Export solution to CSV files

        Args:
            output_csv: Output file path
        """
        if self.solution is None:
            raise ValueError("Solve the problem first!")

        # Facility results
        facility_df = pd.DataFrame({
            'facility_index': self.solution['selected_facilities'],
            'rank': range(1, self.solution['num_facilities'] + 1),
            'assigned_demand_count': pd.Series(self.solution['assignments'].values()).value_counts()[
                self.solution['selected_facilities']
            ].values
        })

        facility_df.to_csv(output_csv, index=False)
        print(f"Facility solution exported to: {output_csv}")

        # Demand assignments
        assignment_path = output_csv.replace('.csv', '_assignments.csv')
        assignment_data = []

        for i, fac_idx in self.solution['assignments'].items():
            assignment_data.append({
                'demand_index': i,
                'assigned_facility': fac_idx,
                'service_distance': self.solution['service_distances'][i],
                'is_critical': abs(self.solution['service_distances'][i] -
                                   self.solution['max_distance']) < 0.01,
                'coverage_count': self.solution['coverage_counts'][i]
            })

        assignment_df = pd.DataFrame(assignment_data)
        assignment_df.to_csv(assignment_path, index=False)
        print(f"Demand assignments exported to: {assignment_path}")

        return facility_df, assignment_df

    def visualize_with_geodata(self, save_path='p_center_solution.png',
                               show_assignments=True, show_coverage_circles=True):
        """
        Create map visualization

        Args:
            save_path: Path to save figure
            show_assignments: Draw lines from demand to assigned facility
            show_coverage_circles: Draw circles at maximum distance
        """
        if self.demand_gdf is None or self.facility_gdf is None:
            print("Geodata not available for visualization")
            return

        if self.solution is None:
            raise ValueError("Solve the problem first!")

        fig, ax = plt.subplots(figsize=(16, 12))

        # Plot all candidate facilities
        self.facility_gdf.plot(ax=ax, color='lightgray', marker='s',
                               markersize=100, alpha=0.4, label='Candidate Facilities',
                               zorder=1)

        # Plot selected facilities
        selected_gdf = self.facility_gdf.iloc[self.solution['selected_facilities']]
        selected_gdf.plot(ax=ax, color='red', marker='*', markersize=600,
                          edgecolors='darkred', linewidth=2.5, label='Selected Facilities (P-Center)',
                          zorder=4)

        # Draw coverage circles at max distance
        if show_coverage_circles:
            for idx, row in selected_gdf.iterrows():
                circle = plt.Circle((row.geometry.x, row.geometry.y),
                                    self.solution['max_distance'],
                                    color='red', fill=False,
                                    linestyle='--', alpha=0.3, linewidth=2)
                ax.add_patch(circle)

        # Draw assignment lines
        if show_assignments:
            for demand_idx, facility_idx in self.solution['assignments'].items():
                demand_pt = self.demand_gdf.iloc[demand_idx].geometry
                facility_pt = self.facility_gdf.iloc[facility_idx].geometry

                # Color by distance
                dist = self.solution['service_distances'][demand_idx]
                is_critical = abs(dist - self.solution['max_distance']) < 0.01

                ax.plot([demand_pt.x, facility_pt.x],
                        [demand_pt.y, facility_pt.y],
                        color='orange' if is_critical else 'gray',
                        alpha=0.6 if is_critical else 0.2,
                        linewidth=2 if is_critical else 0.5,
                        zorder=2)

        # Plot demand points colored by service distance
        scatter = ax.scatter(
            [p.x for p in self.demand_gdf.geometry],
            [p.y for p in self.demand_gdf.geometry],
            c=self.solution['service_distances'],
            cmap='YlOrRd',
            s=150,
            edgecolors='black',
            linewidth=0.8,
            alpha=0.8,
            zorder=3
        )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Service Distance', fontsize=12)

        ax.legend(loc='upper right', fontsize=12)

        title = f'P-Center Solution: {self.solution["p"]} Facilities\n'
        title += f'Maximum Distance: {self.solution["max_distance"]:.2f} | '
        title += f'Average Distance: {self.solution["avg_distance"]:.2f}'
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Map saved to: {save_path}")
        plt.show()


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_basic_p_center():
    """Basic P-Center example"""

    print("="*70)
    print("EXAMPLE: Basic P-Center Problem")
    print("="*70)

    # Create sample data
    np.random.seed(42)
    n_demand = 40
    n_facilities = 15

    distance_matrix = np.random.uniform(0, 10000, (n_demand, n_facilities))

    # Initialize solver
    solver = PCenter_Solver(od_matrix=distance_matrix)

    # Solve with p=5
    solution = solver.solve(p=5)

    # Export results
    solver.export_results('p_center_solution.csv')


def example_sensitivity_analysis():
    """Test different values of p"""

    print("\n" + "="*70)
    print("EXAMPLE: P-Center Sensitivity Analysis")
    print("="*70)

    np.random.seed(42)
    distance_matrix = np.random.uniform(0, 10000, (50, 20))

    solver = PCenter_Solver(od_matrix=distance_matrix)

    # Test p from 3 to 10
    sensitivity_df = solver.sensitivity_analysis(p_values=range(3, 11))

    print("\nSensitivity Results:")
    print(sensitivity_df)

    # Find "elbow" point
    improvements = sensitivity_df['max_distance'].diff().abs()
    print(
        f"\nLargest improvements at p = {sensitivity_df.loc[improvements.idxmax(), 'p']}")


def example_with_qgis_data():
    """Example using QGIS distance matrix and geodata"""

    # Load geodata
    demand_gdf = gpd.read_file('demand_points.shp')
    facility_gdf = gpd.read_file('facilities.shp')

    # Initialize solver
    solver = PCenter_Solver(
        od_matrix='qgis_distance_matrix.csv',
        demand_gdf=demand_gdf,
        facility_gdf=facility_gdf
    )

    # Solve
    solution = solver.solve(p=7)

    # Visualize
    solver.visualize_with_geodata(
        save_path='p_center_map.png',
        show_assignments=True,
        show_coverage_circles=True
    )

    # Export
    solver.export_results('p_center_solution.csv')


def example_compare_methods():
    """Compare standard and heuristic methods"""

    print("="*70)
    print("EXAMPLE: Comparing Solution Methods")
    print("="*70)

    np.random.seed(42)

"""
LSCP Solver with Pre-calculated OD Matrix
Includes flexible coverage constraints (partial coverage support)
"""

from datetime import datetime

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pulp import *


class LSCP_with_OD:
    """
    LSCP solver that works directly with existing OD matrices
    Supports both full (100%) and partial coverage requirements
    """

    def __init__(self, od_matrix=None, demand_gdf=None, facility_gdf=None):
        """
        Initialize with OD matrix

        Args:
            od_matrix: Distance/cost matrix (numpy array, DataFrame, or path to CSV)
            demand_gdf: Optional GeoDataFrame for visualization
            facility_gdf: Optional GeoDataFrame for visualization
        """
        self.demand_gdf = demand_gdf
        self.facility_gdf = facility_gdf
        self.distance_matrix = None
        self.coverage_matrix = None
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
                - path to CSV file
                - path to QGIS distance matrix output

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

        QGIS Distance Matrix tool outputs format like:
        InputID | TargetID | Distance
        --------|----------|----------
           1    |    1     |   0.0
           1    |    2     |  150.5
           ...
        """
        # Identify column names
        origin_col = None
        dest_col = None
        dist_col = None

        # Check for common column names
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
                f"Could not identify columns. Found columns: {df_long.columns.tolist()}\n"
                f"Expected columns like: InputID/OriginID, TargetID/DestinationID, Distance"
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

        # Check for missing values
        if np.isinf(matrix).any():
            n_missing = np.isinf(matrix).sum()
            print(
                f"WARNING: {n_missing} OD pairs have no distance value (set to infinity)")

        return matrix

    def load_qgis_distance_matrix(self, csv_path):
        """
        Convenience function specifically for QGIS Distance Matrix output

        Args:
            csv_path: Path to CSV file from QGIS "Distance Matrix" tool
        """
        return self.load_od_matrix(csv_path)

    def create_coverage_matrix(self, coverage_distance):
        """
        Create binary coverage matrix from OD matrix

        Args:
            coverage_distance: Maximum acceptable distance/time

        Returns:
            Coverage matrix (1 if within threshold, 0 otherwise)
        """
        if self.distance_matrix is None:
            raise ValueError("Load OD matrix first!")

        self.coverage_matrix = (self.distance_matrix <=
                                coverage_distance).astype(int)

        # Analysis
        coverage_per_demand = self.coverage_matrix.sum(axis=1)
        uncovered = np.where(coverage_per_demand == 0)[0]

        print(f"\nCoverage Analysis (distance ≤ {coverage_distance}):")
        print(
            f"  Facilities covering each demand (avg): {coverage_per_demand.mean():.2f}")
        print(
            f"  Min facilities covering any demand: {coverage_per_demand.min()}")
        print(
            f"  Max facilities covering any demand: {coverage_per_demand.max()}")

        if len(uncovered) > 0:
            print(
                f"\n  ⚠ WARNING: {len(uncovered)} demand points CANNOT be covered!")
            print(f"  Uncovered demand indices: {uncovered.tolist()}")
            min_distances = self.distance_matrix[uncovered, :].min(axis=1)
            print(f"  Minimum distances for uncovered points: {min_distances}")
            print(
                f"  Suggestion: Increase coverage to at least {min_distances.max():.2f} for full coverage")
            print(
                f"  Or use partial coverage constraint (e.g., coverage_percentage=0.95)")
        else:
            print(f"  ✓ All demand points can be covered")

        return self.coverage_matrix

    def solve(self, coverage_distance=None, coverage_percentage=1.0,
              demand_weights=None, time_limit=300):
        """
        Solve LSCP with flexible coverage constraint

        Args:
            coverage_distance: Maximum service distance (if not already set)
            coverage_percentage: Fraction of demand that must be covered (0.0 to 1.0)
                                0.95 = 95% coverage, 1.0 = 100% coverage (default)
            demand_weights: Array of demand weights (e.g., population). If provided,
                          coverage_percentage applies to weighted demand
            time_limit: Solver time limit in seconds

        Returns:
            Solution dictionary
        """
        # Create coverage matrix if needed
        if coverage_distance is not None:
            self.create_coverage_matrix(coverage_distance)

        if self.coverage_matrix is None:
            raise ValueError(
                "Set coverage distance first using create_coverage_matrix()")

        # Validate coverage percentage
        if not 0 < coverage_percentage <= 1.0:
            raise ValueError("coverage_percentage must be between 0 and 1")

        n_demand, n_facilities = self.coverage_matrix.shape

        # Handle demand weights
        if demand_weights is None:
            demand_weights = np.ones(n_demand)
        elif len(demand_weights) != n_demand:
            raise ValueError(
                f"demand_weights length ({len(demand_weights)}) must match number of demand points ({n_demand})")

        total_demand = demand_weights.sum()
        required_coverage = coverage_percentage * total_demand

        print(f"\n{'='*70}")
        print(f"SOLVING LSCP WITH PARTIAL COVERAGE")
        print(f"{'='*70}")
        print(f"Demand points: {n_demand}")
        print(f"Candidate facilities: {n_facilities}")
        print(
            f"Coverage requirement: {coverage_percentage*100:.1f}% of demand")
        if demand_weights is not None and not np.all(demand_weights == 1):
            print(f"Total demand weight: {total_demand:.2f}")
            print(f"Required coverage weight: {required_coverage:.2f}")
        print(f"{'='*70}\n")

        # Create model
        model = LpProblem("LSCP_Partial_Coverage", LpMinimize)

        # Decision variables
        x = LpVariable.dicts("facility", range(n_facilities), cat='Binary')
        y = LpVariable.dicts("covered", range(n_demand), cat='Binary')

        # Objective: minimize facilities
        model += lpSum([x[j] for j in range(n_facilities)]), "Total_Facilities"

        # Constraint 1: Coverage definition - demand i is covered if at least one facility covers it
        for i in range(n_demand):
            model += (
                lpSum([self.coverage_matrix[i, j] * x[j]
                      for j in range(n_facilities)]) >= y[i],
                f"Coverage_Definition_{i}"
            )

        # Constraint 2: Minimum coverage requirement (weighted or unweighted)
        model += (
            lpSum([demand_weights[i] * y[i]
                  for i in range(n_demand)]) >= required_coverage,
            "Minimum_Coverage"
        )

        # Solve
        print("Solving...")
        start = datetime.now()
        model.solve(PULP_CBC_CMD(msg=1, timeLimit=time_limit))
        solve_time = (datetime.now() - start).total_seconds()

        # Extract solution
        if model.status == 1:
            selected = [j for j in range(n_facilities) if x[j].varValue > 0.5]
            covered_demand = [i for i in range(
                n_demand) if y[i].varValue > 0.5]
            uncovered_demand = [i for i in range(
                n_demand) if y[i].varValue < 0.5]

            # Calculate coverage statistics
            coverage_counts = np.zeros(n_demand)
            for i in covered_demand:
                coverage_counts[i] = sum(
                    [self.coverage_matrix[i, j] for j in selected])

            # Calculate service distances
            service_distances = np.full(n_demand, np.inf)
            for i in covered_demand:
                available_facilities = [
                    j for j in selected if self.coverage_matrix[i, j] == 1]
                if available_facilities:
                    service_distances[i] = min(
                        [self.distance_matrix[i, j] for j in available_facilities])

            # Calculate actual coverage achieved
            actual_coverage_weight = sum(
                [demand_weights[i] for i in covered_demand])
            actual_coverage_pct = actual_coverage_weight / total_demand

            self.solution = {
                'status': 'Optimal',
                'num_facilities': len(selected),
                'selected_facilities': selected,
                'covered_demand': covered_demand,
                'uncovered_demand': uncovered_demand,
                'coverage_counts': coverage_counts,
                'service_distances': service_distances,
                'solve_time': solve_time,
                'requested_coverage': coverage_percentage,
                'actual_coverage': actual_coverage_pct,
                'num_covered': len(covered_demand),
                'num_uncovered': len(uncovered_demand),
                'demand_weights': demand_weights,
                'total_demand': total_demand
            }

            # Add service quality metrics for covered demand
            covered_distances = service_distances[service_distances != np.inf]
            if len(covered_distances) > 0:
                self.solution['avg_service_distance'] = covered_distances.mean()
                self.solution['max_service_distance'] = covered_distances.max()
            else:
                self.solution['avg_service_distance'] = None
                self.solution['max_service_distance'] = None

            self._print_solution()
            return self.solution
        else:
            print(f"No solution found. Status: {LpStatus[model.status]}")
            if model.status == -1:  # Infeasible
                print("\nPossible reasons:")
                print(
                    "1. Coverage distance too small - no facilities can cover enough demand")
                print("2. Coverage percentage too high - try lowering it")
                print("3. Check your distance matrix for errors")
            return {'status': LpStatus[model.status]}

    def _print_solution(self):
        """Print solution details"""
        s = self.solution

        print(f"\n{'='*70}")
        print(f"SOLUTION")
        print(f"{'='*70}")
        print(f"Status: {s['status']}")
        print(f"Solve time: {s['solve_time']:.2f} seconds")
        print(f"Facilities needed: {s['num_facilities']}")
        print(f"Selected facility indices: {s['selected_facilities']}")

        print(f"\nCoverage Achievement:")
        print(f"  Requested coverage: {s['requested_coverage']*100:.1f}%")
        print(f"  Actual coverage: {s['actual_coverage']*100:.1f}%")
        print(
            f"  Demand points covered: {s['num_covered']} / {s['num_covered'] + s['num_uncovered']}")

        if s['num_uncovered'] > 0:
            print(f"  Uncovered demand points: {s['uncovered_demand']}")
            uncovered_weights = [s['demand_weights'][i]
                                 for i in s['uncovered_demand']]
            if not np.all(s['demand_weights'] == 1):
                print(
                    f"  Uncovered demand weight: {sum(uncovered_weights):.2f}")

        if s['avg_service_distance'] is not None:
            print(f"\nService Quality (for covered demand):")
            print(
                f"  Average service distance: {s['avg_service_distance']:.2f}")
            print(
                f"  Maximum service distance: {s['max_service_distance']:.2f}")

        covered_with_service = s['coverage_counts'][s['coverage_counts'] > 0]
        if len(covered_with_service) > 0:
            print(f"\nCoverage Redundancy (for covered demand):")
            print(
                f"  Min facilities per demand: {covered_with_service.min():.0f}")
            print(
                f"  Max facilities per demand: {covered_with_service.max():.0f}")
            print(
                f"  Avg facilities per demand: {covered_with_service.mean():.2f}")

            single = np.sum(covered_with_service == 1)
            if single > 0:
                pct = 100 * single / len(covered_with_service)
                print(
                    f"\n  ⚠ {single} covered points ({pct:.1f}%) have only 1 facility")
                print(f"    Consider increasing coverage distance for redundancy")

        print(f"{'='*70}\n")

    def find_minimum_coverage_feasible(self, coverage_distance, min_percentage=0.5,
                                       max_percentage=1.0, step=0.05):
        """
        Find the minimum coverage percentage that yields a feasible solution

        Args:
            coverage_distance: Coverage distance to test
            min_percentage: Minimum coverage to try (default 0.5 = 50%)
            max_percentage: Maximum coverage to try (default 1.0 = 100%)
            step: Step size for testing (default 0.05 = 5%)

        Returns:
            Dict with minimum feasible coverage and corresponding solution
        """
        if self.coverage_matrix is None:
            self.create_coverage_matrix(coverage_distance)

        print(f"\nFinding minimum feasible coverage percentage...")
        print(
            f"Testing range: {min_percentage*100:.0f}% to {max_percentage*100:.0f}%")

        # Try from low to high
        percentages = np.arange(min_percentage, max_percentage + step, step)

        for pct in percentages:
            print(f"\nTrying {pct*100:.1f}% coverage...")
            sol = self.solve(coverage_percentage=pct, time_limit=60)

            if sol['status'] == 'Optimal':
                print(
                    f"\n✓ Found feasible solution at {pct*100:.1f}% coverage")
                print(f"  Requires {sol['num_facilities']} facilities")
                return {
                    'minimum_feasible_coverage': pct,
                    'solution': sol
                }

        print("\n✗ No feasible solution found in tested range")
        return None

    def sensitivity_analysis(self, distance_values, coverage_percentage=1.0):
        """
        Test multiple coverage distances

        Args:
            distance_values: List of distances to test
            coverage_percentage: Coverage requirement (0.0 to 1.0)

        Returns:
            DataFrame with results
        """
        results = []

        print(
            f"\nRunning sensitivity analysis (coverage requirement: {coverage_percentage*100:.1f}%)...")
        for dist in distance_values:
            print(f"  Testing coverage distance: {dist}")
            sol = self.solve(coverage_distance=dist,
                             coverage_percentage=coverage_percentage,
                             time_limit=60)

            if sol['status'] == 'Optimal':
                results.append({
                    'coverage_distance': dist,
                    'num_facilities': sol['num_facilities'],
                    'actual_coverage_pct': sol['actual_coverage'] * 100,
                    'num_covered': sol['num_covered'],
                    'num_uncovered': sol['num_uncovered'],
                    'avg_service_dist': sol.get('avg_service_distance', None),
                    'max_service_dist': sol.get('max_service_distance', None),
                    'min_redundancy': sol['coverage_counts'][sol['coverage_counts'] > 0].min() if np.any(sol['coverage_counts'] > 0) else 0,
                    'avg_redundancy': sol['coverage_counts'][sol['coverage_counts'] > 0].mean() if np.any(sol['coverage_counts'] > 0) else 0
                })

        if not results:
            print("No feasible solutions found!")
            return None

        df = pd.DataFrame(results)

        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Facilities vs Distance
        axes[0, 0].plot(df['coverage_distance'],
                        df['num_facilities'], 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Coverage Distance')
        axes[0, 0].set_ylabel('Number of Facilities')
        axes[0, 0].set_title('Facilities Required vs Coverage Distance')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Coverage achieved
        axes[0, 1].plot(df['coverage_distance'], df['actual_coverage_pct'],
                        'o-', linewidth=2, markersize=8, color='green')
        axes[0, 1].axhline(y=coverage_percentage*100, color='r',
                           linestyle='--', label=f'Target: {coverage_percentage*100:.0f}%')
        axes[0, 1].set_xlabel('Coverage Distance')
        axes[0, 1].set_ylabel('Coverage Achieved (%)')
        axes[0, 1].set_title('Actual Coverage vs Distance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Service quality
        if df['avg_service_dist'].notna().any():
            axes[1, 0].plot(df['coverage_distance'], df['avg_service_dist'],
                            'o-', linewidth=2, markersize=8, label='Average')
            axes[1, 0].plot(df['coverage_distance'], df['max_service_dist'],
                            's--', linewidth=2, markersize=8, label='Maximum')
            axes[1, 0].set_xlabel('Coverage Distance')
            axes[1, 0].set_ylabel('Service Distance')
            axes[1, 0].set_title('Service Distance vs Coverage Distance')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Redundancy
        axes[1, 1].plot(df['coverage_distance'], df['avg_redundancy'],
                        'o-', linewidth=2, markersize=8, label='Average')
        axes[1, 1].plot(df['coverage_distance'], df['min_redundancy'],
                        's--', linewidth=2, markersize=8, label='Minimum')
        axes[1, 1].set_xlabel('Coverage Distance')
        axes[1, 1].set_ylabel('Coverage Redundancy')
        axes[1, 1].set_title('Coverage Redundancy vs Distance')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        print("\nSensitivity plot saved as 'sensitivity_analysis.png'")
        plt.show()

        return df

    def export_results(self, output_csv='lscp_solution.csv'):
        """
        Export solution to CSV

        Args:
            output_csv: Output file path
        """
        if self.solution is None:
            raise ValueError("Solve the problem first!")

        # Facility results
        facility_df = pd.DataFrame({
            'facility_index': self.solution['selected_facilities'],
            'rank': range(1, self.solution['num_facilities'] + 1),
            'covers_n_demand': [self.coverage_matrix[:, j].sum()
                                for j in self.solution['selected_facilities']]
        })

        facility_df.to_csv(output_csv, index=False)
        print(f"Solution exported to: {output_csv}")

        # Detailed demand coverage
        assignment_path = output_csv.replace('.csv', '_demand_coverage.csv')
        demand_data = []

        for i in range(len(self.distance_matrix)):
            is_covered = i in self.solution['covered_demand']

            if is_covered:
                covering = [j for j in self.solution['selected_facilities']
                            if self.coverage_matrix[i, j] == 1]
                nearest_fac = covering[np.argmin(
                    [self.distance_matrix[i, j] for j in covering])] if covering else None
                nearest_dist = self.solution['service_distances'][i]
            else:
                covering = []
                nearest_fac = None
                nearest_dist = None

            demand_data.append({
                'demand_index': i,
                'is_covered': is_covered,
                'demand_weight': self.solution['demand_weights'][i],
                'num_covering_facilities': len(covering),
                'covering_facility_indices': str(covering) if covering else 'None',
                'nearest_facility': nearest_fac,
                'distance_to_nearest': nearest_dist
            })

        demand_df = pd.DataFrame(demand_data)
        demand_df.to_csv(assignment_path, index=False)
        print(f"Demand coverage details exported to: {assignment_path}")

        return facility_df, demand_df

    def visualize_with_geodata(self, save_path='solution_map.png', show_uncovered=True):
        """
        Create map visualization (requires demand_gdf and facility_gdf)

        Args:
            save_path: Path to save figure
            show_uncovered: Highlight uncovered demand points
        """
        if self.demand_gdf is None or self.facility_gdf is None:
            print("Geodata not available for visualization")
            return

        if self.solution is None:
            raise ValueError("Solve the problem first!")

        fig, ax = plt.subplots(figsize=(14, 10))

        # Plot all facilities
        self.facility_gdf.plot(ax=ax, color='lightgray', marker='s',
                               markersize=80, alpha=0.4, label='Candidate Facilities')

        # Plot selected facilities
        selected_gdf = self.facility_gdf.iloc[self.solution['selected_facilities']]
        selected_gdf.plot(ax=ax, color='red', marker='*', markersize=500,
                          edgecolors='darkred', linewidth=2, label='Selected Facilities')

        # Separate covered and uncovered demand
        covered_gdf = self.demand_gdf.iloc[self.solution['covered_demand']]

        if show_uncovered and self.solution['num_uncovered'] > 0:
            uncovered_gdf = self.demand_gdf.iloc[self.solution['uncovered_demand']]
            uncovered_gdf.plot(ax=ax, color='orange', marker='x', markersize=150,
                               linewidth=3, label='Uncovered Demand', zorder=3, alpha=0.8)

        # Plot covered demand with coverage coloring
        covered_counts = self.solution['coverage_counts'][self.solution['covered_demand']]
        scatter = ax.scatter(
            [p.x for p in covered_gdf.geometry],
            [p.y for p in covered_gdf.geometry],
            c=covered_counts,
            cmap='YlOrRd',
            s=120,
            edgecolors='black',
            linewidth=0.5,
            alpha=0.7,
            zorder=2
        )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Coverage Redundancy', fontsize=11)

        ax.legend(loc='upper right', fontsize=11)

        title = f'LSCP Solution: {self.solution["num_facilities"]} Facilities\n'
        title += f'Coverage: {self.solution["actual_coverage"]*100:.1f}% '
        title += f'({self.solution["num_covered"]}/{self.solution["num_covered"]+self.solution["num_uncovered"]} points)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Map saved to: {save_path}")
        plt.show()

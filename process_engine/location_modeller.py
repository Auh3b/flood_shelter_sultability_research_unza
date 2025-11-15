"""
LSCP Solver with Pre-calculated OD Matrix
Supports various OD matrix formats commonly used in QGIS
"""

from datetime import datetime

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pulp import *


class LSCP_Solver:
    def __init__(self, od_matrix, dest_col, origin_col, value_col):

        self.ref_matrix = od_matrix
        self.dest_col = dest_col
        self.origin_col = origin_col
        self.value_col = value_col
        self.origin_dict = None
        self.dest_dict = None
        self.matrix = None
        self.coverage_matrix = None
        self.solution = None

        if self.matrix == None:
            self.load_od_matrix(od_matrix)

    def load_od_matrix(self, od_matrix):
        df = od_matrix
        if (isinstance(od_matrix, str)):
            df = pd.read_csv(od_matrix)

        self.matrix = self._convert_df_to_matrix(df)

    def _convert_df_to_matrix(self, df: pd.DataFrame):

        origins = sorted(df[self.origin_col].unique())
        dests = sorted(df[self.dest_col].unique())

        origin_idx = {oid: i for i, oid in enumerate(origins)}
        self.origin_dict = origin_idx

        dest_idx = {did: i for i, did in enumerate(dests)}
        self.dest_dict = dest_idx

        # Initialize matrix
        matrix = np.full((len(origins), len(dests)), np.inf)

        # Fill matrix
        for _, row in df.iterrows():
            i = origin_idx[row[self.origin_col]]
            j = dest_idx[row[self.dest_col]]
            matrix[i, j] = row[self.value_col]

        # Check for missing values
        if np.isinf(matrix).any():
            n_missing = np.isinf(matrix).sum()
            print(
                f"WARNING: {n_missing} OD pairs have no distance value (set to infinity)")

        return matrix

    def create_coverage_matrix(self, coverage):
        """
        Create binary coverage matrix from OD matrix

        Args:
            coverage: Maximum acceptable distance/time

        Returns:
            Coverage matrix (1 if within threshold, 0 otherwise)
        """
        if self.matrix is None:
            raise ValueError("Load OD matrix first!")

        self.coverage_matrix = (self.matrix <=
                                coverage).astype(int)

        # Analysis
        coverage_per_demand = self.coverage_matrix.sum(axis=1)
        uncovered = np.where(coverage_per_demand == 0)[0]

        print(f"\nCoverage Analysis (distance ≤ {coverage}):")
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
            print(
                f"  Minimum distances for uncovered: {self.matrix[uncovered, :].min(axis=1)}")
        else:
            print(f"  ✓ All demand points can be covered")

        return self.coverage_matrix

    def solve(self, coverage, coverage_percentage=None, demand_weights=None):

        # Create coverage matrix if needed
        if coverage is not None:
            self.create_coverage_matrix(coverage)

        if self.coverage_matrix is None:
            raise ValueError(
                "Set coverage distance first using create_coverage_matrix()")

        # Validate coverage percentage
        if not 0 < coverage_percentage <= 1.0:
            raise ValueError("coverage_percentage must be between 0 and 1")

        n_demand, n_facilities = self.matrix.shape

        # Handle demand weights
        if demand_weights is None:
            demand_weights = np.ones(n_demand)
        elif len(demand_weights) != n_demand:
            raise ValueError(
                f"demand_weights length ({len(demand_weights)}) must match number of demand points ({n_demand})")

        total_demand = demand_weights.sum()
        required_coverage = coverage_percentage * total_demand

        print("required", required_coverage)

        print(f"\n{'='*70}")
        print(f"SOLVING LSCP WITH OD MATRIX")
        print(f"{'='*70}")
        print(f"Demand points: {n_demand}")
        print(f"Candidate facilities: {n_facilities}")
        print(f"{'='*70}\n")

        # Create model
        model = LpProblem("LSCP_OD", LpMinimize)

        # Decision variables
        x = LpVariable.dicts("facility", range(n_facilities), cat='Binary')
        y = LpVariable.dicts("covered", range(n_demand), cat='Binary')

        # Objective: minimize facilities
        model += lpSum([x[j] for j in range(n_facilities)]), "Total_Facilities"

        # Constraints: cover all demand
        for i in range(n_demand):
            model += (
                lpSum([self.coverage_matrix[i, j] * x[j]
                      for j in range(n_facilities)]) >= required_coverage,
                f"Cover_{i}"
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
        model.solve(PULP_CBC_CMD(msg=1, timeLimit=300))
        solve_time = (datetime.now() - start).total_seconds()
        print(model.status)
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
                        [self.matrix[i, j] for j in available_facilities])

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

        print(f"\nService Quality:")
        print(f"  Average service distance: {s['avg_service_distance']:.2f}")
        print(f"  Maximum service distance: {s['max_service_distance']:.2f}")

        print(f"\nCoverage Redundancy:")
        print(f"  Min coverage per demand: {s['coverage_counts'].min()}")
        print(f"  Max coverage per demand: {s['coverage_counts'].max()}")
        print(f"  Avg coverage per demand: {s['coverage_counts'].mean():.2f}")

        single = np.sum(s['coverage_counts'] == 1)
        if single > 0:
            pct = 100 * single / len(s['coverage_counts'])
            print(
                f"\n  ⚠ {single} demand points ({pct:.1f}%) have only 1 facility")
            print(f"    Consider increasing coverage distance for redundancy")

        print(f"{'='*70}\n")

    def export_results(self, output_csv='lscp_solution.csv'):
        """
        Export solution to CSV

        Args:
            output_csv: Output file path
        """
        if self.solution is None:
            raise ValueError("Solve the problem first!")

        facility_dict = {k: j for j, k in self.dest_dict.items()}
        demand_dict = {k: j for j, k in self.origin_dict.items()}

        # Facility results
        facility_df = pd.DataFrame({
            'facility_index': self.solution['selected_facilities'],
            'rank': range(1, self.solution['num_facilities'] + 1),
            'covers_n_demand': [self.coverage_matrix[:, j].sum()
                                for j in self.solution['selected_facilities']]
        })

        facility_df.to_csv(output_csv, index=False)
        print(f"Solution exported to: {output_csv}")

        # Detailed assignment
        assignment_path = output_csv.replace('.csv', '_assignments.csv')
        assignments = []

        for i in range(len(self.matrix)):
            covering = [j for j in self.solution['selected_facilities']
                        if self.coverage_matrix[i, j] == 1]

            demand_index = demand_dict[i]
            num_covering_facilities = len(covering)
            covering_facility_indices = str(covering)
            _nearest_facilities = covering[np.argmin(
                [self.matrix[i, j] for j in covering])] if covering else None
            nearest_facilities = facility_dict[_nearest_facilities]
            distance_to_nearest = self.solution['service_distances'][i]
            assignments.append({
                'demand_index': demand_index,
                'num_covering_facilities': num_covering_facilities,

                "covering_facility_indices": covering_facility_indices,
                "nearest_facilities": nearest_facilities,
                "distance_to_nearest":  distance_to_nearest
            })

        assign_df = pd.DataFrame(assignments)
        assign_df.to_csv(assignment_path, index=False)
        print(f"Assignments exported to: {assignment_path}")

        return facility_df, assign_df

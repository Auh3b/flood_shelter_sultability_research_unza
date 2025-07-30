import json
import os

import process_engine.stats as pes

with open(os.path.join(os.getcwd(), 'raw_data', 'sample.json'), 'r') as file:
    data = json.load(file)
idx, idx_alt = pes.generate_criteria_indecies(data)
criteria_length = len(idx)
empty_matrix = pes.generate_decision_matrix(criteria_length)
filled_matrix = pes.populate_matrix(empty_matrix, data, idx)
weighted_matrix = pes.weighted_matrix(filled_matrix)
weighted_criteria = pes.weighted_criteria(weighted_matrix)
# print(weighted_criteria)
lambda_max = pes.lambda_max(filled_matrix, weighted_criteria)
ci = pes.consistency_index(lambda_max, criteria_length)
cr = pes.consistency_ratio(ci, pes.get_random_interval(criteria_length))
print(lambda_max, ci, cr)

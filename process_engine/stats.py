import numpy as np
import numpy.typing as npt


def consistency_index(lambda_max: float, criteria_number: int) -> float:
    return (lambda_max - criteria_number)/(criteria_number - 1)


def consistency_ratio(ci: float, ri: float) -> float:
    return ci/ri


def reciprocical_value(value) -> float:
    return 1/value


def generate_decision_matrix(criteria_number: int):
    return np.random.rand(criteria_number, criteria_number)


def populate_matrix(matrix: npt.NDArray, data: dict[str, dict], idx_lookup: dict[str, int]):
    output_matrix = matrix.copy()
    for key, value in data.items():
        idx1 = idx_lookup[key]

        if (idx1 is None):
            continue

        output_matrix[idx1, idx1] = 1

        for subkey, subvalue in value.items():
            idx2 = idx_lookup[subkey]

            if (idx1 is None):
                continue

            importancy = subvalue['importancy']
            scale = int(subvalue['scale'])

            if (importancy == 'A'):
                output_matrix[idx1, idx2] = scale
                output_matrix[idx2, idx1] = reciprocical_value(scale)
            else:
                output_matrix[idx2, idx1] = scale
                output_matrix[idx1, idx2] = reciprocical_value(scale)

    return output_matrix


def generate_criteria_indecies(input: dict[str, dict]) -> dict[str, int]:
    values = []
    for key, value in input.items():
        values.append(key)
        for sub_key, sub_value in value.items():
            values.append(sub_key)

    idx = {value: index for index,
           value in enumerate(list(set(values)))}

    idx_alt = {value: index for index,
               value in enumerate(list(set(values)))}
    return [idx, idx_alt]


def weighted_matrix(filled_matrix: npt.NDArray) -> npt.NDArray:
    output_matrix = filled_matrix.copy()
    sum_columns = np.sum(filled_matrix, axis=0)
    with np.nditer(output_matrix, flags=['multi_index'], op_flags=['readwrite'])as it:
        for x in it:
            col, row = it.multi_index
            x[...] = x / sum_columns[col]
    return output_matrix


def weighted_criteria(matrix: npt.NDArray):
    return np.mean(matrix, axis=1)


def lambda_max(matrix: npt.NDArray, weighted_criteria: npt.NDArray) -> float:
    output_matrix = matrix.copy()
    lambdas = [np.matmul(value, weighted_criteria)/weighted_criteria[idx]
               for idx, value in enumerate(output_matrix)]
    return max(lambdas)


def get_random_interval(value: int) -> float:
    ri_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12,
               6: 1.24, 7: 1.32, 8: 1.41, 9: 146, 10: 1.49}
    return ri_dict.get(value)

import numpy as np


def min_max_scale(matrix):
    matrix = np.asarray(matrix, dtype=float)

    minimums = matrix.min(axis=0)
    maximums = matrix.max(axis=0)
    ranges = maximums - minimums
    ranges[ranges == 0] = 1

    return (matrix - minimums) / ranges, minimums, maximums


def scale_one_vector(vector, minimums, maximums):
    vector = np.asarray(vector, dtype=float)

    ranges = maximums - minimums
    ranges = np.where(ranges == 0, 1, ranges)

    return (vector - minimums) / ranges


def standardise(matrix):
    matrix = np.asarray(matrix, dtype=float)

    means = matrix.mean(axis=0)
    standard_deviations = matrix.std(axis=0, ddof=1)
    standard_deviations[standard_deviations == 0] = 1

    return (matrix - means) / standard_deviations


def vector_norm(vector):
    vector = np.asarray(vector, dtype=float)
    return np.sqrt(np.dot(vector, vector))


def cosine_similarity(vector_a, vector_b):
    denominator = vector_norm(vector_a) * vector_norm(vector_b)

    if denominator == 0:
        return 0.0

    return float(np.dot(vector_a, vector_b) / denominator)


def cosine_similarities(matrix, target_vector):
    return np.array(
        [cosine_similarity(row, target_vector) for row in matrix]
    )


def transform_vectors(matrix, weights):
    transformation = np.diag(np.asarray(weights, dtype=float))
    return np.asarray(matrix, dtype=float) @ transformation


def pca_from_scratch(matrix, number_of_components=2):
    standardised = standardise(matrix)

    covariance = (
        standardised.T @ standardised
    ) / (standardised.shape[0] - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    components = eigenvectors[:, :number_of_components]
    projected = standardised @ components

    total = eigenvalues.sum()

    if total == 0:
        explained_ratio = np.zeros_like(eigenvalues)
    else:
        explained_ratio = eigenvalues / total

    return {
        "projected": projected,
        "components": components,
        "explained_ratio": explained_ratio,
    }


def similar_players(feature_matrix, selected_index):
    scaled, _, _ = min_max_scale(feature_matrix)
    target = scaled[selected_index]

    return cosine_similarities(scaled, target)


def role_match_scores(feature_matrix, target_vector, weights):
    scaled, minimums, maximums = min_max_scale(feature_matrix)
    scaled_target = scale_one_vector(
        target_vector,
        minimums,
        maximums,
    )

    transformed_players = transform_vectors(scaled, weights)
    transformed_target = transform_vectors(
        scaled_target.reshape(1, -1),
        weights,
    )[0]

    return cosine_similarities(
        transformed_players,
        transformed_target,
    )

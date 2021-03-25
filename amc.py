"""Given Absorbing Markov Chain find Absorbing Probabilities
From Wikipedia: https://en.wikipedia.org/wiki/Absorbing_Markov_chain

AMC is represented in a form:
[Q R]
[O I]

Q - matrix of probabilities of transition between transient states,
R - matrix of probabilities of transition from transient to absorbing states,
O - zero matrix
I - identity matrix (in our particular case we're provided with zero matrix)

Absorbing Probabilities are determined as:
B = (I - Q)^-1 * R
"""


import copy
import fractions


class EmptyMatrix(Exception):
    pass


class NoAbsorbingStates(Exception):
    pass


class NoTransientStates(Exception):
    pass


class NonSquareMatrix(Exception):
    pass


class DifferentSizedMatrices(Exception):
    pass


class ZeroDeterminant(Exception):
    pass


def number_of_transients(matrix):
    """Get number of transients states.

    Assume absorbing states follow transient states
    without interlieveing."""

    if len(matrix) == 0:
        raise EmptyMatrix

    for r, row in enumerate(matrix):
        for col in row:
            if col != 0:
                break  # This is not an all-zero row, try next one
        else:
            return r  # Has just finished looping over an empty row

    # Reached end of table and didn't encounter all-zero row
    raise NoAbsorbingStates


def decompose(matrix):
    """Decompose input matrix on Q and R components."""

    transients = number_of_transients(matrix)

    if transients == 0:
        raise NoTransientStates

    q_matrix = [matrix[i][:transients] for i in range(transients)]
    r_matrix = [matrix[i][transients:] for i in range(transients)]

    if q_matrix == []:
        raise NoTransientStates

    if r_matrix == []:
        raise NoAbsorbingStates

    return q_matrix, r_matrix


def identity(size):
    """Return identity matrix of given size."""

    matrix = []

    for i in range(size):
        row = []

        for j in range(size):
            row.append(int(i == j))

        matrix.append(row)

    return matrix


def is_zero(matrix):
    """Check if the matrix is zero."""

    for row in matrix:
        for col in row:
            if col != 0:
                return False

    return True


def swap(matrix, i, j):
    """Swap i, j rows/columns of a square matrix."""

    swapped = copy.deepcopy(matrix)
    swapped[i], swapped[j] = swapped[j], swapped[i]

    for row in swapped:
        row[i], row[j] = row[j], row[i]

    return swapped


def sort_matrix(matrix):
    """Reorder matrix so zero-rows go last."""

    size = len(matrix)
    zero_row = -1

    for r in range(size):
        row_sum = 0

        for c in range(size):
            row_sum += matrix[r][c]

        if row_sum == 0:
            zero_row = r  # Save the found zero-row
        elif row_sum != 0 and zero_row > -1:
            # We have found non-zero row after all-zero row:
            # swap these rows and repeat from the begining.

            sorted_matrix = swap(matrix, r, zero_row)
            return sort_matrix(sorted_matrix)

    return matrix  # Nothing to sort, return original matrix


def normalize(matrix, use_fractions=False):
    """Normalize matrix."""

    normalized = copy.deepcopy(matrix)

    for r, row in enumerate(matrix):
        row_sum = sum(row) or 1

        for c, col in enumerate(row):
            if use_fractions:
                normalized[r][c] = fractions.Fraction(col, row_sum)
            else:
                normalized[r][c] = float(col) / row_sum

    return normalized


def subtract(matrix_a, matrix_b):
    """Subtract two matrices."""

    if len(matrix_a) != len(matrix_a[0]) or len(matrix_b) != len(matrix_b[0]):
        raise NonSquareMatrix

    if len(matrix_a) != len(matrix_b):
        raise DifferentSizedMatrices

    subtracted_matrix = []

    for r, row in enumerate(matrix_a):
        subtracted_row = []

        for c, col in enumerate(row):
            subtracted_row.append(col - matrix_b[r][c])

        subtracted_matrix.append(subtracted_row)

    return subtracted_matrix


def multiply(matrix_a, matrix_b):
    """Multiply two matrices."""

    if matrix_a == [] or matrix_b == []:
        raise EmptyMatrix

    if len(matrix_a[0]) != len(matrix_b):
        raise DifferentSizedMatrices

    multiplied_matrix = []

    cols = len(matrix_b[0])
    iters = len(matrix_a[0])

    for r, row in enumerate(matrix_a):
        multiplied_row = []

        for c in range(cols):
            col_sum = 0

            for i in range(iters):
                col_sum += row[i] * matrix_b[i][c]

            multiplied_row.append(col_sum)

        multiplied_matrix.append(multiplied_row)

    return multiplied_matrix


def transposeMatrix(matrix):
    """Transpose matrix."""

    return [list(row) for row in zip(*matrix)]


def get_matrix_minor(matrix, i, j):
    return [row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])]


def get_matrix_determinant(matrix):
    """Get matrix determinant."""

    # Base case for 2x2 matrix
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    determinant = 0

    for c in range(len(matrix)):
        determinant += ((-1) ** c) * matrix[0][c] * get_matrix_determinant(
            get_matrix_minor(matrix, 0, c))

    return determinant


def get_matrix_inverse(matrix):
    """Get matrix inversion."""

    determinant = get_matrix_determinant(matrix)

    if determinant == 0:
        raise ZeroDeterminant

    # Special case for 2x2 matrix
    if len(matrix) == 2:
        return [
            [matrix[1][1] / determinant, -1 * matrix[0][1] / determinant],
            [-1 * matrix[1][0] / determinant, matrix[0][0] / determinant],
        ]

    # Find matrix of cofactors
    cofactors = []

    for r in range(len(matrix)):
        cofactorRow = []

        for c in range(len(matrix)):
            minor = get_matrix_minor(matrix, r, c)
            cofactorRow.append(
                ((-1) ** (r + c)) * get_matrix_determinant(minor))

        cofactors.append(cofactorRow)

    cofactors = transposeMatrix(cofactors)

    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c] / determinant

    return cofactors

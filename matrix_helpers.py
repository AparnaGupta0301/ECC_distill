# sparse_matrix_ops.py

# Third-party imports
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, identity, isspmatrix



# Public Functions
def mod2_matrix_sp(matrix):
    """
    Reduces the elements of a given sparse matrix or dense matrix modulo 2
    and eliminates explicit zeros for optimization.
    
    Args:
        spmatrix (Union[np.ndarray, csr_matrix]): Input matrix, either dense (NumPy array) or sparse (CSR format).
    
    Returns:
        csr_matrix: The updated sparse matrix with elements reduced modulo 2.
    
    Raises:
        ValueError: If the input is not a valid NumPy array or SciPy sparse matrix.
    """
    # If the input is a dense NumPy array, convert it to a sparse CSR matrix after modulo operation
    if isinstance(matrix, np.ndarray):
        matrix = (matrix % 2).astype(np.uint8)  # Reduce modulo 2
        matrix = csr_matrix(matrix)            # Convert to sparse CSR matrix
    elif isspmatrix(matrix):
        matrix = matrix.astype(np.uint8).tocsr()  # Ensure it is in CSR format
        matrix.data %= 2  # Perform modulo 2 operation only on the non-zero elements
        matrix.eliminate_zeros()  # Remove explicit zeros
    else:
        raise ValueError("Input must be a NumPy array or SciPy sparse matrix.")

    return matrix


def mod2_multiply_sp(matrix_a, matrix_b):
    """
    Perform sparse matrix multiplication modulo 2 for very sparse matrices.

    Args:
        matrix_a (Union[np.ndarray, sp.csr_matrix]): First input matrix.
        matrix_b (Union[np.ndarray, sp.csr_matrix]): Second input matrix.

    Returns:
        csr_matrix: Resulting sparse matrix modulo 2.

    Raises:
        ValueError: If the input matrices are not of type np.ndarray or sp.csr_matrix.
    """
    # Ensure both matrices are in sparse CSR format and modulo 2
    if isinstance(matrix_a, np.ndarray):
        matrix_a = (matrix_a % 2).astype(np.uint8)
        matrix_a = csr_matrix(matrix_a)
    elif isspmatrix(matrix_a):
        matrix_a = matrix_a.astype(np.uint8).tocsr()
    else:
        raise ValueError("matrix_a must be a NumPy array or SciPy sparse matrix.")

    if isinstance(matrix_b, np.ndarray):
        matrix_b = (matrix_b % 2).astype(np.uint8)
        matrix_b = csr_matrix(matrix_b)
    elif isspmatrix(matrix_b):
        matrix_b = matrix_b.astype(np.uint8).tocsr()
    else:
        raise ValueError("matrix_b must be a NumPy array or SciPy sparse matrix.")

    # Perform the sparse matrix multiplication
    product = matrix_a @ matrix_b

    # Return reduced modulo 2
    return mod2_matrix_sp(product)

# Public API
__all__ = ["mod2_matrix_sp", "mod2_multiply_sp"]
# Public API
__all__ = [
    "row_echelon_sp",
    "quotient_basis_sp",
    "is_orthogonal_sp",
    "canonicalize_ops",
    "mod2_inverse_sp",
]

# Public Functions
def row_echelon_sp(matrix, full=False):
    """
    Converts a binary matrix to row echelon form via Gaussian Elimination (mod 2),
    using uint8 for storage. Optimization is done by converting to LIL format 
    for efficient row operations, then converting back to CSR at the end.

    Parameters
    ----------
    matrix : np.ndarray or scipy.sparse matrix
        Binary input matrix with entries in {0,1}.
    full : bool, optional
        If True, perform elimination above and below pivot.
        If False, eliminate only below pivot.

    Returns
    -------
    row_ech_form : csr_matrix
        Row echelon form of the matrix (uint8 dtype).
    rank : int
        Rank of the matrix.
    transform_matrix : csr_matrix
        Transformation matrix (uint8 dtype).
    pivot_cols : list
        Indices of pivot columns.
    """
    # Ensure binary and convert to uint8
    if isinstance(matrix, np.ndarray):
        matrix = (matrix % 2).astype(np.uint8)
        matrix = lil_matrix(matrix)
    elif isspmatrix(matrix):
        matrix = matrix.astype(np.uint8).tolil()
    else:
        raise ValueError("Input must be a NumPy array or SciPy sparse matrix.")

    num_rows, num_cols = matrix.shape

    # Handle empty case
    if num_rows == 0 or num_cols == 0:
        transform_matrix = identity(num_rows, format="csr", dtype=np.uint8)
        return matrix.tocsr(), 0, transform_matrix, []

    transform_matrix = identity(num_rows, format="lil", dtype=np.uint8)
    pivot_row = 0
    pivot_cols = []

    for col in range(num_cols):
        # Find a pivot row: first row at or below pivot_row with a 1 in this column
        pivot_candidates = [r for r in range(pivot_row, num_rows) if col in matrix.rows[r]]
        if not pivot_candidates:
            continue
        
        # Choose the topmost candidate as pivot
        swap_row = pivot_candidates[0]
        if swap_row != pivot_row:
            # Swap rows in matrix
            matrix.rows[pivot_row], matrix.rows[swap_row] = matrix.rows[swap_row], matrix.rows[pivot_row]
            matrix.data[pivot_row], matrix.data[swap_row] = matrix.data[swap_row], matrix.data[pivot_row]

            # Swap rows in transform_matrix
            transform_matrix.rows[pivot_row], transform_matrix.rows[swap_row] = transform_matrix.rows[swap_row], transform_matrix.rows[pivot_row]
            transform_matrix.data[pivot_row], transform_matrix.data[swap_row] = transform_matrix.data[swap_row], transform_matrix.data[pivot_row]

        # Now pivot_row is the pivot row
        pivot_cols.append(col)
        pivot_row_indices = set(matrix.rows[pivot_row])

        # Determine which rows to eliminate
        if full:
            # Eliminate in all rows except the pivot row
            eliminate_rows = [r for r in range(num_rows) if r != pivot_row and col in matrix.rows[r]]
        else:
            # Eliminate only below the pivot row
            eliminate_rows = [r for r in range(pivot_row+1, num_rows) if col in matrix.rows[r]]

        # Perform row elimination
        for r in eliminate_rows:
            # XOR operation: symmetric difference of sets of column indices
            row_indices = set(matrix.rows[r])
            new_indices = row_indices.symmetric_difference(pivot_row_indices)

            # Update the matrix row
            matrix.rows[r] = sorted(new_indices)
            matrix.data[r] = [1]*len(new_indices)  # all entries are 1 in a binary row

            # Update the transform_matrix row
            t_row_indices = set(transform_matrix.rows[r])
            p_row_indices = set(transform_matrix.rows[pivot_row])
            new_t_indices = t_row_indices.symmetric_difference(p_row_indices)
            transform_matrix.rows[r] = sorted(new_t_indices)
            transform_matrix.data[r] = [1]*len(new_t_indices)

        pivot_row += 1
        if pivot_row >= num_rows:
            break

    return matrix.tocsr(), pivot_row, transform_matrix.tocsr(), pivot_cols


def quotient_basis_sp(subgroup_matrix, group_matrix):
    """
    Compute the basis of the quotient group (group_matrix / subgroup_matrix).

    Parameters:
        subgroup_matrix : scipy.sparse.csr_matrix
            Matrix representing the subgroup (A).
        group_matrix : scipy.sparse.csr_matrix
            Matrix representing the group (B).

    Returns:
        csr_matrix: Basis of the quotient group B / A.
    """
    from scipy.sparse import vstack, isspmatrix

    # Validate inputs
    if not (isspmatrix(subgroup_matrix) and isspmatrix(group_matrix)):
        raise ValueError("Both subgroup_matrix and group_matrix must be SciPy sparse matrices.")

    num_rows_subgroup, num_cols_subgroup = subgroup_matrix.shape
    num_rows_group, num_cols_group = group_matrix.shape

    if num_cols_subgroup != num_cols_group:
        raise ValueError("The number of columns in both matrices must match.")

    # Step 1: Compute row echelon form of the subgroup matrix
    echelon_subgroup, subgroup_rank, _, pivot_columns = row_echelon_sp(subgroup_matrix)

    # Verify that subgroup_matrix is full row rank
    if subgroup_rank != num_rows_subgroup:
        raise ValueError("The subgroup_matrix must be full row rank.")

    # Step 2: Compute column permutation to bring pivot columns to the front
    column_permutation = pivot_columns + [j for j in range(num_cols_subgroup) if j not in pivot_columns]

    # Step 3: Stack echelon_subgroup and group_matrix vertically
    combined_matrix = vstack([echelon_subgroup, group_matrix])

    # Step 4: Compute row echelon form of the combined matrix with column permutation
    permuted_combined = combined_matrix[:, column_permutation]
    echelon_combined, combined_rank, _, _ = row_echelon_sp(permuted_combined)

    # Check rank conditions
    if combined_rank !=  num_rows_group:
        raise ValueError("The group_matrix is not a subgroup of the given subgroup_matrix.")

    # Step 5: Extract the quotient basis
    inverse_column_perm = [i for i, _ in sorted(enumerate(column_permutation), key=lambda x: x[1])]
    quotient_basis = echelon_combined[subgroup_rank:combined_rank, inverse_column_perm]

    return quotient_basis


def is_orthogonal_sp(matrix_a, matrix_b):
    """
    Checks if the modulo 2 multiplication of two matrices results in a zero matrix.

    Args:
        matrix_a (Union[np.ndarray, sp.csr_matrix]): First input matrix.
        matrix_b (Union[np.ndarray, sp.csr_matrix]): Second input matrix.

    Returns:
        bool: True if matrix_a @ matrix_b.T modulo 2 results in a zero matrix, False otherwise.
    """
    # Perform sparse matrix multiplication modulo 2
    product = mod2_multiply_sp(matrix_a, matrix_b.T)

    # Check if the resulting sparse matrix is equivalent to a zero matrix
    return product.nnz == 0


def canonicalize_ops(op_x, op_z):
    """
    Transforms logical X and Z operators to satisfy the canonical commutation relation (op_x @ op_z^T = I).
    
    Args:
        op_x (csr_matrix): Logical X operators represented as a sparse matrix.
        op_z (csr_matrix): Logical Z operators represented as a sparse matrix.
    
    Returns:
        tuple: A tuple containing:
            - Transformed op_x matrix (csr_matrix).
            - Original op_z matrix (csr_matrix, unchanged).
            - Transformation matrix (csr_matrix).
    """
    # Compute the commutation matrix
    comm_matrix = mod2_multiply_sp(op_x, op_z.T)
    
    # Calculate the transformation matrix to ensure canonical commutation
    transform_matrix = mod2_inverse_sp(comm_matrix)
    
    # Apply the transformation to op_x
    op_x_transformed = mod2_multiply_sp(transform_matrix, op_x)
    
    return op_x_transformed, op_z, transform_matrix


def mod2_inverse_sp(matrix):
    """
    Computes the left inverse of a full-rank binary sparse matrix.

    Parameters
    ----------
    matrix: np.ndarray or scipy.sparse matrix
        The binary sparse matrix to be inverted in scipy.sparse.csr_matrix format. 
        This matrix must either be square full-rank or rectangular with full-column rank.

    Returns
    -------
    scipy.sparse.csr_matrix
        The inverted binary sparse matrix.
    """
    if isinstance(matrix, np.ndarray):
        matrix = (matrix % 2).astype(np.uint8)
        matrix = csr_matrix(matrix)
    elif isspmatrix(matrix):
        matrix = matrix.astype(np.uint8).tocsr()
    else:
        raise ValueError("Input must be a NumPy array or SciPy sparse matrix.")

    m, n = matrix.shape

    # Convert to row echelon form using provided row_echelon function
    row_echelon_form, matrix_rank, transform, _ = row_echelon_sp(matrix.copy(), full=True)

    if m == n and matrix_rank == m:  # Full-rank square matrix
        return transform

    if m > matrix_rank and n == matrix_rank:  # Left inverse for full-column rank
        # Use sparse matrix multiplication and modulo 2 arithmetic
        return mod2_multiply_sp(row_echelon_form.T, transform)

    raise ValueError("This matrix is not invertible. Provide a full-rank square matrix or a rectangular matrix with full-column rank.")


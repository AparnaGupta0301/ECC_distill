from ldpc.mod2 import kernel

from matrix_helpers import quotient_basis_sp, canonicalize_ops

__all__ = [
    "compute_css_logical_operators",
]

def compute_css_logical_operators(x_parity_check_matrix, z_parity_check_matrix):
    """Calculates logical X and Z operators for a quantum code.

    Note:
        Logical operators derived from `x_parity_check_matrix` are logical Z operators.
        Logical operators derived from `z_parity_check_matrix` are logical X operators.

    Args:
        x_parity_check_matrix (np.ndarray): Parity-check matrix for X errors.
        z_parity_check_matrix (np.ndarray): Parity-check matrix for Z errors.

    Returns:
        tuple: A tuple (logical_x_matrix, logical_z_matrix) where:
            logical_x_matrix (np.ndarray): Logical X operator matrix.
            logical_z_matrix (np.ndarray): Logical Z operator matrix.
    """
    # Compute the kernel of the X parity-check matrix to derive logical Z operators
    x_kernel_matrix = kernel(x_parity_check_matrix)
    z_logical_operators = quotient_basis_sp(z_parity_check_matrix, x_kernel_matrix)

    # Compute the kernel of the Z parity-check matrix to derive logical X operators
    z_kernel_matrix = kernel(z_parity_check_matrix)
    x_logical_operators = quotient_basis_sp(x_parity_check_matrix, z_kernel_matrix)

    # Canonicalize the logical operators to ensure orthogonality
    canonical_x_matrix, canonical_z_matrix, _ = canonicalize_ops(x_logical_operators, z_logical_operators)

    return canonical_x_matrix, canonical_z_matrix
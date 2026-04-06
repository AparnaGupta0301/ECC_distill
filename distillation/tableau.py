import numpy as np
import stim

# Define the public API
__all__ = [
    "compute_error_syndrome",
    "compute_logical_phases",
    "compute_symplectic_product",
    "construct_resource_state",
    "construct_css_resource_state",
]


def _compute_mod2_product(matrix: np.ndarray, xx_input: np.ndarray, zz_input: np.ndarray) -> np.ndarray:
    """Computes the mod-2 product for a given matrix and inputs, compensating for YY phase.

    This is a utility function used by both `compute_error_syndrome` and
    `compute_logical_phases`.

    Args:
        matrix (np.ndarray): The input matrix (parity or logical operators).
        xx_input (np.ndarray): The input measurement results for the X basis.
        zz_input (np.ndarray): The input measurement results for the Z basis.

    Returns:
        np.ndarray: The computed mod-2 product.
    """
    # Split the operator matrix into X and Z parts
    x_part = matrix[:, :xx_input.size]
    z_part = matrix[:, xx_input.size:]

    # Compute the diagonal phase contributions from YY = -XX @ ZZ
    yy_phase_correction = np.einsum('ij,ij->i', x_part, z_part) % 2

    return (x_part @ xx_input + z_part @ zz_input + yy_phase_correction) % 2


def compute_error_syndrome(parity_check_matrix: np.ndarray, xx_input: np.ndarray, zz_input: np.ndarray) -> np.ndarray:
    """Computes the error syndrome given the parity check matrix and inputs.

    Args:
        parity_check_matrix (np.ndarray): The parity check matrix (shape: m x 2n).
        xx_input (np.ndarray): The input measurement results for the X basis (shape: n).
        zz_input (np.ndarray): The input measurement results for the Z basis (shape: n).

    Returns:
        np.ndarray: The computed error syndrome (shape: m).
    """
    return _compute_mod2_product(parity_check_matrix, xx_input, zz_input)


def compute_logical_phases(logical_matrix: np.ndarray, xx_input: np.ndarray, zz_input: np.ndarray) -> np.ndarray:
    """Computes the logical operator phases.

    Args:
        logical_matrix (np.ndarray): The logical operator matrix (shape: k x 2n).
        xx_input (np.ndarray): The input measurement results for the X basis (shape: n).
        zz_input (np.ndarray): The input measurement results for the Z basis (shape: n).

    Returns:
        np.ndarray: The computed logical phases (shape: k).
    """
    return _compute_mod2_product(logical_matrix, xx_input, zz_input)


def compute_symplectic_product(mat_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
    """Computes the symplectic product for matrices or vectors.

    Args:
        mat_a (np.ndarray): The first matrix or vector (shape: m x 2n or 2n,).
        mat_b (np.ndarray): The second matrix or vector (shape: k x 2n, or 2n,).

    Returns:
        np.ndarray: The symplectic product. If both inputs are matrices, returns
                    an m x k matrix. If one input is a vector, returns a 1D array.
                    If both inputs are vectors, returns a scalar.
    """
    num_qubits = mat_a.shape[-1] // 2

    # Ensure mat_a and mat_b have compatible dimensions
    if mat_a.shape[-1] != mat_b.shape[-1]:
        raise ValueError("Inputs must have the same number of columns (2n).")

    # Handle different cases for input shapes
    if mat_a.ndim == 1:
        mat_a = mat_a[np.newaxis, :]  # Reshape to a single-row matrix
    if mat_b.ndim == 1:
        mat_b = mat_b[np.newaxis, :]  # Reshape to a single-row matrix

    # Split into X and Z parts
    x_a, z_a = mat_a[:, :num_qubits], mat_a[:, num_qubits:]
    x_b, z_b = mat_b[:, :num_qubits], mat_b[:, num_qubits:]

    # Compute symplectic product
    symplectic_result = (x_a @ z_b.T + z_a @ x_b.T) % 2

    # Return appropriate shape based on input dimensions
    if mat_a.shape[0] == 1 and mat_b.shape[0] == 1:
        return symplectic_result[0, 0]  # Return scalar if both inputs are vectors
    elif mat_a.shape[0] == 1 or mat_b.shape[0] == 1:
        return symplectic_result.flatten()  # Return 1D array if one input is a vector
    else:
        return symplectic_result  # Return 2D array if both inputs are matrices


def construct_resource_state(parity_matrix, logical_x_matrix, logical_z_matrix):
    """Constructs a stim.Tableau for the resource state.

    This function takes the parity check matrix and logical X and Z matrices 
    and constructs a tableau for the corresponding resource state.

    Args:
        parity_matrix (numpy.ndarray): The parity check matrix (H).
        logical_x_matrix (numpy.ndarray): The logical X operators matrix.
        logical_z_matrix (numpy.ndarray): The logical Z operators matrix.

    Returns:
        stim.Tableau: The resulting tableau for the resource state.
    """
    # Ensure inputs are explicitly np.bool_
    parity_matrix = parity_matrix.astype(np.bool_, copy=False)
    logical_x_matrix = logical_x_matrix.astype(np.bool_, copy=False)
    logical_z_matrix = logical_z_matrix.astype(np.bool_, copy=False)

    # Extract dimensions
    num_logicals = logical_z_matrix.shape[0]  # Number of logical qubits
    num_checks = parity_matrix.shape[0]       # Number of parity checks
    num_physical = parity_matrix.shape[1] // 2  # Number of physical qubits

    # Generate stabilizers from the parity check matrix
    stabilizers = [
        stim.PauliString.from_numpy(
            xs=parity_matrix[i, :num_physical], zs=parity_matrix[i, num_physical:]
        )
        for i in range(num_checks)
    ]

    # Precompute identity and zero matrices for constructing Bell pair stabilizers
    identity_matrix = np.eye(num_logicals, dtype=np.bool_)  # Logical identity matrix
    zero_matrix = np.zeros((num_logicals, num_logicals), dtype=np.bool_)  # Logical zero matrix

    # Build logical-physical Bell pair stabilizers
    xx_stabilizer_matrix = np.hstack((
        logical_x_matrix[:, :num_physical], identity_matrix,
        logical_x_matrix[:, num_physical:], zero_matrix
    ))
    zz_stabilizer_matrix = np.hstack((
        logical_z_matrix[:, :num_physical], zero_matrix,
        logical_z_matrix[:, num_physical:], identity_matrix
    ))

    # Add logical-physical Bell pair stabilizers to the stabilizer list
    for logical_idx in range(num_logicals):
        stabilizers.extend([
            stim.PauliString.from_numpy(
                xs=xx_stabilizer_matrix[logical_idx, :num_physical + num_logicals],
                zs=xx_stabilizer_matrix[logical_idx, num_physical + num_logicals:]
            ),
            stim.PauliString.from_numpy(
                xs=zz_stabilizer_matrix[logical_idx, :num_physical + num_logicals],
                zs=zz_stabilizer_matrix[logical_idx, num_physical + num_logicals:]
            )
        ])

    # Construct and return the tableau
    return stim.Tableau.from_stabilizers(
        stabilizers=stabilizers,
        allow_redundant=False,
        allow_underconstrained=False
    )

def construct_css_resource_state(x_parity_check_matrix, z_parity_check_matrix, logical_x_matrix, logical_z_matrix):
    """Constructs a stim.Tableau for the resource state of a CSS code.

    This function extends classical parity-check matrices and logical operator 
    matrices to CSS quantum codes and constructs a tableau for the resource state.

    Args:
        x_parity_check_matrix (numpy.ndarray): The X parity-check matrix (Hx) of shape (m, n).
        z_parity_check_matrix (numpy.ndarray): The Z parity-check matrix (Hz) of shape (m, n).
        logical_x_matrix (numpy.ndarray): The logical X operators matrix of shape (k, n).
        logical_z_matrix (numpy.ndarray): The logical Z operators matrix of shape (k, n).

    Returns:
        stim.Tableau: The resulting tableau for the resource state.
    """
    # Ensure inputs are explicitly np.bool_
    x_parity_check_matrix = x_parity_check_matrix.astype(np.bool_, copy=False)
    z_parity_check_matrix = z_parity_check_matrix.astype(np.bool_, copy=False)
    logical_x_matrix = logical_x_matrix.astype(np.bool_, copy=False)
    logical_z_matrix = logical_z_matrix.astype(np.bool_, copy=False)

    # Pad the classical parity-check matrices for CSS codes
    x_padded_matrix = np.hstack((x_parity_check_matrix, np.zeros_like(x_parity_check_matrix)))
    z_padded_matrix = np.hstack((np.zeros_like(z_parity_check_matrix), z_parity_check_matrix))
    css_parity_check_matrix = np.vstack((x_padded_matrix, z_padded_matrix))

    # Pad the logical operators for CSS codes
    logical_x_padded = np.hstack((logical_x_matrix, np.zeros_like(logical_x_matrix)))
    logical_z_padded = np.hstack((np.zeros_like(logical_z_matrix), logical_z_matrix))

    # Construct and return the resource state tableau
    return construct_resource_state(css_parity_check_matrix, logical_x_padded, logical_z_padded)
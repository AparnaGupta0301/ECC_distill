import os
import subprocess
import random
import pickle

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from ldpc.code_util import estimate_code_distance, search_cycles


class LDPCCodeGenerator:
    """
    A class for generating and constructing LDPC codes and hypergraph product codes.
    """

    TEMP_FILE_PATH = "_parity-check.tmp"
    LDPC_LIBRARY_PATH = "/Users/aparnagupta/Documents/ProtographLDPC/LDPC-library/make-pchk.py"
    PRINT_PCHK_PATH = "/Users/aparnagupta/Documents/ProtographLDPC/LDPC-codes/print-pchk"
    CONVERT_PCHK_PATH = "/Users/aparnagupta/Documents/ProtographLDPC/LDPC-codes/pchk-to-alist"

    def __init__(self, ldpc_library_path: str = None, print_pchk_path: str = None, convert_pchk_path: str = None):
        """
        Initializes the LDPCCodeGenerator with optional custom tool paths.

        Args:
            ldpc_library_path (str, optional): Path to the LDPC library script.
            print_pchk_path (str, optional): Path to the print-pchk binary.
            convert_pchk_path (str, optional): Path to the pchk-to-alist binary.
        """
        if ldpc_library_path:
            self.LDPC_LIBRARY_PATH = ldpc_library_path
        if print_pchk_path:
            self.PRINT_PCHK_PATH = print_pchk_path
        if convert_pchk_path:
            self.CONVERT_PCHK_PATH = convert_pchk_path

    def generate_until_valid(
        self,
        output_file_name: str,
        num_checks: int,
        num_bits: int,
        num_checks_per_col: int,
        code_distance: int,
        code_type: str = "regular",
        construction_method: str = "peg",
        fraction_transmitted: float = 1.0,
        max_attempts: int = 200
    ) -> str:
        
        for attempt in range(1, max_attempts + 1):
            random_seed = random.randint(0, 2**32 - 1)
            print(f"Attempt {attempt}/{max_attempts} | Seed: {random_seed}")

            generate_command = [
                "python", self.LDPC_LIBRARY_PATH,
                "--output-pchk-file", self.TEMP_FILE_PATH,
                "--code-type", code_type,
                "--construction", construction_method,
                "--n-checks", str(num_checks),
                "--n-bits", str(num_bits),
                "--checks-per-col", str(num_checks_per_col),
                "--fraction-transmitted", str(fraction_transmitted),
                "--seed", str(random_seed)
            ]
            print_command = [self.PRINT_PCHK_PATH, self.TEMP_FILE_PATH]
            convert_command = [self.CONVERT_PCHK_PATH, "-t", self.TEMP_FILE_PATH, self.TEMP_FILE_PATH]

            try:
                subprocess.run(generate_command, check=True)
                subprocess.run(print_command, check=True)
                subprocess.run(convert_command, check=True)
            except subprocess.CalledProcessError as error:
                print(f"Error during LDPC code processing: {error}")
                continue

            parity_check_matrix = self.alist_to_sparse(self.TEMP_FILE_PATH)

            estimated_distance, _, _ = estimate_code_distance(parity_check_matrix)
            if estimated_distance < code_distance:
                print(f"Failed: Expected distance >= {code_distance}, got {estimated_distance}")
                continue

            has_short_cycles = search_cycles(parity_check_matrix, girth=4)
            if has_short_cycles:
                print("Failed: Found cycles of girth 4.")
                continue

            with open(output_file_name, "wb") as output_file:
                pickle.dump(
                    (parity_check_matrix, num_checks_per_col, code_type, construction_method, random_seed),
                    output_file
                )

            print(f"Success on attempt {attempt}: distance={estimated_distance}, girth > 4.")
            return "Success"

        raise RuntimeError(
            f"Failed to generate a valid LDPC code after {max_attempts} attempts. "
            f"Try relaxing code_distance (currently {code_distance}) or adjusting num_checks/num_bits."
        )

    @staticmethod
    def alist_to_sparse(alist_filename: str) -> csr_matrix:
        """
        Converts a matrix in alist format to a scipy.sparse.csr_matrix.

        Args:
            alist_filename (str): Path to the alist file.

        Returns:
            csr_matrix: The sparse matrix in CSR format.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        if not os.path.exists(alist_filename):
            raise FileNotFoundError(f"The file '{alist_filename}' does not exist.")

        with open(alist_filename, 'r') as f:
            N, M = map(int, f.readline().split())
            f.readline()  # Skip biggest_num_n, biggest_num_m line
            f.readline()  # Column weights
            f.readline()  # Row weights

            nlist = [list(map(int, f.readline().split())) for _ in range(N)]
            mlist = [list(map(int, f.readline().split())) for _ in range(M)]

        rows, cols = zip(*(
            (i, col_index - 1)
            for i, row_indices in enumerate(mlist)
            for col_index in row_indices if col_index != 0
        ))
        data = np.ones(len(rows), dtype=np.uint8)

        return csr_matrix((data, (rows, cols)), shape=(M, N))

    @staticmethod
    def hypergraph_product_code(H1: sp.spmatrix, H2: sp.spmatrix):
        """
        Constructs Hx and Hz parity-check matrices for a hypergraph product code.

        Args:
            H1 (sp.spmatrix): First sparse parity-check matrix.
            H2 (sp.spmatrix): Second sparse parity-check matrix.

        Returns:
            tuple[sp.csr_matrix, sp.csr_matrix]: The Hx and Hz parity-check matrices.
        """
        r1, n1 = H1.shape
        r2, n2 = H2.shape

        I_r1 = sp.eye(r1, format='csr', dtype=int)
        I_n1 = sp.eye(n1, format='csr', dtype=int)
        I_r2 = sp.eye(r2, format='csr', dtype=int)
        I_n2 = sp.eye(n2, format='csr', dtype=int)

        Hx = sp.hstack([sp.kron(H1, I_n2, format='csr'), sp.kron(I_r1, H2.T, format='csr')], format='csr')
        Hz = sp.hstack([sp.kron(I_n1, H2, format='csr'), sp.kron(H1.T, I_r2, format='csr')], format='csr')

        return Hx, Hz
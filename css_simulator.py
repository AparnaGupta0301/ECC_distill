import datetime
from typing import Dict, Optional, Union

import numpy as np
import scipy.sparse as sp
import stim
from tqdm import tqdm
from ldpc import BpOsdDecoder

from tableau import construct_css_resource_state

__all__ = ["MonteCarloCSS"]

class MonteCarloCSS:
    """Runs Monte Carlo simulations of a CSS code for X or Z errors using syndrome-based decoding.

    Args:
        parity_check_x_matrix (np.ndarray or sp.csr_matrix): Parity-check matrix for X errors.
        parity_check_z_matrix (np.ndarray or sp.csr_matrix): Parity-check matrix for Z errors.
        logical_x_matrix (np.ndarray or sp.csr_matrix): Matrix representing logical X operators.
        logical_z_matrix (np.ndarray or sp.csr_matrix): Matrix representing logical Z operators.
        depol_error_rate (float): Probability of depolarization error in the channel.
        decoder_config (dict): Configuration for decoders.
        target_simulations (int): Number of Monte Carlo simulations to perform.
        disable_progress (bool, optional): Disable the progress bar. Defaults to False.
        save_interval_sec (int, optional): Interval (in seconds) for saving results. Defaults to 60.
        random_seed (int, optional): Seed for the random number generator. Defaults to None.
        auto_run (bool, optional): Start simulation upon initialization. Defaults to False.
    """

    def __init__(
        self,
        parity_check_x_matrix: Union[np.ndarray, sp.csr_matrix],
        parity_check_z_matrix: Union[np.ndarray, sp.csr_matrix],
        logical_x_matrix: Union[np.ndarray, sp.csr_matrix],
        logical_z_matrix: Union[np.ndarray, sp.csr_matrix],
        depol_error_rate: float,
        decoder_config: Dict,
        target_simulations: int = 1000,
        disable_progress: bool = False,
        save_interval_sec: int = 60,
        random_seed: Optional[int] = None,
        auto_run: bool = False,
    ) -> None:
        self._validate_inputs(
            parity_check_x_matrix, parity_check_z_matrix, logical_x_matrix, logical_z_matrix,
            depol_error_rate, decoder_config, target_simulations, disable_progress, save_interval_sec, random_seed
        )

        self.parity_check_x_matrix = parity_check_x_matrix
        self.parity_check_z_matrix = parity_check_z_matrix
        self.logical_x_matrix = logical_x_matrix
        self.logical_z_matrix = logical_z_matrix
        self.depol_error_rate = depol_error_rate
        self.decoder_config = decoder_config
        self.target_simulations = target_simulations
        self.disable_progress = disable_progress
        self.save_interval_sec = save_interval_sec

        if random_seed is not None:
            np.random.seed(random_seed)

        self.x_parity_decoder = BpOsdDecoder(self.parity_check_x_matrix, **self.decoder_config)
        self.z_parity_decoder = BpOsdDecoder(self.parity_check_z_matrix, **self.decoder_config)

        self.num_logical_qubits, self.num_physical_qubits = logical_z_matrix.shape
        self.num_total_qubits = self.num_physical_qubits + self.num_logical_qubits
        self.initial_stabilizers = self._initialize_stabilizers()
        self.circuit = self._build_circuit()

        self.simulation_count = 0
        self.failure_count = 0
        self.output_error_rate = 0.0

        if auto_run:
            self.run()

    def _validate_inputs(
        self,
        parity_check_x_matrix,
        parity_check_z_matrix,
        logical_x_matrix,
        logical_z_matrix,
        depol_error_rate,
        decoder_config,
        target_simulations,
        disable_progress,
        save_interval_sec,
        random_seed,
    ) -> None:
        if not isinstance(parity_check_x_matrix, (np.ndarray, sp.csr_matrix)):
            raise ValueError("parity_check_x_matrix must be a NumPy array or a SciPy sparse matrix.")
        if not isinstance(parity_check_z_matrix, (np.ndarray, sp.csr_matrix)):
            raise ValueError("parity_check_z_matrix must be a NumPy array or a SciPy sparse matrix.")
        if not isinstance(logical_x_matrix, (np.ndarray, sp.csr_matrix)):
            raise ValueError("logical_x_matrix must be a NumPy array or a SciPy sparse matrix.")
        if not isinstance(logical_z_matrix, (np.ndarray, sp.csr_matrix)):
            raise ValueError("logical_z_matrix must be a NumPy array or a SciPy sparse matrix.")
        if not (isinstance(depol_error_rate, float) and 0 <= depol_error_rate <= 1):
            raise ValueError("depol_error_rate must be a float between 0 and 1.")
        if not isinstance(decoder_config, dict):
            raise ValueError("decoder_config must be a dictionary.")
        if not (isinstance(target_simulations, int) and target_simulations > 0):
            raise ValueError("target_simulations must be a positive integer.")
        if not isinstance(disable_progress, bool):
            raise ValueError("disable_progress must be a boolean.")
        if not (isinstance(save_interval_sec, int) and save_interval_sec > 0):
            raise ValueError("save_interval_sec must be a positive integer.")
        if random_seed is not None and not isinstance(random_seed, int):
            raise ValueError("random_seed must be an integer.")

    def _initialize_stabilizers(self):
        """Initializes stabilizers for the CSS code resource state.

        This method constructs the tableau for the CSS code resource state and
        converts it into stabilizers.

        Returns:
            list: A list of stabilizers derived from the tableau.
        """
        tableau = construct_css_resource_state(
            self.parity_check_x_matrix.toarray(),
            self.parity_check_z_matrix.toarray(),
            self.logical_x_matrix.toarray(),
            self.logical_z_matrix.toarray()
        )
        combined_tableau = tableau + tableau  # Duplicate tableau for initialization
        return combined_tableau.to_stabilizers()

    def _build_circuit(self):
        """Builds the circuit for Bell measurements in entanglement distillation.

        The circuit measures Bell stabilizers (XX and ZZ) for each qubit pair.

        Returns:
            stim.Circuit: The constructed Bell measurement circuit.
        """
        circuit = stim.Circuit()
        for qubit_index in range(self.num_total_qubits):
            target_qubit = qubit_index + self.num_total_qubits
            circuit.append("MXX", [qubit_index, target_qubit])
            circuit.append("MZZ", [qubit_index, target_qubit])
        return circuit

    def run(self):
        """Executes Monte Carlo simulations.

        Returns:
            dict: Results of the simulation, including logical error rate and failure count.
        """
        self.start_date = datetime.datetime.now().strftime("%A, %B %d, %Y %H:%M:%S")
        progress_bar = tqdm(
            range(self.simulation_count + 1, self.target_simulations + 1),
            disable=self.disable_progress, ncols=0
        )

        for self.simulation_count in progress_bar:
            simulator = stim.TableauSimulator()
            simulator.set_state_from_stabilizers(self.initial_stabilizers)
            simulator.depolarize1(*range(self.num_physical_qubits), p=self.depol_error_rate)
            simulator.do_circuit(self.circuit)

            measurement_data = np.array(simulator.current_measurement_record(), dtype=np.uint8)
            xx_input, zz_input, xx_output, zz_output = self._partition_measurements(measurement_data)

            x_syndrome = self.parity_check_x_matrix @ xx_input % 2
            decoded_z_errors = self.x_parity_decoder.decode(x_syndrome)

            z_syndrome = self.parity_check_z_matrix @ zz_input % 2
            decoded_x_errors = self.z_parity_decoder.decode(z_syndrome)

            final_x_phase, final_z_phase = self._apply_error_corrections(
                xx_input, zz_input, xx_output, zz_output, decoded_x_errors, decoded_z_errors
            )

            self.failure_count += np.count_nonzero(np.bitwise_or(final_x_phase, final_z_phase))
            self.output_error_rate = self.failure_count / (self.num_logical_qubits * self.simulation_count)
            self.standard_deviation = np.sqrt(
                self.output_error_rate * (1 - self.output_error_rate) / self.simulation_count
            )

            progress_bar.set_description(
                f"Physical error rate: {100 * self.depol_error_rate:.2f}%; Logical error rate: "
                f"{100 * self.output_error_rate:.2f} ± {100 * self.standard_deviation:.2f}%"
            )

        return self.save_results()

    def _partition_measurements(self, measurement_data):
        """Splits measurement records into input and output phases for X and Z."""
        xx_input = measurement_data[::2][:self.num_physical_qubits]
        zz_input = measurement_data[1::2][:self.num_physical_qubits]
        xx_output = measurement_data[::2][self.num_physical_qubits:]
        zz_output = measurement_data[1::2][self.num_physical_qubits:]
        return xx_input, zz_input, xx_output, zz_output

    def _apply_error_corrections(self, x_input, z_input, x_output, z_output, decoded_x_errors, decoded_z_errors):
        """Applies corrections to logical X and Z phases."""
        logical_x_phase = self.logical_x_matrix @ x_input % 2
        logical_z_phase = self.logical_z_matrix @ z_input % 2

        logical_x_correction = self.logical_x_matrix @ decoded_z_errors % 2
        logical_z_correction = self.logical_z_matrix @ decoded_x_errors % 2

        final_x_phase = (x_output + logical_x_phase + logical_x_correction) % 2
        final_z_phase = (z_output + logical_z_phase + logical_z_correction) % 2
        return final_x_phase, final_z_phase

    def save_results(self) -> Dict:
        """Saves the results of the simulation.

        Returns:
            dict: Simulation results including logical error rate and failure count.
        """
        return {
            "output_error_rate": self.output_error_rate,
            "standard_deviation": self.standard_deviation,
            "input_error_rate": self.depol_error_rate,
            "simulation_count": self.simulation_count,
            "failure_count": self.failure_count,
        }
    
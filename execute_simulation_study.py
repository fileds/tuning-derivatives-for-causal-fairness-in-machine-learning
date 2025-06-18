import numpy as np
import sys
import time
import tempfile
import shutil

from simulation_study.simulation_study import serial_execution, parallel_execution, StatusPrinter
from simulation_study.SimulationParameters import select_parameters

def read_arguments(argv):
    # Read arguments
    # Import parameters
    if len(argv) < 3:
        print('Please provide the DGP, number of cores, and number of replicates as argument\n')
        print('Example:\npython execute_simulation_study.py linear 1 2\n')
        sys.exit()

    verbose = False if 'silent' in argv else True
    # Select parameters
    params = select_parameters(
        dgp=argv[1], 
        num_cores=int(argv[2]), 
        n_replicates=int(argv[3]), 
        seed=42,
        torch_threads=1,
        write_batch_size=100,
        verbose=verbose)

    return params

# Read arguments
params = read_arguments(sys.argv)

# Setup printer
status_printer = StatusPrinter(time.time(), params)
if params.verbose:
    status_printer.print_start_message()
    status_printer.print_status_header()

# Run simulation
if params.num_cores > 1:
    parallel_execution(params, status_printer, num_cores=params.num_cores, write_batch_size=params.write_batch_size)
else:
    serial_execution(params, status_printer, write_batch_size=params.write_batch_size)

# Print total time elapsed
if params.verbose: 
    status_printer.print_end_message()

# Safely prepend total simulation time
try:
    # Create a temporary file
    with tempfile.NamedTemporaryFile('w', delete=False) as temp_file:
        # Write the total time at the to
        temp_file.write(f'# Total time elapsed: {status_printer.get_t_elapsed()}\n')
        
        # Read existing data and write it to the temp file
        with open(params.results_file_name, 'r') as original_file:
            shutil.copyfileobj(original_file, temp_file)
    
    # Replace the original file with the temp file
    shutil.move(temp_file.name, params.results_file_name)

except Exception as e:
    print(f"An error occurred: {e}")
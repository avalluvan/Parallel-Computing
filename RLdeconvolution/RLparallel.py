# Import third party libraries
import numpy as np
from mpi4py import MPI
import h5py

# Define the number of rows and columns
NUMROWS = 50
NUMCOLS = 50

MASTER = 0      # Indicates master process

def load_response_matrix(comm, start_row, end_row, filename='example.h5'):
    with h5py.File(filename, "r", driver="mpio", comm=comm) as f:
        
        # Assuming the dataset name is "response_matrix"
        dataset = f["response_matrix"]
        R = dataset[start_row:end_row, :]

    return R

def load_response_matrix_transpose(comm, start_col, end_col, filename='example.h5'):
    with h5py.File(filename, "r", driver="mpio", comm=comm) as f:

        # Assuming the dataset name is "response_matrix"
        dataset = f["response_matrix"]
        RT = dataset[:, start_col:end_col]

    return RT

def load_sky_model():
    # Load the sky model (M) with values from 1 to NUMCOLS
    M = np.arange(1, NUMCOLS + 1, dtype=np.float64)
    return M

def main():
    # Set up MPI
    comm = MPI.COMM_WORLD
    numtasks = comm.Get_size()
    taskid = comm.Get_rank()

    # Initialise vectors required by all processes
    M = np.empty(NUMCOLS, dtype=np.float64)     # Will be loaded and broadcasted by master. 
    epsilon = np.zeros(NUMROWS)                 # Will be "All gatherv-ed".

# ****************************** MPI ******************************

# **************************** Part I *****************************

    # print(f"taskid {taskid}, arr {arr}")

    '''*************** Master ***************'''
    if taskid == MASTER:
        print(f"Hello from Master. Starting with {numtasks} threads.")

        # Initialise C vector. Only master requires full length.
        C = np.empty(NUMCOLS, dtype=np.float64)

        # Load sky model input
        M = load_sky_model()

    '''*************** Worker ***************'''

    if taskid > MASTER:
        # Only separate if... clause for NON-MASTER in this toy model
        # Initialise C vector to None. Only master requires full length.
        C = None

    '''**************** All *****************'''

    # Broadcast M vector
    comm.Bcast([M, MPI.DOUBLE], root=MASTER)

    # Calculate the indices in Rij that the process has to parse. My hunch is that calculating these scalars individually will be faster than the MPI send broadcast overhead
    averow = NUMROWS // numtasks
    extra_rows = NUMROWS % numtasks
    start_row = taskid * averow
    end_row = (taskid + 1) * averow if taskid < (numtasks - 1) else NUMROWS

    epsilon_slice = np.zeros(end_row - start_row)

    # Calculate epsilon slice
    R = load_response_matrix(comm, start_row, end_row)
    epsilon_slice = np.dot(R, M)      # XXX: Can be GPU accelerated

    # All vector gather epsilon slices
    recvcounts = [averow] * (numtasks-1) + [averow + extra_rows]
    displacements = np.arange(numtasks) * averow
    comm.Allgatherv(epsilon_slice, [epsilon, recvcounts, displacements, MPI.DOUBLE])

    # Sanity check: print epsilon
    if taskid == 0:
        print(epsilon)
        print()

# **************************** Part II *****************************
    
    # Calculate the indices in Rji that the process has to parse.
    avecol = NUMCOLS // numtasks
    extra_cols = NUMCOLS % numtasks
    start_col = taskid * avecol
    end_col = (taskid + 1) * avecol if taskid < (numtasks - 1) else NUMCOLS

    # Calculate C slice
    RT = load_response_matrix_transpose(comm, start_col, end_col)
    C_slice = np.dot(RT.T, 1/epsilon)

    # All vector gather C slices
    recvcounts = [avecol] * (numtasks-1) + [avecol + extra_cols]
    displacements = np.arange(numtasks) * avecol
    comm.Gatherv(C_slice, [C, recvcounts, displacements, MPI.DOUBLE], root=MASTER)

    # Sanity check: print C
    if taskid == 0:
        print(C)
        print()
        print("DONE")

    # MPI Shutdown
    MPI.Finalize()

if __name__ == "__main__":
    main()

# Import third party libraries
import numpy as np
from mpi4py import MPI
import h5py

# Define the number of rows and columns
NUMROWS = 50
NUMCOLS = 50

# Define MPI and iteration misc variables
MASTER = 0      # Indicates master process
MAXITER = 10   # Maximum number of iterations

'''
Response matrix
'''
def load_response_matrix(comm, start_row, end_row, filename='example.h5'):
    with h5py.File(filename, "r", driver="mpio", comm=comm) as f:
        # Assuming the dataset name is "response_matrix"
        dataset = f["response_matrix"]
        R = dataset[start_row:end_row, :]
    return R

'''
Response matrix transpose
'''
def load_response_matrix_transpose(comm, start_col, end_col, filename='example.h5'):
    with h5py.File(filename, "r", driver="mpio", comm=comm) as f:
        # Assuming the dataset name is "response_matrix"
        dataset = f["response_matrix"]
        RT = dataset[:, start_col:end_col]
    return RT

'''
Response matrix summed along axis=i
'''
def load_axis0_summed_response_matrix(comm):
    R = load_response_matrix(comm=comm, start_row=0, end_row=NUMROWS)
    Rj = np.sum(R, axis=0)               # TODO: This should be precomputed and stored in a .h5 file
    return Rj

'''
Sky model
'''
def load_sky_model():
    M0 = np.arange(1, NUMCOLS + 1, dtype=np.float64)      # --> Old test case
    # M0 = np.ones(NUMCOLS, dtype=np.float64)                 # <-- New test case
    return M0

'''
Observed data
'''
def load_obs_counts():
    d0 = np.ones(NUMROWS, dtype=np.float64)               # --> Old test case
    # d0 = np.zeros(NUMROWS, dtype=np.float64)                # <-- New test case
    d0[0] = 1
    return d0

def main():
    # Set up MPI
    comm = MPI.COMM_WORLD
    numtasks = comm.Get_size()
    taskid = comm.Get_rank()

    # Initialise vectors required by all processes
    M = np.empty(NUMCOLS, dtype=np.float64)     # Will be loaded and broadcasted by master. 
    d = np.empty(NUMROWS, dtype=np.float64)     # Will be loaded and broadcasted by master. 
    epsilon = np.zeros(NUMROWS)                 # Will be "All gatherv-ed".

    # Calculate the indices in Rij that the process has to parse. My hunch is that calculating these scalars individually will be faster than the MPI send broadcast overhead
    averow = NUMROWS // numtasks
    extra_rows = NUMROWS % numtasks
    start_row = taskid * averow
    end_row = (taskid + 1) * averow if taskid < (numtasks - 1) else NUMROWS

    # Calculate the indices in Rji that the process has to parse.
    avecol = NUMCOLS // numtasks
    extra_cols = NUMCOLS % numtasks
    start_col = taskid * avecol
    end_col = (taskid + 1) * avecol if taskid < (numtasks - 1) else NUMCOLS

# ****************************** MPI ******************************

# **************************** Part I *****************************

    # Set up initial values for iterating variables

    '''*************** Master ***************'''

    if taskid == MASTER:
        # Pretty print definitions
        linebreak_stars = '**********************'
        linebreak_dashes = '----------------------'

        # Initialise C vector. Only master requires full length.
        C = np.empty(NUMCOLS, dtype=np.float64)

        # Load sky model input
        M = load_sky_model()

        # Load observed data counts
        d = load_obs_counts()

    '''*************** Worker ***************'''

    if taskid > MASTER:
        # Only separate if... clause for NON-MASTER in this toy model
        # Initialise C vector to None. Only master requires full length.
        C = None

    # print(f"taskid {taskid}, arr {arr}")

# **************************** Part IIa *****************************

    # Calculate epsilon vector and all gatherv

    # Loop over iterations
    # Exit if
    ## 1. Max iterations are reached
    ## 2. M vector converges
    for iter in range(MAXITER):

        '''*************** Master ***************'''
        if taskid == MASTER:
            # Pretty print - starting
            print(linebreak_dashes)
            print(f"Starting iteration {iter + 1}")
            print(linebreak_dashes)

        '''**************** All *****************'''

        # Broadcast M vector
        comm.Bcast([M, MPI.DOUBLE], root=MASTER)

        # Initialise epsilon_slice
        epsilon_slice = np.zeros(end_row - start_row)

        # Calculate epsilon slice
        R = load_response_matrix(comm, start_row, end_row)
        epsilon_slice = np.dot(R, M)      # XXX: Can be GPU accelerated

        '''Allgatherv'''
        # All vector gather epsilon slices
        recvcounts = [averow] * (numtasks-1) + [averow + extra_rows]
        displacements = np.arange(numtasks) * averow
        comm.Allgatherv(epsilon_slice, [epsilon, recvcounts, displacements, MPI.DOUBLE])

        # Sanity check: print epsilon
        if taskid == MASTER:
            print(epsilon)
            print()

# **************************** Part IIb *****************************

    # Calculate C vector and gatherv
    
        '''**************** All *****************'''

        # Broadcast d vector
        comm.Bcast([d, MPI.DOUBLE], root=MASTER)

        # Initialise C_slice
        C_slice = np.zeros(end_col - start_col)

        # Calculate C slice
        RT = load_response_matrix_transpose(comm, start_col, end_col)
        C_slice = np.dot(RT.T, 1/epsilon)       # XXX: Can be GPU accelerated
        # C_slice = np.dot(RT.T, d/epsilon - 1)       # XXX: Can be GPU accelerated

        '''Gatherv'''
        # All vector gather C slices
        recvcounts = [avecol] * (numtasks-1) + [avecol + extra_cols]
        displacements = np.arange(numtasks) * avecol
        comm.Gatherv(C_slice, [C, recvcounts, displacements, MPI.DOUBLE], root=MASTER)

# **************************** Part IIb *****************************

    # Update M vector

        # Sanity check: print C
        if taskid == MASTER:
            print(C)
            print()

        # Iterative update of M vector
        if taskid == MASTER:

            # Pretty print - completion
            print(linebreak_dashes)
            print(f"Completed iteration {iter + 1}")
            print(linebreak_dashes)
            print(linebreak_stars)

    # MPI Shutdown
    MPI.Finalize()

if __name__ == "__main__":
    main()

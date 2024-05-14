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
    with h5py.File(filename, "r", driver="mpio", comm=comm) as f1:
        # Assuming the dataset name is "response_matrix"
        dataset = f1["response_matrix"]
        R = dataset[start_row:end_row, :]
    return R

'''
Response matrix transpose
'''
def load_response_matrix_transpose(comm, start_col, end_col, filename='example.h5'):
    with h5py.File(filename, "r", driver="mpio", comm=comm) as f1:
        # Assuming the dataset name is "response_matrix"
        dataset = f1["response_matrix"]
        RT = dataset[:, start_col:end_col]
    return RT

'''
Response matrix summed along axis=i
'''
def load_axis0_summed_response_matrix(filename='example_axis0_summed.h5'):
    with h5py.File(filename, "r") as f2:
        # Assuming the dataset name is "response_vector"
        dataset = f2["response_vector"]
        Rj = dataset[:]
    return Rj

'''
Sky model
'''
def load_sky_model():
    M0 = np.ones(NUMCOLS, dtype=np.float64)                 # Initial guess
    return M0

'''
Observed data
'''
def load_obs_counts():
    d0 = np.zeros(NUMROWS, dtype=np.float64)                # Point-source injection
    randint = np.random.randint(0, NUMROWS)
    d0[randint] = 1
    return d0

def main():
    # Set up MPI
    comm = MPI.COMM_WORLD
    numtasks = comm.Get_size()
    taskid = comm.Get_rank()

    # Initialise vectors required by all processes
    M = np.empty(NUMCOLS, dtype=np.float64)     # Loaded and broadcasted by master. 
    d = np.empty(NUMROWS, dtype=np.float64)     # Loaded and broadcasted by master. 
    epsilon = np.zeros(NUMROWS)                 # All gatherv-ed.
    epsilon_BG = 1e-3                           # Fudge parameter to avoid 0/0 error

    # Calculate the indices in Rij that the process has to parse. My hunch is that calculating these scalars individually will be faster than the MPI send broadcast overhead.
    averow = NUMROWS // numtasks
    extra_rows = NUMROWS % numtasks
    start_row = taskid * averow
    end_row = (taskid + 1) * averow if taskid < (numtasks - 1) else NUMROWS

    # Calculate the indices in Rji, i.e., Rij transpose, that the process has to parse.
    avecol = NUMCOLS // numtasks
    extra_cols = NUMCOLS % numtasks
    start_col = taskid * avecol
    end_col = (taskid + 1) * avecol if taskid < (numtasks - 1) else NUMCOLS

# ****************************** MPI ******************************

# **************************** Part I *****************************

    '''*************** Master ***************'''

    if taskid == MASTER:
        # Pretty print definitions
        linebreak_stars = '**********************'
        linebreak_dashes = '----------------------'

        # Load sky model input
        M = load_sky_model()

        # Load observed data counts
        d = load_obs_counts()

        # Sanity check: print d
        print()
        print('Observed data counts d vector:')
        print(d)
        ## Pretty print
        print()
        print(linebreak_stars)

        # Initialise C vector. Only master requires full length.
        C = np.empty(NUMCOLS, dtype=np.float64)

        # Initialise update delta vector
        delta = np.empty(NUMCOLS, dtype=np.float64)

    '''*************** Worker ***************'''

    if taskid > MASTER:
        # Only separate if... clause for NON-MASTER processes. 
        # Initialise C vector to None. Only master requires full length.
        C = None

    # print(f"taskid {taskid}, arr {arr}")

# **************************** Part IIa *****************************

    '''***************** Begin Iterative Segment *****************'''
    # Set up initial values for iterating variables.
    # Exit if:
    ## 1. Max iterations are reached
    ## 2. M vector converges
    for iter in range(MAXITER):

        '''*************** Master ***************'''
        if taskid == MASTER:
            # Pretty print - starting
            print(f"Starting iteration {iter + 1}")


    # Calculate epsilon vector and all gatherv

        '''**************** All *****************'''

        '''Synchronization Barrier 1'''
        # Broadcast M vector
        comm.Bcast([M, MPI.DOUBLE], root=MASTER)

        # Initialise epsilon_slice
        epsilon_slice = np.zeros(end_row - start_row)

        # Calculate epsilon slice
        R = load_response_matrix(comm, start_row, end_row)
        epsilon_slice = np.dot(R, M) + epsilon_BG      # XXX: Can be GPU accelerated

        '''Synchronization Barrier 2'''
        # All vector gather epsilon slices
        recvcounts = [averow] * (numtasks-1) + [averow + extra_rows]
        displacements = np.arange(numtasks) * averow
        comm.Allgatherv(epsilon_slice, [epsilon, recvcounts, displacements, MPI.DOUBLE])

        # Sanity check: print epsilon
        # if taskid == MASTER:
        #     print('epsilon')
        #     print(epsilon)
        #     print()

# **************************** Part IIb *****************************

    # Calculate C vector and gatherv
    
        '''**************** All *****************'''

        '''Synchronization Barrier 3'''
        # Broadcast d vector
        comm.Bcast([d, MPI.DOUBLE], root=MASTER)

        # Initialise C_slice
        C_slice = np.zeros(end_col - start_col)

        # Calculate C slice
        RT = load_response_matrix_transpose(comm, start_col, end_col)
        C_slice = np.dot(RT.T, d/epsilon)       # TODO: Can be GPU accelerated

        '''Synchronization Barrier 4'''
        # All vector gather C slices
        recvcounts = [avecol] * (numtasks-1) + [avecol + extra_cols]
        displacements = np.arange(numtasks) * avecol
        comm.Gatherv(C_slice, [C, recvcounts, displacements, MPI.DOUBLE], root=MASTER)

# **************************** Part IIb *****************************

    # Update M vector

        # Sanity check: print C
        # if taskid == MASTER:
        #     print('C')
        #     print(C)
        #     print()

        # Iterative update of M vector
        if taskid == MASTER:
            # Load Rj vector (response matrix summed along axis=i)
            Rj = load_axis0_summed_response_matrix()
            delta = C / Rj - 1
            M = M + delta * M           # Allows for optimization features presented in Siegert et al. 2020

            # Sanity check: print M
            # print('M')
            # print(M)

            # Pretty print - completion
            print(f"Done")
            print(linebreak_dashes)

            # MAXITER
            if iter == (MAXITER - 1):
                print(f'Reached maximum iterations = {MAXITER}')
                print(linebreak_stars)
                print()
    
    '''****************** End Iterative Segment ******************'''

    # Print converged M
    if taskid == MASTER:
        print('Converged M vector:')
        print(np.round(M, 2))
        print()

    # MPI Shutdown
    MPI.Finalize()

if __name__ == "__main__":
    main()

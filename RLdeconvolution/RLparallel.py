# Import third party libraries
import numpy as np
from mpi4py import MPI
import h5py

# Define the number of rows and columns
NUMROWS = 50
NUMCOLS = 50

MASTER = 0
FROM_MASTER = 1  # Message tag
FROM_WORKER = 2  # Message tag

def load_response_matrix(comm, start_row, end_row, filename='example.h5'):
    with h5py.File(filename, "r", driver="mpio", comm=comm) as f:
        # Assuming the dataset name is "response_matrix"
        dataset = f["response_matrix"]
        R = dataset[start_row:end_row, :]
    return R

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

    # Calculate the indices in Rij that the node has to parse. My hunch is that calculating these scalars individually will be faster than the MPI send broadcast overhead
    averow = NUMROWS // numtasks
    extra_rows = NUMROWS % numtasks
    start_row = taskid * averow
    end_row = (taskid + 1) * averow if taskid < (numtasks - 1) else NUMROWS   # Provision to eventually remove criteria 1
    # print(f"taskid {taskid}, endrow {end_row}")

    epsilon_slice = np.zeros(end_row - start_row)

    if taskid == MASTER:
        print(f"Hello from Master. Starting with {numtasks} threads.")

        # Initialise C vector. Only master requires full length.
        C = np.zeros(NUMCOLS)

        # Load sky model input
        M = load_sky_model()

    comm.Bcast([M, MPI.DOUBLE], root=MASTER)

    # Calculate epsilon slice
    R = load_response_matrix(comm, start_row, end_row)
    epsilon_slice = np.dot(R, M)      # XXX: Can be GPU accelerated

    # All gather epsilon slices
    recvcounts = [averow] * (numtasks-1) + [averow + extra_rows]
    displacements = np.arange(numtasks) * averow
    comm.Allgatherv(epsilon_slice, [epsilon, recvcounts, displacements, MPI.DOUBLE])

    if taskid == 0:
        print(epsilon)
        print()

    # if taskid == MASTER:
    #     for dest in range(1, numworkers + 1):
    #         mtype = FROM_WORKER
    #         offset_cols = comm.recv(source=dest, tag=mtype)
    #         cols = comm.recv(source=dest, tag=mtype)
    #         C[offset_cols : offset_cols + cols] = comm.recv(source=dest, tag=mtype)
    #     print(C)

    # elif taskid > MASTER:
    #     # Calculate C slice
    #     for j in range(cols):
    #         for i in range(NUMROWS):
    #             C[j] += R[i, offset_cols + j] / epsilon[i]

    #     # Send C slice back to master
    #     comm.send(offset_cols, dest=MASTER, tag=FROM_WORKER)
    #     comm.send(cols, dest=MASTER, tag=FROM_WORKER)
    #     comm.send(C[:cols], dest=MASTER, tag=FROM_WORKER)

    print("Almost Done\n")
    MPI.Finalize()
    print("Done")

if __name__ == "__main__":
    main()

import numpy as np
from mpi4py import MPI
import h5py

# Define the number of rows and columns
NUMROWS = 50
NUMCOLS = 50

MASTER = 0
FROM_MASTER = 1  # Message tag
FROM_WORKER = 2  # Message tag

def init_response_matrix():
    # Initialize the response matrix R
    R = np.zeros((NUMROWS, NUMCOLS))
    np.fill_diagonal(R, 2.0)  # Set diagonal elements to 2.0
    return R

def load_response_matrix(comm, start_row, end_row, filename='example.h5'):
    with h5py.File(filename, "r", driver="mpio", comm=comm) as f:
        # Assuming the dataset name is "response_matrix"
        dataset = f["response_matrix"]
        R = dataset[start_row:end_row, :]
    return R

def load_sky_model():
    # Load the sky model (M) with values from 1 to NUMCOLS
    M = np.arange(1, NUMCOLS + 1)
    return M

def main():
    comm = MPI.COMM_WORLD
    numtasks = comm.Get_size()
    taskid = comm.Get_rank()

    if numtasks < 2:
        print("Need at least two MPI tasks. Quitting...")
        MPI.Abort(MPI.COMM_WORLD, 1)
        exit(1)

    numworkers = numtasks - 1

    epsilon = np.zeros(NUMROWS)
    C = np.zeros(NUMCOLS)

    if taskid == 0:
        print(f"Hello from Master. Starting with {numworkers} worker threads.")

        # Send matrix data to the worker tasks
        averow = NUMROWS // numworkers
        extra_rows = NUMROWS % numworkers

        avecol = NUMCOLS // numworkers
        extra_cols = NUMCOLS % numworkers

        offset_rows = 0
        offset_cols = 0

        # Task distribution loop
        for dest in range(1, numworkers + 1):
            # Load sky model input
            M = load_sky_model()

            if dest <= extra_rows:
                rows = averow + 1
            else:
                rows = averow

            if dest <= extra_cols:
                cols = avecol + 1
            else:
                cols = avecol

            # Send data to worker
            mtype = FROM_MASTER
            comm.send(offset_rows, dest=dest, tag=mtype)
            comm.send(rows, dest=dest, tag=mtype)
            comm.send(offset_cols, dest=dest, tag=mtype)
            comm.send(cols, dest=dest, tag=mtype)
            comm.send(M, dest=dest, tag=mtype)

            offset_rows = offset_rows + rows
            offset_cols = offset_cols + cols

        for dest in range(1, numworkers + 1):
            mtype = FROM_WORKER
            offset_rows = comm.recv(source=dest, tag=mtype)
            rows = comm.recv(source=dest, tag=mtype)
            epsilon[offset_rows : offset_rows + rows] = comm.recv(source=dest, tag=mtype)

        print(epsilon)
        print()

    elif taskid > 0:
        # Receive index range for R and R-transpose
        mtype = FROM_MASTER  # Message tag
        offset_rows = comm.recv(source=0, tag=mtype)
        rows = comm.recv(source=0, tag=mtype)
        offset_cols = comm.recv(source=0, tag=mtype)
        cols = comm.recv(source=0, tag=mtype)
        M = np.empty(NUMCOLS, dtype=np.float64)
        M = comm.recv(source=0, tag=mtype)

        # Calculate epsilon slice
        R = init_response_matrix()
        for i in range(rows):
            for j in range(NUMCOLS):
                epsilon[i] += R[offset_rows + i, j] * M[j]

        # Send epsilon slice back to master
        comm.send(offset_rows, dest=0, tag=FROM_WORKER)
        comm.send(rows, dest=0, tag=FROM_WORKER)
        comm.send(epsilon[:rows], dest=0, tag=FROM_WORKER)

    comm.Bcast(epsilon, root=MASTER)

    if taskid == 0:
        for dest in range(1, numworkers + 1):
            mtype = FROM_WORKER
            offset_cols = comm.recv(source=dest, tag=mtype)
            cols = comm.recv(source=dest, tag=mtype)
            C[offset_cols : offset_cols + cols] = comm.recv(source=dest, tag=mtype)
        print(C)

    elif taskid > 0:
        # Calculate C slice
        for j in range(cols):
            for i in range(NUMROWS):
                C[j] += R[i, offset_cols + j] / epsilon[i]

        # Send C slice back to master
        comm.send(offset_cols, dest=0, tag=FROM_WORKER)
        comm.send(cols, dest=0, tag=FROM_WORKER)
        comm.send(C[:cols], dest=0, tag=FROM_WORKER)

if __name__ == "__main__":
    main()

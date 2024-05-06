// Import standard libraries
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

// Adapted from code mpi_mm.c https://hpc-tutorials.llnl.gov/mpi/examples/mpi_mm.c

// Needs to be listed before runtime or inputted from a configuration file
#define NUMROWS 1000      /* Number of rows in response matrix.
                            Also equal to the number of rows in epsilon (intermediate matrix)
                            and the number of rows in the expected / observed data */
#define NUMCOLS 1000      /* Number of columns in response matrix
                            Also equal to the number of rows in the input / predicted sky model */

// Message Passing definitions
#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

void init_response_matrix(double R[NUMROWS][NUMCOLS])
{
    // This is a very simple definition for Rij. Only i=j terms are set to nonzero.
    // FIXME: If NUMROWS > NUMCOLS then too many terms are missing which can result in nans in C vector
    int i, j; // Iterator variables
    for (i = 0; i < NUMROWS; i++)
        for (j = 0; j < NUMCOLS; j++)
            R[i][j] = (i == j) ? 2.0 : 0;
}

void load_sky_model(double M[NUMCOLS])
{
    int j; // Iterator variable
    for (j = 0; j < NUMCOLS; j++)
        M[j] = j + 1;
}

int main(int argc, char *argv[])
{
int numtasks, // number of tasks in partition
    taskid,   // a task identifier
    numworkers, // number of worker tasks
    source,     // task id of message source
    dest,       // task id of message destination
    mtype,      // message type
    rows,       // indices (rows) of matrix R to be worked on by each worker
    cols,       // indices (rows in R-transpose) of matrix R
	averow, extra_rows, offset_rows, /* used to determine rows sent to each worker */
	avecol, extra_cols, offset_cols, /* used to determine cols sent to each worker */
    rc,                     // Abort status flag
    i, j;                   // Iterator variables

double	R[NUMROWS][NUMCOLS],            // Response matrix
        M[NUMCOLS],                     // sky model
        epsilon[NUMROWS],               // intermediate vector
        C[NUMCOLS];                     // intermediate vector

MPI_Status status;          // Status flag for MPI_RECV calls

MPI_Init(&argc,&argv);      // Initialise MPI
MPI_Comm_rank(MPI_COMM_WORLD, &taskid);      // What is my task ID?
MPI_Comm_size(MPI_COMM_WORLD, &numtasks);    // How many tasks are there?

if (numtasks < 2 ) {
  printf("Need at least two MPI tasks. Quitting...\n");
  MPI_Abort(MPI_COMM_WORLD, rc);
  exit(1);
  }
numworkers = numtasks - 1;

//**************************** master task part 1 ************************************

    if (taskid == MASTER)
    {
        printf("Hello from Master. Starting with %d worker threads.\n", numworkers);

        /* Send matrix data to the worker tasks */
        averow = NUMROWS / numworkers;      /* Will send either averow or averow+1 number of rows to each worker
                                        depending on partitioning of the response matrix */
        extra_rows = NUMROWS % numworkers;       /* One extra row that will be sent to the first <extra> workers */

        avecol = NUMCOLS / numworkers;
        extra_cols = NUMCOLS % numworkers;

        offset_rows = 0;                     // Offset is incremented at the end of each iteration of the task distribution loop
        offset_cols = 0;

        mtype = FROM_MASTER;            // Message tag

        // Distributing tasks one-by-one -- the task distribution loop
        // Variable `dest` used to disseminate task to correct worker
        for (dest=1; dest <= numworkers; dest++)
        {
            // Load sky model input
            load_sky_model(M);

            if (dest <= extra_rows) 
                rows = averow + 1;
            else
                rows = averow;

            if (dest <= extra_cols) 
                cols = avecol + 1;
            else
                cols = avecol;

            // printf("Sending %d rows to task %d offset=%d\n", rows, dest, offset_rows);
            // TODO: Replace blocking sends with non-blocking sends
            MPI_Send(&offset_rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&offset_cols, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&cols, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&M, NUMCOLS, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);

            offset_rows = offset_rows + rows;
            offset_cols = offset_cols + cols;
        }

        // Receive epsilon slices
        mtype = FROM_WORKER;
        for (i=1; i <= numworkers; i++)
        {
            source = i;
            MPI_Recv(&offset_rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&epsilon[offset_rows], rows, MPI_DOUBLE, source, mtype, 
                    MPI_COMM_WORLD, &status);
            // printf("Received first round of results from task %d\n",source);
        }
    }

//**************************** worker task part 1 ************************************

    if (taskid > MASTER)
    {
        // printf("Hello from Worker %d\n", taskid);

        // Receive index range for R and R-transpose
        mtype = FROM_MASTER;        // Message tag
        // TODO: Replace blocking receives with non-blocking receives
        MPI_Recv(&offset_rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&offset_cols, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&cols, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&M, NUMCOLS, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);      // TODO: Receive only relevant slice of M from master instead of full vector

        // Use offset and rows to extract relevant slice from response_matrix.h5
        init_response_matrix (R); 
        // TODO: Change from basic implementation to actually importing data
        for (i=0; i < rows; i++)
            {
            epsilon[i] = 0;
            for (j=0; j < NUMCOLS; j++)
                epsilon[i] += R[offset_rows + i][j] * M[j];
                // TODO: offset_rows will not be necessary if only relevant slice of R is sent. 
            }

        // TODO: Replace send and receive with MPI_ALLGATHER
        mtype = FROM_WORKER;
        MPI_Send(&offset_rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&epsilon, rows, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    }

    // Relay full epsilon -- effectively a sync barrier. Will be removed once epsilon is sent via MPI_GATHERALL
    MPI_Bcast(&epsilon, NUMROWS, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);





//**************************** master task part 2 ************************************

    if (taskid == MASTER)
    {
        // Receive C slices
        mtype = FROM_WORKER;
        for (i=1; i <= numworkers; i++)
        {
            source = i;
            MPI_Recv(&offset_cols, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&cols, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&C[offset_cols], cols, MPI_DOUBLE, source, mtype, 
                    MPI_COMM_WORLD, &status);
            // printf("Received second round of results from task %d\n",source);
        }
        // printf("\nPrinting epsilon vector\n");
        // for (j=0; j < NUMROWS; j++)
        // {
        //     printf("%f\n", epsilon[j]);
        // }

        // printf("\nPrinting C vector\n");
        // for (j=0; j < NUMCOLS; j++)
        // {
        //     printf("%f\n", C[j]);
        // }

        printf("Done\n");
    }


//**************************** worker task part 2 ************************************

    if (taskid > MASTER)
    {
        // printf("Restarting the engine, taskid: %d\n", taskid);
        for (j=0; j < cols; j++)
            {
            C[j] = 0;
            for (i=0; i < NUMROWS; i++)
                C[j] += R[i][offset_cols + j] / epsilon[i];
                // TODO: offset_cols will not be necessary if only relevant slice of R is sent. 
            }

        // TODO: Replace send and receive with MPI_ALLGATHER
        mtype = FROM_WORKER;
        MPI_Send(&offset_cols, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&cols, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&C, cols, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    }


//*************************** exit program ************************************

MPI_Finalize();

}
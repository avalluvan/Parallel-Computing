#include <stdio.h>
#include <stdlib.h>
#include "hdf5.h"
#include "H5FDmpio.h"
#include <mpi.h>

#define MASTER 0

// Function to read row-wise chunks in parallel
void read_row_chunks_parallel(const char* filename, const char* dataset_name, MPI_Comm comm) {
    hid_t file_id, dataset_id, dataspace_id, hyperslab_id, fapl_id;
    hsize_t dims[2];
    herr_t status;
    int rank, size;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Create a file access property list (fapl) for parallel access
    fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id, comm, MPI_INFO_NULL);
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, fapl_id);
    if (file_id < 0) {
        fprintf(stderr, "Error opening HDF5 file: %s\n", filename);
        MPI_Abort(comm, 1);
    }

    // Only the master process opens the HDF5 file
    if (rank == MASTER) {

        // Open the dataset
        dataset_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
        if (dataset_id < 0) {
            fprintf(stderr, "Error opening dataset: %s\n", dataset_name);
            H5Fclose(file_id);
            MPI_Abort(comm, 1);
        }

        // Get the dimensions of the dataset
        dataspace_id = H5Dget_space(dataset_id);
        if (dataspace_id < 0) {
            fprintf(stderr, "Error getting dataspace for dataset: %s\n", dataset_name);
            H5Dclose(dataset_id);
            H5Fclose(file_id);
            MPI_Abort(comm, 1);
        }
        H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
    }

    // Broadcast the dimensions to all processes
    MPI_Bcast(&file_id, 1, MPI_UNSIGNED_LONG_LONG, MASTER, comm);
    MPI_Bcast(&dataset_id, 1, MPI_UNSIGNED_LONG_LONG, MASTER, comm);
    MPI_Bcast(dims, 2, MPI_UNSIGNED_LONG_LONG, MASTER, comm);

    // Determine the chunk size for each process
    int numworkers = size - 1;
    hsize_t chunk_size = dims[0] / numworkers;      /* The chunk size is calculated based on the number of worker processes 
                                                    and the total number of rows in the dataset (dims[0]).
                                                    Each process will read a chunk of rows from the dataset. */
    // printf("Dimensions %llu, %llu", dims[0], dims[1]);   // Correctly prints 50, 50
    hsize_t start[2] = {(rank - 1) * chunk_size, 0};  // start[0]: starting row, start[1]: starting column (always 0)
    hsize_t count[2] = {chunk_size, dims[1]};   // FIXME: Needs to be made variable based on divisibility of #rows by #worker processes

    // Create a memory hyperslab for each process
    hyperslab_id = H5Screate_simple(2, count, NULL);
    /* hid_t H5Screate_simple (int          rank,
                            const hsize_t 	dims[],
                            const hsize_t 	maxdims[] )		
    Creates a new simple dataspace and opens it for access.
    Parameters
    [in]	rank	Number of dimensions of dataspace
    [in]	dims	Array specifying the size of each dimension
    [in]	maxdims	Array specifying the maximum size of each dimension */
    H5Sselect_hyperslab(hyperslab_id, H5S_SELECT_SET, start, NULL, count, NULL);

    // Allocate memory for the chunk (2D array of floats)
    float** chunk_data = (float**)malloc(count[0] * sizeof(float*));
    for (hsize_t i = 0; i < count[0]; ++i) {
        chunk_data[i] = (float*)malloc(count[1] * sizeof(float));
    }

    // Read data from the dataset (workers only)
    if (rank != MASTER) {
        status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, chunk_data[0]);
        // Process the chunk (e.g., compute statistics, etc.)
        // ...
        // Print the data
        for (int i = 0; i < dims[0]; ++i) {
            for (int j = 0; j < dims[1]; ++j) {
                // printf("%.2f ", *chunk_data[i * dims[1] + j]);
            }
            printf("\n");
        }
        printf("DEBUG %d\n", rank);
    }

    // Clean up
    for (hsize_t i = 0; i < count[0]; ++i) {
        free(chunk_data[i]);
    }
    free(chunk_data);
    H5Sclose(hyperslab_id);
    if (rank == MASTER) {
        H5Dclose(dataset_id);
        H5Fclose(file_id);
    }
    H5Pclose(fapl_id);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    const char* filename = "example.h5";
    const char* dataset_name = "response_matrix";

    MPI_Comm comm = MPI_COMM_WORLD;
    read_row_chunks_parallel(filename, dataset_name, comm);

    MPI_Finalize();
    return 0;
}

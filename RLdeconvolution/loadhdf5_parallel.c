#include <stdio.h>
#include <stdlib.h>
#include "hdf5.h"
#include "H5FDmpio.h"
#include "mpi.h"

float* read_hdf5_dataset(const char *filename, const char *dataset_name, hsize_t *dims,
                         MPI_Comm comm) // TODO: Is there a better way of passing MPI_COMM_WORLD
                                                        // TODO: I want to pass info by reference
{
    hid_t file_id, dataset_id, dataspace_id,
          fapl_id, xf_id;           // TODO: Add comments about their purpose
    herr_t status;
    float* data;

    // int numtasks, taskid;
    // MPI_Comm_rank(comm, &taskid);      // What is my task ID?
    // MPI_Comm_size(comm, &numtasks);    // How many tasks are there?

    // Create a file access property list
    fapl_id = H5Pcreate(H5P_FILE_ACCESS);

    // Sets driver for files on parallel file systems (MPI I/O)
    status = H5Pset_fapl_mpio(fapl_id, comm, MPI_INFO_NULL); // This line is the source of the error
                                                // TODO: Change MPI_INFO_NULL to appropriate MPI_Info object
    if (status < 0) {
        printf("Error setting file access property list\n");
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }

    // Open the HDF5 file
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, fapl_id);
    if (file_id < 0) {
        printf("Error opening file: %s\n", filename);
        return NULL;
    }

    // Open the dataset
    dataset_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
    if (dataset_id < 0) {
        printf("Error opening dataset: %s\n", dataset_name);
        H5Fclose(file_id);
        return NULL;
    }

    // Set xf for parallel HDF5
    // xf_id = H5Pcreate(H5P_DATASET_XFER);
    // H5Pset_dxpl_mpio(xf_id, H5FD_MPIO_COLLECTIVE);

    // Get the dataspace
    dataspace_id = H5Dget_space(dataset_id);
    status = H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
    if (status < 0) {
        printf("Error getting dataspace dimensions\n");
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }

    // Allocate memory for the data
    data = (float*)malloc(dims[0] * dims[1] * sizeof(float));

    // Read the data
    status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    if (status < 0) {
        printf("Error reading dataset\n");
        free(data);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }

    // Clean up (do not close the file here)
    // FIXME: Where is H5Fclose(file_id); supposed to be called before exiting program?
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    return data;
}

int main(int argc, char *argv[]) 
{
    int numtasks, // number of tasks in partition
        taskid;   // a task identifier

    MPI_Status status;          // Status flag for MPI_RECV calls
    MPI_Init(&argc,&argv);      // Initialise MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);      // What is my task ID?
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);    // How many tasks are there?

    const char* filename = "example.h5";
    const char* dataset_name = "/response_matrix";
    hsize_t dims[2];
    float* data = read_hdf5_dataset(filename, dataset_name, dims, MPI_COMM_WORLD);

    if (data) {
        // Print the data (you can modify this part as needed)
        for (int i = 0; i < dims[0]; ++i) {
            printf("%d | ", taskid);
            for (int j = 0; j < dims[1]; ++j) {
                printf("%.2f ", data[i * dims[1] + j]);
            }
            printf("\n");
        }

        // Clean up
        free(data);
    } else {
        printf("Error loading data from HDF5 file.\n");
    }

    MPI_Finalize();

    return 0;
}

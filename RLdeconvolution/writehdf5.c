// Import standard libraries
// Code source: https://github.com/HDFGroup/hdf5-tutorial/blob/main/00-CPP-101.ipynb
#include <stdio.h>
#include <stdlib.h>
#include "hdf5.h"

#define NUMROWS 5000
#define NUMCOLS 5000

int main(int argc, char *argv[])
{
    unsigned majnum, minnum, relnum;
    H5get_libversion(&majnum, &minnum, &relnum);
    printf("Hello, HDF5 library %d.%d.%d!\n", majnum, minnum, relnum);

    // If the user provided a filename, create a file under that name
    // otherwise, create a file called "hello.h5"
    hid_t file = H5Fcreate("hello.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    H5Fclose(file);

    return 0;
}
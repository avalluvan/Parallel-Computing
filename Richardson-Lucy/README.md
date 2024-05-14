# MPI4py-based Parallel Implementation of the Richardson-Lucy Algorithm

## Directory Contents

Richardson-Lucy is abbreviated to RL.

- [RLparallel.ipynb](RLparallel.ipynb): generates a simple tridiagonal matrix and creates the example.h5 and example_axis0_summed.h5 HDF5 files.
- [RLparallel.py](RLparallel.py): main function for the parallel RL implementation. In its current form, it simulates the signal data vector $d_i$, 
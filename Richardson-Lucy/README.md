# MPI4py-based Parallel Implementation of the Richardson-Lucy Algorithm

## Directory Contents

Richardson-Lucy is abbreviated to RL.

- [RLparallel.ipynb](RLparallel.ipynb): generates a simple tridiagonal matrix and creates the example.h5 and example_axis0_summed.h5 HDF5 files. If `NUMROWS` or `NUMCOLS` are modified, please make the relevant changes in the RLparallel.py function too. 
- [RLparallel.py](RLparallel.py): main function for the parallel RL implementation. It spawns worker processes through the relevant MPI calls,  the master process facilitates the transfer of intermediate vectors $\epsilon_i$ and $C_j$, and $M_j^{(k)}$, and facilitates parallel reads of example.h5. In its current form, the signal data vector $d_i$ is simulated, and the initial guess $M_j{(0)}$ is arbitrarily defined. Although the code implementation is indifferent to a general $p\times q, p\neq q$ matrix, I currently do not have a suitable test case for this. 
- [example.h5](example.h5): contains the response matrix $R_{ij}$ as a HDF5 dataset `['response_matrix']`. 
- [example_axis0_summed.h5](example_axis0_summed.h5): contains the vector $\sum_i R_{ij}$. This is stored as a separate file as $R_{ij}$ does not change at each iteration and the sum operation (over `NUMROWS` rows) can be avoided. 

## Dependencies
- `numpy`
- `mpi4py`
- `h5py` with parallel read access enabled. It is enabled by default in the standard installation [[source](https://docs.h5py.org/en/latest/mpi.html)]. 

## Execution

To execute the pipeline on a local computer, ensure that the python environment supports parallel read access. 
```
$ conda activate <venv>       # activate your python environment
$ export TMPDIR=/tmp          # truncation can occur on MacOS with the default TMPDIR
$ mpiexec -n <numproc> python RLparallel.py     # run the code with the intended number of nodes
```
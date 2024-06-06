# Richardson-Lucy Toy Model

## Directory Contents

Richardson-Lucy is abbreviated to RL.

- [RLparallel.ipynb](Richardson-Lucy/RLparallel.ipynb): generates a simple tridiagonal matrix and creates the example.h5 and example_axis0_summed.h5 HDF5 files. If `NUMROWS` or `NUMCOLS` are modified, please make the relevant changes in the RLparallel.py function too. 
- [RLparallel.py](Richardson-Lucy/RLparallel.py): main function for the parallel RL implementation. It spawns worker processes through the relevant MPI calls,  the master process facilitates the transfer of intermediate vectors $\epsilon_i$ and $C_j$, and $M_j^{(k)}$, and facilitates parallel reads of example.h5. In its current form, the signal data vector $d_i$ is simulated, and the initial guess $M_j{(0)}$ is arbitrarily defined. Although the code implementation is indifferent to a general $p\times q, p\neq q$ matrix, I currently do not have a suitable test case for this. 
- [example.h5](Richardson-Lucy/example.h5): contains the response matrix $R_{ij}$ as a HDF5 dataset `['response_matrix']`. 
- [example_axis0_summed.h5](Richardson-Lucy/example_axis0_summed.h5): contains the vector $\sum_i R_{ij}$ as a HDF5 dataset `['response_vector']`. This is stored as a separate file as $R_{ij}$ does not change at each iteration and the sum operation (over `NUMROWS` rows) can be avoided. 

## Dependencies
- `numpy`
- `mpi4py`
- `h5py` with parallel read access enabled. It is enabled by default in the standard installation [[source](https://docs.h5py.org/en/latest/mpi.html)]. 

## Executing on Expanse

```
$ cd /expanse/lustre/scratch/$USER/temp_project
$ salloc --nodes=1 --ntasks-per-node=4 -A csd759 -t 0:05:00 -p shared       # request for node allocation
$ module reset                                                              # reset your environment and load the required modules. Thank you Dr. Tatineni!
$ module load gcc/10.2.0
$ module load openmpi/4.1.3
$ module load python/3.8.12
$ module load py-mpi4py/3.1.2
$ module load hdf5/1.10.7
$ module load py-numpy/1.20.3
$ module load py-h5py/3.4.0
$ mpiexec -n 4 python ~/Richardson-Lucy/code/toymodel/toy_RLparallel.py		# works perfectly out-of-the-box. Code will run in its own directory.
```

## Executing on a Personal Computer

To execute the pipeline on a local computer, ensure that the python environment supports parallel read access and that your computer has at least two cores (although the code itself can be run on a single core). 
```
$ conda activate <venv>       # activate your python environment
$ export TMPDIR=/tmp          # truncation can occur on MacOS with the default TMPDIR
$ mpiexec -n <numproc> python toy_RLparallel.py     # run the code with the intended number of nodes
```
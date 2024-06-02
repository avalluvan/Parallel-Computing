# MPI4py-based Parallel Implementation of the Richardson-Lucy Algorithm

## Directory Contents

Richardson-Lucy is abbreviated to RL.

- [datapreprocessing.py](code/datapreprocessing.py): runs a file check. If unsuccessful, the script downloads missing files from the COSIpy server and preprocesses them. [input.yaml](code/input.yaml) is a dependency.
- [input.yaml](code/input.yaml): configuration file for [datapreprocessing.py](code/datapreprocessing.py).
- [RLparallel.py](code/RLparallel.py): main function for the parallel RL implementation. It spawns worker processes through the relevant MPI calls,  the master process facilitates the transfer of intermediate vectors $\epsilon_i$ and $C_j$, and $M_j^{(k)}$, and facilitates parallel reads of the response matrix. In its current form, the observed data vector $d_i$ combines a simulated point source signal and full-sky background model, and the initial guess $M_j{(0)}$ is defined to $10^{-4}$ counts. NUMROWS and NUMCOLS need to be predefined, although a later version may support directly reading the shape of the response matrix dataset. 
- [datavisualization.ipynb](code/datavisualization.ipynb): python notebook to visualize code inputs and outputs and facilitates a comparison with current COSIpy implementation.

## Dependencies
- `numpy`
- `mpi4py`
- `h5py` with parallel read access enabled. It is enabled by default in the standard installation [[source](https://docs.h5py.org/en/latest/mpi.html)]. 
- `cosipy` and `histpy` are required to run [datapreprocessing.py](code/datapreprocessing.py). The main function does not require these libraries.

## Execution

To execute the pipeline on a local computer, ensure that the h5py installation in your python environment supports parallel read access. 
```
$ conda activate <venv>       # activate your python environment
$ export TMPDIR=/tmp          # truncation can occur on MacOS with the default TMPDIR
$ python3 datapreprocessing.py                  # to check if all file dependencies are satisfied or download them
$ mpiexec -n <numproc> python RLparallel.py     # run the code with the intended number of nodes
```
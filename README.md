# MPI4py-based Parallel Implementation of the Richardson-Lucy Algorithm
Documentation and code repository for PHYS 244: Parallel Computing

## Directory Contents

- [assets](assets): hosts auxiliary files such as the image files of the results shown below.
- [code](code): contains python scripts for a point-source toy model, $^{44}$Ti (point-source) and positron (diffuse-source) imaging analyses.
- [docs](docs): references used to write the code.

## Dependencies
- `numpy`
- `mpi4py`
- `h5py`
- `cosipy` and `histpy` (optional)

## Result
The code successfully recovers the injected signal of 3 supernovae, Cassiopeia A, G1.9+0.3 (Galactic Center), and SN1987A (Large Magellanic Cloud) in just 12 seconds. This is a 5x speed-up with respect to the code in the COSIpy library, which took ~1 minute. The parallel implementation also circumvents the need to load extremely large onto memory.
![image](assets/44Timap.png)
# Phonopy calculations to compute the harmonic phonon free energy and entropy

The python scripts provided herein are used to 
1. optimize the snapshots from the NVT trajectories (`opt.py`)
2. calculate the hessians (`calc_hessian.py`)
3. run the phonopy calculation given the hessian as an input (`run_phonopy.py`)

* Optimized Structures are given in `.mfpx` and `.cif` file format.
* the phonopy.out files in each folder contain the result of the respective calculation 3.
* Hessian `.pickle` and molden `.freq` files are not included in this repository due to their file size. They can be obtained per request. 

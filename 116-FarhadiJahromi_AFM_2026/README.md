Supporting Data for "Predicting Atomic Charges in MOFs by Topological Charge Equilibration"
====

This repository contains code and data for the publication "Predicting Atomic Charges in MOFs by Topological Charge Equilibration" by Babak Farhadi Jahromi Sumukh Shankar Sharadaprasad & Rochus Schmid published in Advanced Functional Materials, ...

Code
----

Prerequisite: [cmc-tools](https://github.com/MOFplus/cmc-tools).
* chargefitter.py contains the classes used for training fluctuating charge models in serial and in parallel.
* mpifit.py is the script used to run the TopoQEq training procedure in parallel.
    * Usage: `mpirun -np 64 python mpifit.py > mpifit.log` (for 64 processes)
* compile\_params.py compiles parameters from CMAES output and writes to file
    * Usage: `python compile_params.py outcmaes/xrecentbest.dat`

Data
---

Provided are:
* the curated QMOF database split into the archives `testset.zip` and `trainset.zip`
* the final TopoQEq parameters obtained from the training procedure for each atom typing scheme (`etypes.par`, `ctypes.par` and `atypes.par`)



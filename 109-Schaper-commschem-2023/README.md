Supporting Data for "Simulating the structural phase transitions of metal-organic frameworks with control over the volume of nanocrystallites"
====

This folder contains code and data for the paper "Simulating the structural phase transitions of metal-organic frameworks with control over the volume of nanocrystallites" by Larissa Schaper & Rochus Schmid published in Communication Chemistry volume 6, Article number: 233, 2023, DOI: [10.1038/s42004-023-01025-x](https://doi.org/10.1038/s42004-023-01025-x)

Code
----

In order to reproduce the calculations performed in this study, the lammps and pylmps version from the [cmc-tools](https://github.com/MOFplus/cmc-tools) repository should be employed. In addition, we have used an external potential for volume control of the finite size systems as an "expot" class, derived from the base class defined in pylmps. Here you can find the code for the external potential ncvol.
In order to accelerate the python code, the software package numba was added via pip to the specific conda environment for running all the [cmc-tools](https://github.com/MOFplus/cmc-tools).

The following python code is an example to run a simulation with lammps and the external potential ncvol for the NC4 system. It will be loaded from the corresponding mfpx/ric/par files. With ["zn2p", "co2"] the two fragments define the SBU for which a COM is computed and used to define the tetrahedra. With mode="US" the umbrella sampling mode (fixed k) is used:

```python
import molsys
import pylmps
import ncvol

pl = pylmps.pylmps("NC4")
m = molsys.mol.from_file("NC4.mfpx")
ncv = ncvol.ncvol(["zn2p", "co2"],mode='US',k=0.001,Vref = 60000 ,nstep=100)
callback = ncv.callback
pl.add_external_potential(ncv, callback="callback")
pl.setup(mol=m, ff='file', kspace=False, bcond=0, boundary=("m","m","m"))
pl.MD_init('run1', ensemble='nvt', T=300.0, thermo='ber', relax=[0.1], startup=True)
pl.MD_run(100)
```

To run steered molecular dynamics simulations with lammps and the external potential forcing a linear volume change during simulation, the above python code is modified by adding the reference volume attribute:

```python
import molsys
import pylmps
import ncvol

pl = pylmps.pylmps("NC4")
m = molsys.mol.from_file("NC4.mfpx")
ncv = ncvol.ncvol(["zn2p", "co2"],mode='US',k=0.001,Vref = 60000 ,nstep=100)
callback = ncv.callback
pl.add_external_potential(ncv, callback="callback")
pl.setup(mol=m, ff='file', kspace=False, bcond=0, boundary=("m","m","m"))
ncv.Vrefs=[60000.0,59000.0,100]
pl.MD_init('run1', ensemble='nvt', T=300.0, thermo='ber', relax=[0.1], startup=True)
pl.MD_run(100)
```
The reference volume attribute is a list of three elements containing the initial volume as first reference volume, the target volume as last refernce volume, and the stepsize indicating how many reference volumes are given between the initial and the traget volume. The stepsize has to be equal to the number of simulated steps.


Data
---

In this folder, you find input files for pylmps and the raw lammps input files for the different nanocrystallites investigated in this study. The initial structures are given as mfpx files. The MOF-FF parameter are defined in the corresponding .ric/.par files. Using pylmps, calcualtions can be performed starting from mfpx/ric/par. Note, that pylmps will write raw lammps input (in/data) on the fly, which can be used to investigate the systems without pylmps. However, the volume constraint (see /code folder) can only be used within pylmps, where ncvol is employed as an external potental (derived from expot class in pylmps).



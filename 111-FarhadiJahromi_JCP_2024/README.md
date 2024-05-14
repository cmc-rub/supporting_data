Supporting Data for "Dielectric Response of Metal-Organic Frameworks as a Function of Confined Guest Species Investigated by Molecular Dynamics Simulations"
====

This repository contains data and analysis scripts used in the publication "Dielectric Response of Metal-Organic Frameworks as a Function of Confined Guest Species Investigated by Molecular Dynamics Simulations" by Babak Farhadi Jahromi & Rochus Schmid published in the Journal of Chemical Physics Vol.160, Issue 28, 2024, DOI: [10.1063/5.0203820](https://doi.org/10.1063/5.0203820)

Code
----

In order to reproduce the calculations performed in this study, the lammps and pylmps version from the [cmc-tools](https://github.com/MOFplus/cmc-tools) repository should be employed.

Data
---

pylmps input files and the corresponding raw lammps input files for the investigated guest-molecule loaded MOFs are given. The initial coordinates are given in the in-house .mfpx format. The employed MOF-FF parameters are defined in the corresponding .ric/.par files. Using pylmps, calcualtions can be performed starting from mfpx/ric/par. Note, that pylmps will write raw lammps input (in/data) on the fly, which can be used to investigate the systems without pylmps.



# Supporting information

Supporting information for: [“Evaluating Computational Shortcuts in Supercell-Based Phonon Calculations of Molecular Crystals: The Instructive Case of Naphthalene”](https://doi.org/10.1021/acs.jctc.0c00119), Tomas Kamencek, Sandro Wieser, Hirotaka Kojima, Natalia Bedoya-Martínez, Johannes P. Dürholt, Rochus Schmid, and Egbert Zojer 

This paper was posted as a [preprint on arXiv(https://arxiv.org/abs/2002.02689).

The three folders contain the phonon input data (geometry in POSCAR format, harmonic force constants and phonopy input files "band.conf") for each level of theory addressed in the paper:
	DFT: Density-functional theory calculations varying the used a posteriori van der Waals correction (TS, D3-BJ and D2)
	     For each vdW correction a full optimization of atomic coodinates and the unit cell parameters has been performed.
	     Also containing VASP input files (KPOINTS, POSCARs, INCARs, POTCARs) for the single point calculations carried out to calculate the phonons.
	DFTB: Density functional based tight binding calculations varying the unit cell of the naphthalene crystal.
		1. DFTB_pure: unit cell optimized within DFTB level of theory
		2. DFTB@DFT: unit cell taken from VASP (PBE/D3-BJ) optimization, only optimization of atomic coordinates
		3. DFTB95%DFT: unit cell of VASP scaled by a factor of 0.95 in order to minimize the RMS-error in frequencies compared to the PBE/D3-BJ reference
		Also including input files to set up the necessary single point calculations in DFTB+
	FF: Force field calculations varying the force field (all geometries and unit cells have been optimized using the respective force field)
		1. COMPASS
		2. our parametrization of the MOF-FF
		3. GAFF
		Also including the input files for LAMMPS

In order to reproduce phonons with the given POSCARS and FORCE_CONSTANTS, please consult the phonopy manual: https://phonopy.github.io/phonopy/setting-tags.html

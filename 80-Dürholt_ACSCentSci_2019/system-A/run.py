#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pydlpoly
import molsys
#  initialize a pydlpoly instance
m = molsys.mol()
m.read("bdc_0opt.mfpx")

m.addon("ff")
m.ff.read("bdc_0")

pd = pydlpoly.pydlpoly("test")
pd.control["shift"] = ""
pd.control["spme"] = "precision 1.0d-6"
pd.setup(mol=m, local = False)

# set field
pd.set_efield([0.00, 0.0, 0.0], use_ref = False, const_D = False)
pd.set_atoms_moved()
pd.calc_energy_force()


# equilibrate temperature
pd.MD_init("equil", T=150,  ensemble="nvt", thermo = "ber", relax=[0.2])
pd.MD_run(100000, printout=100)

# sampling
pd.MD_init("cons", T=150,  ensemble="nvt", traj=["dipole"], tnstep=[10], thermo = "hoover", relax=[1.0])
pd.MD_run(10000000, printout=100)

pd.end()

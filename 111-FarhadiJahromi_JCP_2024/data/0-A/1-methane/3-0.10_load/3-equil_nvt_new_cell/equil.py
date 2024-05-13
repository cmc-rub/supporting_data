import molsys
import pylmps


pl = pylmps.pylmps("A_methane_010")
m = molsys.mol.from_file("A_methane_010.mfpx")
m.addon("ff")
m.ff.read("A_methane_010")
pl.setup(mol=m, bcond=2, local=True, kspace_style="pppm")
pl.MD_init("equil_ber", T=298.15, startup=True, ensemble="nvt", thermo="ber", relax=[0.2], traj=["xyz", "vel"], tnstep=1000, rnstep=100)
pl.MD_run(500000, printout=100)
pl.MD_init("equil_nh", T=298.15, startup=False, ensemble="nvt", thermo="hoover", relax=[1.0], traj=["xyz", "vel"], tnstep=1000, rnstep=100)
pl.MD_run(500000, printout=100)
pl.end()

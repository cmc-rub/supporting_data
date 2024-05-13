import molsys
import pylmps


pl = pylmps.pylmps("B_methanol_010")
pl.setup(bcond=2, local=True, kspace_style="pppm", restart='equil_nh', restart_vel=True)
pl.command("fix exfield all efield 0.020 0.000 0.000")
pl.MD_init("equil_e", T=298.15, startup=False, ensemble="nvt", thermo="hoover", relax=[1.0], traj=["xyz"], tnstep=100, rnstep=100)
pl.MD_run(500000, printout=100)
pl.MD_init("nvt_e", T=298.15, startup=False, ensemble="nvt", thermo="hoover", relax=[1.0], traj=["xyz"], tnstep=100, rnstep=100)
pl.MD_run(10000000, printout=100)
pl.end()

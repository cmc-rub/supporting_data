import molsys
import pylmps


pl = pylmps.pylmps("B_methanol_100")
pl.setup(bcond=2, local=True, kspace_style="pppm", restart='equil_nh', restart_vel=True)
pl.MD_init("nvt_e0", T=298.15, startup=False, ensemble="nvt", thermo="hoover", relax=[1.0], traj=["xyz"], tnstep=100, rnstep=100)
pl.MD_run(50000000, printout=100)
pl.end()

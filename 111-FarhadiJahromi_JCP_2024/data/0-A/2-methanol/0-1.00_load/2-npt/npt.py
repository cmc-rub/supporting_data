import molsys
import pylmps


pl = pylmps.pylmps("A_methanol_100")
pl.setup(bcond=2, local=True, kspace_style="pppm", restart='equil_nh', restart_vel=True)
pl.MD_init("equil_npt", p=1.0, T=298.15, startup=False, ensemble="npt", thermo="hoover", relax=[1.0,1.0], traj=["cell", "xyz"], tnstep=1000, rnstep=1000)
pl.MD_run(500000, printout=100)
pl.MD_init("cell", p=1.0, T=298.15, startup=False, ensemble="npt", thermo="hoover", relax=[1.0,1.0], traj=["cell", "xyz"], tnstep=100, rnstep=100)
pl.MD_run(2000000, printout=100)
pl.end()

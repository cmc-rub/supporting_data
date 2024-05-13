import molsys
import pylmps


pl = pylmps.pylmps("B_methanol_025")
m = molsys.mol.from_file("B_methanol_025.mfpx")
m.addon("ff")
m.ff.read("B_methanol_025")
pl.setup(mol=m, bcond=2, local=True)

pl.MIN(0.1, maxiter=100, maxeval=1000)

pl.LATMIN_boxrel(0.1, 0.1, maxiter=1000, maxeval=10000)
pl.LATMIN_sd(0.1, 0.1, lat_maxiter=100, maxiter=1000)
pl.LATMIN_sd(0.05, 0.05, lat_maxiter=100, maxiter=1000)

pl.MIN(0.01, maxiter=100, maxeval=1000)

pl.LATMIN_boxrel(0.01, 0.01, maxiter=1000, maxeval=10000, maxstep=10)
pl.LATMIN_sd(0.01, 0.01, lat_maxiter=1000, maxiter=10000)
pl.LATMIN_sd(0.005, 0.005, lat_maxiter=1000, maxiter=10000)

pl.MIN(0.001, maxiter=1000, maxeval=10000)

pl.LATMIN_boxrel(0.001, 0.001, maxiter=1000, maxeval=10000, maxstep=10)
pl.LATMIN_sd(0.001, 0.001, lat_maxiter=1000, maxiter=10000)
pl.LATMIN_sd(0.0005, 0.0005, lat_maxiter=1000, maxiter=10000)

pl.MIN(0.0001, maxiter=1000, maxeval=10000)

pl.LATMIN_boxrel(0.0001, 0.0001, maxiter=1000, maxeval=10000, maxstep=10)
pl.LATMIN_sd(0.0001, 0.0001, lat_maxiter=1000, maxiter=10000)
pl.LATMIN_sd(0.00005, 0.00005, lat_maxiter=1000, maxiter=10000)

pl.MIN(0.00001, maxiter=1000, maxeval=10000)

m.set_xyz(pl.get_xyz())
m.set_cell(pl.get_cell())
m.write("B_methanol_025_opt.mfpx")

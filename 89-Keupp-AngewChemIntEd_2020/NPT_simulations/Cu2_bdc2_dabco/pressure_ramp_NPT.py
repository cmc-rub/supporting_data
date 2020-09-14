import pylmps
import molsys

T=300.0
P=5000   # lammps real units: [P] = atm

# these are the pylmps defaults
Trelax = 100
Prelax = 1000

# read the structure and force field
name = 'IAST_112'
filename = name +'.mfpx'
m = molsys.mol.from_file(filename)
m.addon('ff')
m.ff.read(name)

# lammps wrapper setup
pl = pylmps.pylmps(name)
pl.setup(mol=m,kspace=True)

# equilibration
pl.MD_init('equil',ensemble='npt',
                  p=0.0,
                  T=T,
                  thermo='mttk',
                  startup=True,
                  mttk_volconstraint='no',
                  )
pl.MD_run(101000)

# production
pl.MD_init('produ',ensemble='npt',
                  T=T,
                  p=[0.0,P],
                  thermo='mttk',
                  startup=False,
                  mttk_volconstraint='no',
                  )
pl.MD_run(10000000)


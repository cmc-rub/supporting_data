import os
import sys
import pylmps
import molsys

# get the name of the structure file in this folder
name = [x.rsplit('.',1)[0] for x in os.listdir('.') if x.rsplit('.',1)[-1] == 'ric' ][0]

# read the structure and force field
filename = name +'.mfpx'
m = molsys.mol.from_file(filename)
m.addon('ff')
m.ff.read(name)

# lammps wrapper setup
pl = pylmps.pylmps(name)
pl.setup(mol=m,kspace=True)

T=300.0
Trelax = 0.1
P=0.0
Prelax = 1.0
nsteps = 1000000

# equilibration
pl.MD_init('equil',ensemble='npt',
                  T=300.0,
                  p=P,
                  thermo='mttk',
                  relax=(Trelax,Prelax),
                  tnstep=10000,
                  startup=True,
                  mttk_volconstraint='yes',
                  )
pl.MD_run(100000)

# production
pl.MD_init('produ',ensemble='npt',
                  T=300.0,
                  p=P,
                  thermo='mttk',
                  relax=(Trelax,Prelax),
                  tnstep=500,
                  startup=False,
                  mttk_volconstraint='yes',
                  )
pl.MD_run(nsteps)


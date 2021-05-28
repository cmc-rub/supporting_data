import numpy
import os
import sys
import pylmps
import molsys

name = [x.rsplit('.',1)[0] for x in os.listdir('.') if x.rsplit('.',1)[-1] == 'ric' ][0]
filename = name +'.mfpx'
numpy.random.seed(4000010)
try:
    m = molsys.mol.fromFile(filename)
except:
    m = molsys.mol.from_file(filename)
m.addon('ff')
m.ff.read(name)

pl = pylmps.pylmps(name)
pl.setup(mol=m,kspace=True)

T=300.0
Trelax = 0.1
P=0.0
Prelax = 1.0
nsteps = 1000000

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




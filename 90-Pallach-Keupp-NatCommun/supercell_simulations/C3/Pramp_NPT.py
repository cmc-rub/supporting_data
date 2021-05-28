
import numpy
import os
import sys
import pylmps
import molsys

T=300.0
Tequil = 300.0
P=2500   # lammps real units: [P] = atm

relax= [0.1,1.0] # thermostat, barostat

numpy.random.seed(1600332275)


name = [x for x in os.listdir('.') if x.rsplit('.')[-1] == 'mfpx']
if len(name) != 1:
    print('ERROR, multiple mfpx files found')
    exit()
name = name[0].rsplit('.',1)[0]

filename = name +'.mfpx'

m = molsys.mol.from_file(filename)
m.addon('ff')
m.ff.read(name)

pl = pylmps.pylmps(name)
pl.setup(mol=m,kspace=True)


pl.MD_init('equil_NVT_high',ensemble='nvt',
                  T=Tequil,
                  thermo='hoover',
                  relax = [relax[0]],
                  tnstep=1000,
                  startup=True,
                  )
pl.MD_run(250000)

pl.MD_init('equil_npt',ensemble='npt',
                  p=0.0,
                  T=T,
                  thermo='mttk',
                  startup=False,
                  relax=relax,
                  tnstep=1000,
                  mttk_volconstraint='yes',
                  )
pl.MD_run(100000)

pl.MD_init('produ_Pup',ensemble='npt',
                  T=T,
                  p=[0.0,P],
                  thermo='mttk',
                  startup=False,
                  relax=relax,
                  tnstep=500,
                  mttk_volconstraint='no',
                  )
pl.MD_run(2500000)

pl.MD_init('produ_atP',ensemble='npt',
                  T=T,
                  p=P,
                  thermo='mttk',
                  startup=False,
                  relax=relax,
                  mttk_volconstraint='no',
                  )
pl.MD_run(500000)

pl.MD_init('produ_Pdown',ensemble='npt',
                  T=T,
                  p=[P,0.0],
                  thermo='mttk',
                  startup=False,
                  relax=relax,
                  tnstep=500,
                  mttk_volconstraint='no',
                  )
pl.MD_run(1000000)

pl.MD_init('produ_ambientP',ensemble='npt',
                  T=T,
                  p=0.0,
                  thermo='mttk',
                  startup=False,
                  relax=relax,
                  tnstep=500,
                  mttk_volconstraint='no',
                  )
pl.MD_run(1000000)

pl.MD_init('produ_Pminus',ensemble='npt',
                  T=T,
                  p=[0.0,-5000.0],
                  thermo='mttk',
                  startup=False,
                  relax=relax,
                  tnstep=500,
                  mttk_volconstraint='no',
                  )
pl.MD_run(1000000)

pl.MD_init('produ_atminus',ensemble='npt',
                  T=T,
                  p=-5000.0,
                  thermo='mttk',
                  startup=False,
                  relax=relax,
                  tnstep=500,
                  mttk_volconstraint='no',
                  )
pl.MD_run(1000000)













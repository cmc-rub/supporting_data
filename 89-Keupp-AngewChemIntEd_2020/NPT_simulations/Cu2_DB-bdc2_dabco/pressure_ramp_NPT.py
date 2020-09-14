
import os
import pylmps
import molsys

T=300.0
P=10000   # lammps real units: [P] = atm

# these are the default pylmps settings (in ps)
relax= [0.1,1.0] # thermostat, barostat

# get the name of the structure file in this folder
name = [x for x in os.listdir('.') if x.rsplit('.')[-1] == 'mfpx']

# read the structure and force field
name = name[0].rsplit('.',1)[0]
filename = name +'.mfpx'
m = molsys.mol.from_file(filename)
m.addon('ff')
m.ff.read(name)

# lammps wrapper setup
pl = pylmps.pylmps(name)
pl.setup(mol=m,kspace=True)

# equilibration 1/3
pl.MD_init('equil_NVT_ber',ensemble='nvt',
                  T=[10,T],
                  thermo='ber',
                  relax = [0.01],
                  tnstep=1000,
                  startup=True,
                  )
pl.MD_run(100000)

# equilibration 2/3
pl.MD_init('equil_NVT',ensemble='nvt',
                  T=T,
                  thermo='hoover',
                  relax = [relax[0]],
                  tnstep=1000,
                  startup=False,
                  )
pl.MD_run(1000000)

# equilibration 3/3
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

# production
pl.MD_init('produ',ensemble='npt',
                  T=T,
                  p=[0.0,P],
                  thermo='mttk',
                  startup=False,
                  relax=relax,
                  tnstep=100,
                  mttk_volconstraint='no',
                  )
pl.MD_run(10000000)



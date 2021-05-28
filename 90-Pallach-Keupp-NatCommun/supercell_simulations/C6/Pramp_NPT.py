
import numpy
import os
import sys
import pylmps
import molsys

T=300.0
P=5000   # lammps real units: [P] = atm

relax= [0.1,1.0] # thermostat, barostat

numpy.random.seed(1585903674)


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

pl.MD_init('equil_NVT_ber',ensemble='nvt',
                  T=[10,T],
                  thermo='ber',
                  relax = [0.01],
                  tnstep=1000,
                  startup=True,
                  startup_seed=1585903674,
                  )
pl.MD_run(100000)

pl.MD_init('equil_NVT',ensemble='nvt',
                  T=T,
                  thermo='hoover',
                  relax = [relax[0]],
                  tnstep=1000,
                  startup=False,
                  )
pl.MD_run(250000)

pl.MD_init('produ',ensemble='npt',
                  T=T,
                  p=[0.0,P],
                  thermo='mttk',
                  startup=False,
                  relax=relax,
                  tnstep=500,
                  mttk_volconstraint='no',
                  )
pl.MD_run(5000000)

















#pl.lmps.command('fix             1 all mttknhc temp ${temperature} ${temperature} ${tdamp} tri ${pressure} ${pressure} ${pdamp} volconstraint yes')
#pl.lmps.command('fix             1 all mttknhc temp %8.4f %8.4f %8.4f tri %12.6f %12.6f %12.6f volconstraint yes' % (T,T,Trelax,P,P,Prelax))
#pl.lmps.command('fix_modify      1 energy yes') # Add thermo/baro contributions to 

#pl.lmps.command('compute thermo_temp2 all temp')
#pl.lmps.command('compute thermo_press2 all pressure thermo_temp2')
#pl.lmps.command('dump ptens_dump all local 10 ptens.dump index thermo_press')
#pl.lmps.command('dump ptens_dump all custom 10 ptens.dump id pxx pyy pzz pxy pxz pyz')
#pl.lmps.command('thermo_style custom step ecoul elong ebond eangle edihed eimp pe\
#                ke etotal temp press vol cella cellb cellc cellalpha cellbeta cellgamma\
#                pxx pyy pzz pxy pxz pyz')

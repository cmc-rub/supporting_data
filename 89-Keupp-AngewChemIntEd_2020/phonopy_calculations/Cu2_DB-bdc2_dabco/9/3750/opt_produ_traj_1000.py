
import numpy
import os
import sys
import pylmps
import molsys

name = 'produ_traj_1000'
filename = name +'.mfpx'
try:
    m = molsys.mol.fromFile(filename)
except:
    m = molsys.mol.from_file(filename)
m.addon('ff')
m.ff.read(name)

pl = pylmps.pylmps(name)
pl.setup(mol=m,kspace=True)

#pl.lmps.command('dielectric 4.0')

pl.MIN_cg(0.15)
pl.write(name+'_opt15.mfpx')
pl.MIN_cg(0.10)
pl.write(name+'_opt10.mfpx')
pl.MIN_cg(0.05)
pl.write(name+'_opt05.mfpx')
pl.MIN_cg(0.01)
pl.write(name+'_opt01.mfpx')
pl.MIN_cg(0.005)
pl.write(name+'_opt005.mfpx')
#pl.MIN_cg(0.001)
#pl.write(name+'_opt001.mfpx')
#pl.MIN_cg(0.0001)
#pl.write(name+'_opt0001.mfpx')

energies = pl.calc_energy()
print 'we are done!', energies
if pl.mpi_rank == 0:
    print 'I am done !', energies
    f = open(name+'.res','w')
    f.write(name +' '+ str(m.natoms)+' '+str(energies)+' '+str(energies/float(m.natoms)))
    f.close()


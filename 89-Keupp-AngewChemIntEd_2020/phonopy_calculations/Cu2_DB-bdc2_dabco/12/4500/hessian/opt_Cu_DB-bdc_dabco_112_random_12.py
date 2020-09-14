
import numpy
import os
import sys
import pylmps
import molsys

name = 'Cu_DB-bdc_dabco_112_random_12'
filename = name +'.mfpx'
try:
    m = molsys.mol.fromFile(filename)
except:
    m = molsys.mol.from_file(filename)
m.addon('ff')
m.ff.read(name)

pl = pylmps.pylmps(name)
pl.control['kspace'] = True
pl.setup(mol=m,kspace=True)


pl.MIN(0.1)  
pl.LATMIN_boxrel(0.1,0.1)
pl.LATMIN(0.1,0.1)
pl.LATMIN(0.05,0.05)
pl.LATMIN(0.01,0.01)
pl.MIN(0.001)  


energies = pl.calc_energy()
print 'we are done!', energies
pl.write(name+'_latopt.mfpx')
if pl.mpi_rank == 0:
    print 'I am done !', energies
    f = open(name+'.energy','w')
    f.write(name +' '+ str(m.natoms)+' '+str(energies)+' '+str(energies/float(m.natoms)))
    f.close()


import molsys.stow as stow
import pickle
import molsys
import pydlpoly
import pylmps
import hessutils
import shutil
import numpy
import sys
import os
import copy

name = [i.split('.')[0] for i in os.listdir('.') if i.split('.')[-1] == 'ric'][0]
mfpx = name+'.mfpx'
m = molsys.mol.from_file(mfpx)
m.addon('ff')
try:
    m.ff.read(name)
except:
    m.ff.read(name,fit=True)


pl = pylmps.pylmps(name)
pl.control['kspace'] = True
pl.setup(mol=m)


hess = hessutils.doublehessian(pl);
hessian = hess()
hu = hessutils.hessutils(pl.get_xyz(),hessian,pl.get_elements())
hu.write_molden_input(name+'.freq')

h =  hu.s_hessian
h2 = numpy.zeros((m.natoms,m.natoms,3,3),dtype='double')

for i in range(m.natoms):
    for j in range(m.natoms):
        i3,j3 = 3*i,3*j
        h2[i,j,:,:] = h[i3:i3+3,j3:j3+3]

pickle.dump(hessian,open('hessian.pickle','wb'))









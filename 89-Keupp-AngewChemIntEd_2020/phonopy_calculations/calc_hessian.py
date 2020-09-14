import pickle
import molsys
import pylmps
import hessutils
import numpy
import os

name = [i.split('.')[0] for i in os.listdir('.') if i.split('.')[-1] == 'ric'][0]
mfpx = name+'.mfpx'
m = molsys.mol.from_file(mfpx)
m.addon('ff')

m.ff.read(name)

# setup our in-house lammps wrapper pylmps
# https://github.com/MOFplus/pylmps
pl = pylmps.pylmps(name)
pl.control['kspace'] = True
pl.setup(mol=m)


hess = hessutils.doublehessian(pl);
# compute the hessian by finite differencing of the forces
hessian = hess()

hu = hessutils.hessutils(pl.get_xyz(),hessian,pl.get_elements())
# write a -freq file to be read by molden and alike programs
hu.write_molden_input(name+'.freq')
# pickle the hessian so it can be read from the script that does the actual phonopy calculation.
pickle.dump(hessian,open('hessian.pickle','wb'))









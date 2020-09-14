import molsys
import pickle
import time
import os
import sys

f = open('phonopy.out', 'w+')

# get the name of the system to run on. ric, par and mpfx have the same basename!
for names in os.listdir('.'):
    if names.rsplit('.',1)[-1] == 'par':
        name =  names.rsplit('.',1)[0]

filename = name +'.mfpx'

# https://github.com/MOFplus/molsys
m = molsys.mol.from_file(filename)

hessian = pickle.load(open('hessian.pickle','rb'))

phonon = m.to_phonopy(hessian=hessian)


mesh_size = 12

mesh_param = str(mesh_size)

print ('calculating mesh of size ' + mesh_param *3 + ' for ' + filename)

phonon.run_mesh([mesh_size, mesh_size, mesh_size])
phonon.run_thermal_properties(t_step=10,
                              t_max=1000,
                              t_min=0)
tp_dict = phonon.get_thermal_properties_dict()
temperatures = tp_dict['temperatures']
free_energy = tp_dict['free_energy']
entropy = tp_dict['entropy']
heat_capacity = tp_dict['heat_capacity']

if m.is_master:
    f.write('mesh size = ' + mesh_param + '\n')
    f.write('temperatures   free_energy      entropy        heat_capacity\n')
    for t, F, S, cv in zip(temperatures, free_energy, entropy, heat_capacity):
        print(("%12.3f " + "%15.7f" * 3) % ( t, F, S, cv ))
        f.write(("%12.3f " + "%15.7f" + "%15.7f" + "%15.7f\n") % ( t, F, S, cv ))

#phonon.plot_thermal_properties().show()

f.close()


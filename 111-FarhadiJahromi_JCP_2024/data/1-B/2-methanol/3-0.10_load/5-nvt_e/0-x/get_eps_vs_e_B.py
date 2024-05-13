import molsys
import numpy as np
import h5py
import pyblock

# old units.py style units:
meter = 1.0/0.5291772083e-10
angstrom = 1.0e-10*meter

def load_data(fname, stage, start, end):
    f = h5py.File(fname, "r")
    cha_pars = f["system"]["ff"]["par"]["cha"]["pars"][:,0]
    cha_rics = f["system"]["ff"]["ric"]["cha"][:,0]-1
    charges = np.array([cha_pars[i] for i in cha_rics])
    fragnumbers = f["system"]["fragnumbers"][:]
    fragtypes = f["system"]["fragtypes"][:]
    cell = f[stage]["restart"]["cell"][:]
    cell_diag = np.diag(cell)
    xyz_array = f[stage]["traj"]["xyz"][start:end]
    nimages = xyz_array.shape[0]

    m = molsys.mol.from_file(fname[:-4]+"mfpx")
    mols = m.get_separated_molecules()
    nmols = len(mols)
    natoms_per_mol = [len(i) for i in mols]
    mof_id = natoms_per_mol.index(max(natoms_per_mol))

    m.addon("fragments")
    m.fragments.add_frag_graph()
    m.fragments.merge_frags("base", "post", ["pyr", "phtalaz"], "rotor")
    m.fragments.merge_frags("post", "final", ["zn2p", "co2"], "sbu")
    fgraph = m.fragments.fgraphs["final"]
    fgtypes = np.array([ i for i in fgraph.vp.type])
    atom_maps = [i for i in fgraph.vp.atom_map]
    verts = fgraph.get_vertices()
    tgroups = []
    mask = fgraph.new_vertex_property("bool")
    mask.set_value(False)
    fgraph.set_vertex_filter(mask, inverted=True)
    for i, v in enumerate(verts[fgtypes=="sbu"]):
        tgroups.append([v])
        for nb in fgraph.iter_out_neighbors(v):
            if fgtypes[nb] == "rotor":
                tgroups[i].append(nb)
                mask[nb] = True
                break
        ph_neighbors = [nb for nb in fgraph.get_all_neighbors(v) if fgtypes[nb] == "ph"]
        coc_v = m.get_coc(atom_maps[v])
        dists = np.array([(m.get_coc(atom_maps[ph]) - coc_v) for ph in ph_neighbors])
        dists -= cell_diag*np.around(dists/cell_diag)
        idx = [i for i,d in enumerate(dists) if (d[0] > 0 and  d[1] > 0)][0]
        first_ph = ph_neighbors[idx]
        tgroups[i].append(first_ph)
        mask[first_ph] = True
        ph_neighbors = [nb for nb in fgraph.get_all_neighbors(v) if fgtypes[nb] == "ph"]
        if len(ph_neighbors) > 1:
            dists = np.array([(m.get_coc(atom_maps[ph]) - coc_v) for ph in ph_neighbors])
            dists -= cell_diag*np.around(dists/cell_diag)
            idx = [i for i,d in enumerate(dists) if (d[0] < 0 and  d[1] > 0)][0]
            second_ph = ph_neighbors[idx]
        else:
            second_ph = ph_neighbors[0]
        tgroups[i].append(second_ph)
        mask[second_ph] = True
    
    neutralfrags = []
    for g in tgroups:
        neutralfrags.append(np.concatenate([atom_maps[i] for i in g]))

    dipoles = np.zeros((nimages,nmols,3))
    for i in range(nmols):
        if i == mof_id:
            for mol_ids in neutralfrags:
                d = xyz_array[:,mol_ids]-xyz_array[:,np.newaxis,mol_ids[0]]
                mol_unwrap = xyz_array[:,mol_ids] - cell_diag*np.around(d/cell_diag)
                dipoles[:,i] += (charges[mol_ids,np.newaxis]*mol_unwrap).sum(axis=1)
        else:
            mol_ids = mols[i]
            d = xyz_array[:,mol_ids]-xyz_array[:,np.newaxis,mol_ids[0]]
            mol_unwrap = xyz_array[:,mol_ids] - cell_diag*np.around(d/cell_diag)
            dipoles[:,i] = (charges[mol_ids,np.newaxis]*mol_unwrap).sum(axis=1)
    dipole_per_image = {"mof":dipoles[:,mof_id], "guests":np.delete(dipoles, mof_id, axis=1).sum(axis=1)}
    f.close()
    return dipole_per_image, cell

if __name__ == "__main__":
    fields = []
    with open("xe.dat", "r") as f:
        for l in f.readlines():
            fields.append(l.split()[0])
    eps_0 = 8.854187817e-12
    dip, cell = load_data("../../4-nvt_e0/B_methanol_010.mfp5", "nvt_e0", 0, 500000)
    V = np.dot(np.cross(cell[0],cell[1]),cell[2])
    dip = np.array(list(dip.values())).sum(axis=0)
    pol = dip/V * 1.602177e-19*(meter/angstrom)**2
    reblock_data = pyblock.blocking.reblock(pol[:,0])
    opt = pyblock.blocking.find_optimal_block(pol.shape[0], reblock_data)
    err_pol_nf = reblock_data[opt[0]].std_err
    pol_nf = pol.mean(axis=0)
    p_nf = dip.mean(axis=0)/V
    f = open("xpol.dat", "w")
    fd = open("xd.dat", "w")
    feps = open("xeps.dat", "w")
    ferr = open("xerr.dat","w")
    ferr_p = open("xerr_p.dat","w")
    for i in fields:
        dip, cell = load_data(f"e_{i}/B_methanol_010.mfp5", "nvt_e", 0, 100000)
        dip = np.array(list(dip.values())).sum(axis=0)
        p = dip/V
        p_diff = p-p_nf
        pol = p * 1.602177e-19*(meter/angstrom)**2
        reblock_data = pyblock.blocking.reblock(pol[:,0])
        opt = pyblock.blocking.find_optimal_block(pol.shape[0], reblock_data)
        err_pol = reblock_data[opt[0]].std_err
        err_p = np.sqrt(err_pol**2.0+err_pol_nf**2.0)/(1.602177e-19*(meter/angstrom)**2)
        pol = pol.mean(axis=0) 
        E = np.array([float(i),0.0,0.0])
        field = E*(meter/angstrom)
        eps = 1.0+((pol-pol_nf)/(eps_0*field[0]))
        D = 5.526348e-3*field+pol
        print("FIELD:",i)
        print(eps)
        # calc error
        err = np.sqrt((err_pol/(eps_0*field[0]))**2.0+(err_pol_nf/(eps_0*field[0]))**2.0)
        print("errors:",err_pol,err_pol_nf,err)
        #print numpy.std(vals2[:,0]), numpy.mean(vals2[:,0])

        p_diff = p_diff.mean(axis=0)
        f.write("%14.8f %14.8f %14.8f %14.8f\n" % (E[0], p_diff[0], p_diff[1], p_diff[2]))
        fd.write("%12.6f %12.6f %12.6f %12.6f\n" % (E[0], D[0], D[1], D[2]))
        feps.write("%14.8f %14.8f %14.8f %14.8f\n" % (E[0], eps[0], eps[1], eps[2]))
        ferr.write("%14.8f %14.8f\n" % (E[0],err))
        ferr_p.write("%14.8f %14.8f\n" % (E[0],err_p))

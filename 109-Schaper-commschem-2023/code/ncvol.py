"""
    ncvol

    this module implements pressure or a volume constraint for MOF nanocrystallites 
    based on cuboids for each unit cell based on expot 


"""

import numpy as np
import time
from mpi4py import MPI

from pylmps import expot_base

from molsys.util.timer import Timer, timer

import cuboid

import numba as nb

class ncvol(expot_base):

    def __init__(self, sbu_frags, mode='off', k=None, Vref = None, outfile="ncvol.out", nstep=100, inner_cn=6, omit_inner_grad=True):
        """generate a volume measure 

        Args:
            sbu_frags (list of strings): fragments to be considered as SBU
            mode (string): the mode to run ncvol, can be 'off' or 'US'
            k (float): force constant for the harmonic potential [UNIT?]
            nstep (int): number of steps until outfile is written (default 100)
            inner_cn (int): coordination number of inner vertices (default 6)
            omit_inner_grad: if True (default) do not compute gradients for any inner tetrahedron 

        """
        super(ncvol, self).__init__()
        self.sbu_frags = sbu_frags
        self.inner_cn = inner_cn
        self.omit_inner_grad = omit_inner_grad
        self.use_mpi = True
        self.use_numba = True
        self.mode = mode
        assert mode in ["off", "US"]
        self.press_counter=0
        self.us_used = False
        self.fix_used = False
        if mode == 'US':
            # set up umbrella sampling restraint potential on top 
            self.us_used = True
            assert k != None
            assert Vref != None
            self.k = k 
            self.Vref = Vref
            self.counter = 0
        self.outf_name = outfile
        self.step = 0
        self.nstep = nstep
        self.time = 0.0
        self.timer = Timer("ncvol")
        return

    def setup(self, pl):
        # print ("DEBUG DEBUG this ncvol setup")
        super(ncvol, self).setup(pl)
        self.name = "ncvol"
        # generate the fragment graph to get the atom_maps
        # TBI: assert that sbu_frags are really in fragments
        self.pl = pl
        self.mol = pl.mol 
        self.mol.addon("fragments")
        self.mol.fragments.add_frag_graph()
        self.mol.fragments.merge_frags("base", "sbumerge", self.sbu_frags, "sbu")
        self.mol.fragments.remove_dangling_frags("sbumerge", "sbured")
        self.cgraph = self.mol.fragments.remove_bridge_frags("sbured", "cubes")
        # boolean list of vertices with coord = 6 (inner vertices that do not chenge the volume)
        self.inner = np.array([(v.out_degree() == self.inner_cn) for v in self.cgraph.vertices()], dtype="bool")
        # find the cuboids in the cubes fgraph 
        self.cub_ids = (self.mol.fragments.find_subgraph("cubes", cuboid.cuboid_graph))
        # DEBUG just for debug reasons
        # tm = self.mol.fragments.fgraph_to_mol("cubes", {"sbu": "c"})
        # tm.write("topo.mfpx")
        # DEBUG END
        self.cuboids = []
        for c in self.cub_ids:
            if self.omit_inner_grad:
                # get the list of boolean list for current vertices
                is_inner = [self.inner[i] for i in c]
            else:
                is_inner = False
            self.cuboids.append(cuboid.cuboid(c, inner=is_inner))
        self.ncubs = len(self.cuboids)
        self.cub_V = np.zeros([self.ncubs], dtype="float64")  # this is only to keep the volume per cube for analysis
        # array for sbu COMs and COM forces
        self.ncom = self.cgraph.num_vertices()
        self.com  = np.zeros([self.ncom, 3], dtype="float64")
        self.dcom  = np.zeros([self.ncom, 3], dtype="float64")
        self.fcom  = np.zeros([self.ncom, 3], dtype="float64")
        # get the mass array for COm calculation (needs to be done only once here)
        self.mol.set_real_mass()
        self.mass = np.array(self.mol.get_mass()) # make a numpy array from the list to allow index lookup
        # prepare the index lists (first check maximum of atoms per map) and masses
        max_atom = 0
        for v in self.cgraph.vertices():
            na = len(self.cgraph.vp.atom_map[v])
            if na > max_atom:
                max_atom = na
        # now make numpy arrays for atom map indices, the number of atoms per vertex and mass&summass
        # to allow fast computation of coms and com forces in each iteration
        #  ==> consider making a numba subroutine for it (maybe no need to parallelize it)
        self.am = np.zeros((self.ncom, max_atom), dtype="int32")
        self.am_n = np.zeros((self.ncom), dtype="int32")
        self.am_mass = np.zeros((self.ncom, max_atom), dtype="float64")
        self.am_summass = np.zeros((self.ncom), dtype="float64")
        for i, v in enumerate(self.cgraph.iter_vertices()):
            am = np.array(self.cgraph.vp.atom_map[v], dtype="int32")
            n = len(am)
            mass = self.mass[am]
            self.am[i,:n] = am
            self.am_n[i] = n
            self.am_mass[i,:n] = mass
            self.am_summass[i] = mass.sum()
        ## open debug outfile
        self.outf = open(self.outf_name, "w")
        #self.outf.write("NCVOL is set up\n")
        #self.outf.write("%5d COM positions\n" % self.ncom)
        #self.outf.write("%5d cuboids\n" % self.ncubs)
        return

    def current_Vref(self):
        """ returns the respective reference volume Vref for the current MD step
        """
        if hasattr(self,'Vrefs'):
             Vref_used=np.linspace(self.Vrefs[0],self.Vrefs[1], num=self.Vrefs[2])
             #print(Vref_used)
             self.Vref=Vref_used[self.counter]
             self.counter+=1
             if self.counter==self.Vrefs[2]:
                 self.counter=0
             #print(Vref)
        return self.Vref
        
    def current_k(self):
        """  allows to ramp over different force constants k of the quadratic potential used in the us mode. Useful to get the range of the force constant for unknown systems"""
        if hasattr(self,'force_const'):
            k_used=np.linspace(self.force_const[0],self.force_const[1], num=self.force_const[2])
            self.k=k_used[self.counter]
            self.counter+=1
            if self.counter==self.force_const[2]:
                self.counter=0
            #print(self.k)    
        return self.k
		
 
    def calc_energy_force(self):
        """compute the energy and forces on all atoms from the cuboid volumes times pressure

        Returns:
            float, ndarray: energy and force
        """
        tstart = time.time()
        self.timer.start()
        # first compute the COM positions from the atom positions per cgraph vertex using its atommap
        # if we use fix then we need to get the actual forces from the MM engine and project them onto the COMs
        if self.use_numba:
            with self.timer.fork("generate coms numba"):
                generate_coms(self.com, self.xyz, self.am, self.am_n, self.am_mass, self.am_summass)
        else:
            self.generate_coms()
        # now compute volume and gradient of COMs
        cuboid_timer = self.timer.fork("compute cuboids")
        self.cub_V[...] = 0.0
        self.dcom[...] = 0.0
        if self.use_mpi:
            # distribute cuboid work over the nodes
            for i in range(self.mpi_rank, len(self.cuboids), self.mpi_size):
                c = self.cuboids[i]
                self.cub_V[i] = c.calc_volume_deriv(self.com, self.dcom)
            if self.mpi_size > 1:
                cubcomm_timer = cuboid_timer.fork("cuboid data communicate")
                # broadcast volumes and forces to all nodes (can we do this inplace?)
                Vtemp = np.zeros(self.cub_V.shape, dtype="float64")
                self.mpi_comm.Allreduce(self.cub_V, Vtemp, MPI.SUM)
                self.cub_V = Vtemp
                Vdtemp = np.zeros(self.dcom.shape, dtype="float64")
                self.mpi_comm.Allreduce(self.dcom, Vdtemp, MPI.SUM)
                self.dcom = Vdtemp
                cubcomm_timer.stop()
        else:
            # for debug reasons do it without
            for i, c in enumerate(self.cuboids):
                self.cub_V[i] = c.calc_volume_deriv(self.com, self.dcom)
        cuboid_timer.stop()
        if self.omit_inner_grad:
            # in this case we did not computed forces of tetrahedrons that are completely inside
            # ==> as a consequnce some vertices have now a net force that is not balanced by the other force, not computed
            #     all vertices that are inner have threfore to get a zero force enforced
            self.dcom = np.where(self.inner[:,None], 0.0, self.dcom)
        self.V = self.cub_V.sum()
        self.energy = 0.0
        self.fcom[...] = 0.0
        if self.us_used:
            k=self.current_k()
            Vref=self.current_Vref()
            dV = self.V-Vref
            self.energy_us = 0.5 * k * dV*dV
            #print(self.energy,Vref,k)
            self.fcom_us = -k * dV * self.dcom
            # add US energy and force to total
            self.energy += self.energy_us
            self.fcom += self.fcom_us
        # now put COMs back on the corresponding atoms
        if self.use_numba:
            with self.timer.fork("distribute force numba"):
                distribute_force(self.fcom, self.force, self.am, self.am_n, self.am_mass, self.am_summass)
        else:
            self.distribute_force()
        # write output every nsteps
        if (self.step % self.nstep) == 0:
            tavrg = self.time/(self.step+1)
            ttavrg = self.expot_time/(self.step+1)
            if self.us_used:
                self.outf.write('%3d %12.6f %12.6f %12.6f %10.5f %10.5f %10.5f %10.5f\n' % (self.step, self.V, self.energy, self.energy_us, k, Vref, tavrg, ttavrg))
            else:
                self.outf.write('%3d %12.6f %12.6f %10.5f %10.5f\n' % (self.step, self.V, self.energy, tavrg, ttavrg))
            self.outf.flush()
        self.time += time.time() - tstart
        self.timer.stop()
        return self.energy, self.force
    

    @timer("generate coms")
    def generate_coms(self):
        for i in range(self.ncom):
            n = self.am_n[i]
            self.com[i] = (self.xyz[self.am[i,:n]]*self.am_mass[i,:n,np.newaxis]).sum(axis=0)/self.am_summass[i]
        return

    @timer("distribute force")
    def distribute_force(self):
        self.force[::] = 0.0
        for i in range(self.ncom):
            n = self.am_n[i]
            self.force[self.am[i,:n]] += self.fcom[i]/self.am_summass[i]*self.am_mass[i,:n,np.newaxis]
        return


    def test_com_deriv(self, delta=0.0001):
        """test the derivatives of the COMs by numerical differentiation

        assumes that the energy and force has already been computed so that self.xyz is present

        Args:
            delta (float, optional): shift of COMs. Defaults to 0.0001.
        """
        assert self.mode == "p", "This test works only in pressure mode with a finite pressure"
        self.outf.write("\nTEST DERIVATIVES OF COM\n\n")
        # get analytik reference forces
        self.calc_energy_force()
        fcom_ref = self.fcom.copy()
        com_ref  = self.com.copy()
        fcom_dummy = np.zeros(self.fcom.shape)
        for i in range(self.ncom):
            for j in range(3):
                keep = com_ref[i,j]
                com_ref[i,j] += delta
                Vm = 0.0
                for c in self.cuboids:
                    Vm += c.calc_volume_deriv(com_ref, fcom_dummy)
                com_ref[i,j] -= 2*delta
                Vp = 0.0
                for c in self.cuboids:
                    Vp += c.calc_volume_deriv(com_ref, fcom_dummy)
                com_ref[i,j] = keep
                force = (Vp-Vm)/(2.0*delta)*self.p # this is the force of the energy (V*p)
                deltf = fcom_ref[i,j]-force
                if abs(deltf) > 10*delta:
                    warning = "WARNING!"
                else:
                    warning = ""
                self.outf.write("  com %3d  position %10.5f %10.5f %10.5f  dim %3d:  analyt %10.5f  numeric %10.5f (delta: %8.2f %s) \n" % \
                    (i, com_ref[i,0], com_ref[i,1], com_ref[i,2], j, fcom_ref[i,j], force, deltf, warning))
                self.outf.flush()
        return

#### numba variants of generate com and dist_force

@nb.jit(nopython=True)
def generate_coms(com, xyz, am, am_n, am_mass, am_summass):
    for i in range(len(com)):
        com[i] = 0.0
        for n in range(am_n[i]):
            com[i] += xyz[am[i, n]]*am_mass[i,n]
        com[i] /= am_summass[i]
    return

@nb.jit(nopython=True)
def distribute_force(fcom, force, am, am_n, am_mass, am_summass):
    force[::] = 0.0
    for i in range(len(fcom)):
        fom = fcom[i]/am_summass[i]
        for n in range(am_n[i]):
            j = am[i, n]
            force[j] += fom*am_mass[i,n]
    return




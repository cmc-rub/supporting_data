"""
cuboid class to handle a deformed arbitrary unit cell within a NC

R. Schmid (April 2021, RUB)

"""

import numpy as np
from graph_tool import Graph

import numba as nb

onesix = 1/6.0

# this is a helper function
@nb.jit('Tuple((float64, float64[:,:]))(int32[:], float64[:,:])', nopython=True)
def tetrahed_vol_deriv(vertices, xyz):
    """compute volume and derivative of volume for a tetrahedron

    Args:
        vert (list of int len(4)): indices of tetrahedron vertices in xyz
        xyz (numpy [N,3]): coordinate array

    Returns:    
        (float, numpy array) : Volume, Volume derivative 
    """
    dV = np.zeros((4, 3), dtype="float64")
    o = xyz[vertices[0]]
    a = xyz[vertices[1]]
    b = xyz[vertices[2]]
    c = xyz[vertices[3]]
    ax, ay, az = a-o
    bx, by, bz = b-o
    cx, cy, cz = c-o
    det = ax*(by*cz-bz*cy) + ay*(bz*cx-bx*cz) + az*(bx*cy-by*cx)
    V = abs(det)
    dV[1,0] = by*cz-bz*cy   # ax
    dV[1,1] = bz*cx-bx*cz   # ay
    dV[1,2] = bx*cy-by*cx   # az
    dV[2,0] = az*cy-ay*cz   # bx
    dV[2,1] = ax*cz-az*cx   # by
    dV[2,2] = ay*cx-ax*cy   # bz
    dV[3,0] = ay*bz-az*by   # cx
    dV[3,1] = az*bx-ax*bz   # cy
    dV[3,2] = ax*by-ay*bx   # cz
    # add origin forces
    dV[0,0] = -dV[1:,0].sum()
    dV[0,1] = -dV[1:,1].sum()
    dV[0,2] = -dV[1:,2].sum()
    return onesix*V, onesix*np.sign(det)*dV

# volume only (just in case this is ever needed)
@nb.jit('float64(int32[:], float64[:,:])', nopython=True)
def tetrahed_vol(vertices, xyz):
    """compute volume for a tetrahedron

    Args:
        vert (list of int len(4)): indices of tetrahedron vertices in xyz
        xyz (numpy [N,3]): coordinate array

    Returns:    
        float : Volume 
    """
    o = xyz[vertices[0]]
    a = xyz[vertices[1]]
    b = xyz[vertices[2]]
    c = xyz[vertices[3]]
    ax, ay, az = a-o
    bx, by, bz = b-o
    cx, cy, cz = c-o
    V = abs(ax*(by*cz-bz*cy) + ay*(bz*cx-bx*cz) + az*(bx*cy-by*cx))
    return onesix*V



class cuboid:
    """
        This class implements a geometric object defined by eight vertices in 
        three-dimensional space. It's called cuboid, because it's easiest to think
        of it as a cube. However, the eight coordinates defining the object
        do not necessarily need to form a cube, but can have any shape possible 
    """

    def __init__(self, vertices, inner=False):
        """instantiate a cuboid defiend by the positions in an array indexed with vertices
           
          6------7
         /|     /|
        3------5 |
        | 2----|-4
        |/     |/
        0------1

        0     =  origin
        1,2,3 = connect to origin
        4,5,6 = connect to 1,2,3 (id = id1+id2+1)
        7     = connects to 4,5,6

        ==> the following tetrahedrons are defined
        (why is there A and B? becasue you have two choices for tesselation: choice of the central tetrahedron
        See: https://en.wikipedia.org/wiki/Marching_tetrahedra
        For an arbitrary cuboid it makes a differnce e.g. on the volume which tesselation to choose. Threfore we use
        an average of both which is safe and no ambiguity arises. Internally both volumes A and B are determined. 
        In case of a perfect cube A and B is equal.)

        central
        0, 4, 5, 6       (cA)
        1, 2, 3, 7       (cB)

        on the tips
        tA                 tB
        1,0,4,5            0,1,2,3
        2,0,4,6            4,1,2,7
        3,0,5,6            5,1,3,7 
        7,4,5,6            6,2,3,7

        we generate lists with proper indices at instantiation. These indices can then be used to pick 
        vertex positions from a gloabl xyz array. forces are collected first locally and then are distributed to the 
        global forces array

        Args:
            - vertices [iterable of ints]: vertex indices forming the cuboid
            - inner [iterable of booleans]: if True the vertex is an inner vertex

        Returns:
            instance of cuboid: the cuboid instance
        """    
        assert len(vertices) == 8
        self.omit_inner_forces = False
        if inner != False:
            assert len(inner) == 8
            self.omit_inner_forces = True
        self.vertices = list(vertices)
        self.VA = 0.0
        self.VB = 0.0
        self.V  = 0.0
        self.dV = np.zeros([8,3], dtype="float64")
        self.dVA = np.zeros([8,3], dtype="float64")
        self.dVB = np.zeros([8,3], dtype="float64")
        # define the five tetrahedron defintions for both tesselation variants A and B (inner is the first)
        self.tetA_ind = [
            [0,4,5,6],
            [1,0,4,5],
            [2,0,4,6],
            [3,0,5,6],
            [7,4,5,6],
        ]
        self.tetB_ind = [
            [1,2,3,7],
            [0,1,2,3],
            [4,1,2,7],
            [5,1,3,7],
            [6,2,3,7],
        ]
        self.tetA = []
        for ti in self.tetA_ind:
            self.tetA.append(np.array([vertices[i] for i in ti], dtype="int32"))
        self.tetB = []
        for ti in self.tetB_ind:
            self.tetB.append(np.array([vertices[i] for i in ti], dtype="int32"))
        if self.omit_inner_forces == True:
            # if any vertex is not inner we need to compute forces
            self.tetA_inner = []
            for ti in self.tetA_ind:
                flag = True
                for i in ti:
                    if inner[i] == False:
                        flag = False
                self.tetA_inner.append(flag)
            self.tetB_inner = []
            for ti in self.tetB_ind:
                flag = True
                for i in ti:
                    if inner[i] == False:
                        flag = False
                self.tetB_inner.append(flag)
        return
        
    def calc_volume(self, xyz):
        """calculates the cuboid volume

        No derivatives are computed

        Returns:
            float: the cuboid volume
        """  
        self.VA = 0.0
        for t in self.tetA:
            self.VA += tetrahed_vol(t, xyz)
            # print ("A: tet %10s %10.5f" % (t, tetrahed_vol(t, xyz)))
        self.VB = 0.0
        for t in self.tetB:
            self.VB += tetrahed_vol(t, xyz)
            # print ("B: tet %10s %10.5f" % (t, tetrahed_vol(t, xyz)))
        self.V = (self.VA+self.VB)*0.5
        return self.V

    def calc_volume_deriv(self, xyz, dxyz):
        """calculates the cuboid volume and the derivative of the volume wrt the defining vertices

        The gradients are added to the dxyz array of the same shape as the xyz array (make sure to zero in the proper way)
        Note we keep a local gradient internally

        Returns:
            float: the cuboid volume
        """
        self.VA = 0.0
        self.dVA[::] = 0.0
        self.VB = 0.0
        self.dVB[::] = 0.0
        # if omit force check for each tetrahedron
        if self.omit_inner_forces:
            for i in range(5):
                if self.tetA_inner[i]:
                    VA = tetrahed_vol(self.tetA[i], xyz)
                else:
                    VA, dVA = tetrahed_vol_deriv(self.tetA[i], xyz)
                    self.dVA[self.tetA_ind[i]] += dVA
                self.VA += VA
            for i in range(5):
                if self.tetB_inner[i]:
                    VB = tetrahed_vol(self.tetB[i], xyz)
                else:
                    VB, dVB = tetrahed_vol_deriv(self.tetB[i], xyz)
                    self.dVB[self.tetB_ind[i]] += dVB        
                self.VB += VB
        else:
            for i in range(5):
                VA, dVA = tetrahed_vol_deriv(self.tetA[i], xyz)
                self.VA += VA
                self.dVA[self.tetA_ind[i]] += dVA
            for i in range(5):
                VB, dVB = tetrahed_vol_deriv(self.tetB[i], xyz)
                self.VB += VB
                self.dVB[self.tetB_ind[i]] += dVB
        # average and add to the dxyz external gradient
        self.V  = (self.VA+self.VB)*0.5
        self.dV = (self.dVA+self.dVB)*0.5
        dxyz[self.vertices] += self.dV
        return self.V
    
    def test_volume_numderiv(self, xyz, delta = 1e-2):
        """ For fixing any typo bugs we do it for A and B seperate 
        """
        dxyz = np.zeros(xyz.shape, dtype="float64")
        # compute ref analytic gradient
        self.calc_volume_deriv(xyz, dxyz)
        dVA = self.dVA.copy()
        dVB = self.dVB.copy()
        # now do the numeric diff
        lxyz = xyz.copy()
        ndVA = np.zeros([8,3], dtype="float64")
        ndVB = np.zeros([8,3], dtype="float64")
        for i in range(8):
            v = self.vertices[i]
            for j in range(3):
                lxyz[v,j] += delta
                self.calc_volume(lxyz)
                VAp = self.VA
                VBp = self.VB
                lxyz[v,j] -= 2*delta
                self.calc_volume(lxyz)
                VAm = self.VA
                VBm = self.VB
                lxyz[v,j] += delta
                ndVA[i,j] = (VAp - VAm)/(2*delta)
                ndVB[i,j] = (VBp - VBm)/(2*delta)
        print ("Comparing A")
        print (dVA-ndVA)
        print ("Comparing B")
        print (dVB-ndVB)
        return

########### helper graph for a properly setup cuboid

# make a cube subgraph
cuboid_graph = Graph(directed=False)
cuboid_graph.vp.type = cuboid_graph.new_vertex_property("string")
for i in range(8):
    v = cuboid_graph.add_vertex()
    cuboid_graph.vp.type[v] = "sbu"
# make a cube in the right connectivity (this is not arbitrary!)
cuboid_graph.add_edge(0, 1)
cuboid_graph.add_edge(0, 2)
cuboid_graph.add_edge(0, 3)
cuboid_graph.add_edge(1, 4)
cuboid_graph.add_edge(1, 5)
cuboid_graph.add_edge(2, 4)
cuboid_graph.add_edge(2, 6)
cuboid_graph.add_edge(3, 5)
cuboid_graph.add_edge(3, 6)
cuboid_graph.add_edge(4, 7)
cuboid_graph.add_edge(5, 7)
cuboid_graph.add_edge(6, 7)




if __name__ == "__main__":
    xyz = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]], dtype="float64")
    xyz = xyz * 10.0
    # add rnadom displacements TBI
    dxyz = np.zeros(xyz.shape)

    cub = cuboid([0,1,2,3,4,5,6,7])
    print (cub.calc_volume_deriv(xyz, dxyz))
    print (cub.dVA)
    print (cub.dVB)
    cub.test_volume_numderiv(xyz)


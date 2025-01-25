# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 12:40:05 2019

@author: fyalcin
"""

import numpy as np
import inspect
from functools import wraps
import scipy.linalg as la
import copy
import time
import matplotlib.pyplot as plt

latConst = 1.42
V2p = 0;
#V2s = -8.868;
V2s = 0;
Vpp = -5.037*latConst**2
Vss = -6.769*latConst**2
Vsp = -5.580*latConst**2
#Vpi = -3.033*latConst**2
Vpi = -2.7*latConst**2
Ssp = 0.1*latConst**2
Sss = 0.20*latConst**2
Spp = 0.15*latConst**2
Spi = 0.15*latConst**2

def NeighborStr(atom):
    neighborstr = ''
    for neighbor in atom.neighbors:
        separationVec = neighbor.coord - atom.coord
        separationSepDist = (separationVec.dot(separationVec))**0.5
        neighborstr += 'Carbon ' + str(neighbor.MUCindex[1]) + ' at coord. ' + \
            str(np.round(neighbor.coord,2)) + ', Rvec ' + str(np.round(neighbor.Rvec,2)) + \
           " and SepDistance "+ str(np.round(separationSepDist,2)) + '\n'
    return neighborstr
    
def initializer(fun):
    names, varargs, keywords, defaults = inspect.getargspec(fun)
    @wraps(fun)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)
        for i in range(len(defaults)):
            index = -(i + 1)
            if not hasattr(self, names[index]):
                setattr(self, names[index], defaults[index])
        fun(self, *args, **kargs)
    return wrapper

def check(v1,v2):
    if len(v1)!=len(v2):
        raise ValueError("the length of both arrays must be the same")
    pass

def dot(v1,v2):
    """                                                                                                     
    d0 is Nominal approach:                                                                                 
    multiply/add in a loop                                                                                  
    """
    check(v1,v2)
    out = 0
    for k in range(len(v1)):
        out += v1[k] * v2[k]
    return out

def gaussianize(values, xarray, sigma):
    yarray = []
    total = 0
    for x in xarray:
        total = 0
        for value in values:
            total += np.exp(-(x-value)**2/(2*sigma**2)) 
        yarray.append(total)
    return xarray, yarray

def dist(a,b,sigma):
    return np.exp(-(a-b)**2/(2*sigma**2))

def DOS(E, eigvals, sigma):
    dos = []
    for i in range(len(E)):
        total = 0
        for j in range(len(eigvals)):
            total += dist(E[i],eigvals[j],sigma)
        dos.append(total)
    return dos
        
def PDOS(E, eigvals, eigvecs, S, sigma):
    eigvecs = eigvecs.T
    pdos = eigvecs[0].conj()*np.dot(S,eigvecs[0])*dist(E,eigvals[0],sigma)
    for i in range(1,len(eigvals)):
        pdos += eigvecs[i].conj()*np.dot(S,eigvecs[i])*dist(E,eigvals[i],sigma)
    return pdos.real

def sumPDOS(E, eigvals, eigvecs, S, sigma):
    totpdos = PDOS(E[0], eigvals, eigvecs, S, sigma)
    for i in range(1, len(E)):
        totpdos += PDOS(E[i], eigvals, eigvecs, S, sigma)
    return totpdos

def plotPDOS(cell, pdos, size):
    xcoord = [atom.coord[0] for atom in cell.atoms]
    ycoord = [atom.coord[1] for atom in cell.atoms]
    plt.scatter(xcoord, ycoord, s= size*pdos, c = "r")
    
def GetPeierlsPhase(lattice, atom, neighbor, alpha): 
    if lattice.Lattice is "custom":
        if atom.Spaceindex[0] == 0 and neighbor.Spaceindex[0] == 0:
            PeierlsFactor = -2*np.pi*alpha*int(atom.Spaceindex[1]>5)*np.sign(neighbor.sepVec[0])
            PeierlsPhase = np.exp(np.complex(0, PeierlsFactor))
        else:
            PeierlsPhase = 1
    elif lattice.Lattice is "flake":
        if atom.Spaceindex[0]+neighbor.Spaceindex[0] == 0 and atom.Spaceindex[1] > 0:
            PeierlsFactor = 2*np.pi*alpha*np.sign(neighbor.sepVec[0])
            PeierlsPhase = np.exp(np.complex(0, PeierlsFactor))
        else:
            PeierlsPhase = 1
    elif lattice.Lattice is "AG" or lattice.Lattice is "PAG":
        if np.abs(neighbor.sepAngle-2*np.pi/3) < 0.1 or np.abs(neighbor.sepAngle+np.pi/3) < 0.1:
            PeierlsFactor = 2*np.pi*alpha*(atom.Spaceindex[0])*np.sign(neighbor.sepAngle)
            PeierlsPhase = np.exp(np.complex(0, PeierlsFactor))
        else:
            PeierlsPhase = 1
    elif lattice.Lattice is "square":
        if np.abs(neighbor.sepAngle-np.pi/2) < 0.1 or np.abs(neighbor.sepAngle+np.pi/2) < 0.1:
            PeierlsFactor = 2*np.pi*alpha*(atom.UCindex[0][0])*np.sign(neighbor.sepAngle)
            PeierlsPhase = np.exp(np.complex(0, PeierlsFactor))
        else:
            PeierlsPhase = 1
    elif lattice.Lattice is "PG":
        if np.abs(neighbor.sepAngle-2*np.pi/3) < 0.1 or np.abs(neighbor.sepAngle+np.pi/3) < 0.1:
            PeierlsFactor = 2*np.pi*alpha*(atom.Spaceindex[0])*np.sign(neighbor.sepAngle)
            PeierlsPhase = np.exp(np.complex(0, PeierlsFactor))
        else:
            PeierlsPhase = 1
    else:
        PeierlsPhase = 1
    return PeierlsPhase

def intMatrix(SepAngle, SepDist, orbitals = "s"):
    """ Creates 4x4 interaction matrices between possible orbitals,
    hopping parameters are scaled to SepDistance by an inverse SepDistance squared 
    relation """
    if orbitals is "s":
        return Vpi/SepDist**2, Spi/SepDist**2
    n11 = Vss
    n12 = -Vsp*np.cos(SepAngle)
    n13 = -Vsp*np.sin(SepAngle)
    n14 = n24 = n34 = n41 = n42 = n43 = 0
    n21 = -n12
    n22 = (-Vpp*np.cos(SepAngle)**2) + (Vpi*np.sin(SepAngle)**2)
    n23 = -(Vpp+Vpi)*np.sin(SepAngle)*np.cos(SepAngle)
    n31 = -n13
    n32 = n23
    n33 = (-Vpp*np.sin(SepAngle)**2) + (Vpi*np.cos(SepAngle)**2)
    n44 = Vpi
      
    m1 = np.array([[n11, n12, n13, n14],
                   [n21, n22, n23, n24],
                   [n31, n32, n33, n34],
                   [n41, n42, n43, n44]])
    m1 = m1/SepDist**2;
               
    s11 = Sss
    s12 = -Ssp*np.cos(SepAngle);
    s13 = -Ssp*np.sin(SepAngle);
    s14 = s24 = s34 = s41 = s42 = s43 = 0
    s21 = -s12
    s22 = (-Spp*np.cos(SepAngle)**2 + Spi*np.sin(SepAngle)**2)
    s23 = -(Spp+Spi)*np.sin(SepAngle)*np.cos(SepAngle)
    s31 = -s13
    s32 = s23
    s33 = (-Spp*np.sin(SepAngle)**2 + Spi*np.cos(SepAngle)**2)
    s44 = Spi
            
    m2 = np.array([[s11, s12, s13, s14],
                   [s21, s22, s23, s24],
                   [s31, s32, s33, s34],
                   [s41, s42, s43, s44]])
    m2 = m2/SepDist**2
    return m1, m2

Lattices = {}        
Lattices["square"] = {"basisVec":np.array([[0, 0, 0]]), \
                    "latVec":latConst*np.array([[1, 0, 0],[0, 1, 0]])}
Lattices["PG"] = {"basisVec":np.array([[0.0, 0.0, 0.0], [latConst, 0.0, 0.0]]), \
                    "latVec":latConst*0.5*np.array([[3,np.sqrt(3),0.0],[3,-np.sqrt(3),0.0]])}
Lattices["AG"] = {"basisVec":np.array([[0, 0, 0], [latConst/2, latConst*np.sqrt(3)/2, 0.0],\
                    [0.0, latConst*np.sqrt(3), 0.0], [latConst/2, 3*latConst*np.sqrt(3)/2, 0.0]]),\
                    "latVec":latConst*0.5*np.array([[3, np.sqrt(3), 0.0], [0.0, 4*np.sqrt(3), 0.0]])}
Lattices["PAG"] = {"basisVec":np.array([[0.0, 0.0, 0.0], [latConst, 0.0, 0.0]]),\
                    "latVec":latConst*0.5*np.array([[3,np.sqrt(3),0.0],[0,-2*np.sqrt(3),0.0]])}
Lattices["flake"] = {"basisVec":np.array([[0.0, 0.0, 0.0], [latConst, 0.0, 0.0]]), \
                    "latVec":latConst*0.5*np.array([[3,np.sqrt(3),0.0],[3,-np.sqrt(3),0.0]])}
Lattices["armchair"] = {"basisVec":np.array([[0.0, 0.0, 0.0], [latConst, 0.0, 0.0],[3*latConst/2,\
                        latConst*np.sqrt(3)/2,0],[5*latConst/2,latConst*np.sqrt(3)/2,0]]),\
                        "latVec":latConst*np.array([[3,0,0],[0,np.sqrt(3),0]])}

def getlatinfo(lattice):
    if lattice in Lattices.keys():
        return Lattices[lattice]["basisVec"], Lattices[lattice]["latVec"]
    raise Exception('Please input a correct lattice type')

IntMatGrapS = [Vpi, Spi]

IntMatGrapAO = []
for i in range(6):
    IntMatGrapAO.append(intMatrix(np.pi*i/3, latConst, orbitals="all"))
    
def kpath(N):
    """ Creates kpoints between the high symmetry points in the
    first BZ of graphene """
#    kpath = []
    kpath = np.empty((0,3), float)
    for n in range(N+1):
        ky = (1-n/N)*2*np.pi/(latConst*3*(3**0.5))
        kx = ky*(3**0.5)
        kpoint = np.array([kx ,ky ,0])
        kpath = np.vstack((kpath, kpoint))
    for n in range(N+1):
        kx = (n/N)*2*np.pi/(latConst*3)
        ky = 0
        kpoint = np.array([kx, ky, 0])
        kpath = np.vstack((kpath, kpoint))
    for n in range(N+1):
        kx = 2*np.pi/(3*latConst)
        ky = (n/N)*2*np.pi/(latConst*3*(3**0.5));
        kpoint = np.array([kx, ky, 0])
        kpath = np.vstack((kpath, kpoint))
    return kpath       

def rotate(point, angle):
    """ This is just a rotation function that rotates the given point
    counterclockwise by <N*angle> degrees """
    rotMatrix = np.array([[np.cos(angle), -np.sin(angle), 0], \
        [np.sin(angle), np.cos(angle), 0],[0, 0, 1]])
    point = np.dot(rotMatrix, point)
    return point

def kpointsBZ(N):
    ''' Generate kpoints inside the irreducible triangle in the first BZ of graphene '''
    prop = (2*np.pi/(3*latConst))
    irred = [np.array([prop*nx/(N-1), prop*ny/(N-1), 0]) for nx in range(N) \
             for ny in range(N) if nx > ny*np.sqrt(3)]
    tmp = copy.deepcopy(irred)
    for point in tmp:
        point[1] = -point[1]
        irred.append(point)
    tmp = copy.copy(irred)
    for n in range(1,6):
        for point in tmp:
            point = rotate(point, n*np.pi/3)
            irred.append(point)
    return irred

def BuildJunction(width, length):
    angle = np.pi/3
    rotMatrix = np.array([[np.cos(angle), -np.sin(angle), 0], \
            [np.sin(angle), np.cos(angle), 0],[0, 0, 1]])
    basis = []
    for i in range(6):
        rotMatrix = np.array([[np.cos(i*angle), -np.sin(i*angle), 0], \
                        [np.sin(i*angle), np.cos(i*angle), 0],[0, 0, 1]])
        rotatedcoord = np.dot(rotMatrix, np.array([latConst,0,0]))
        basis.append(rotatedcoord)
    arms = np.empty((0,3))
    for w in range(width):
        for l in range(length):
            newcoords = basis + np.array([l*(3*latConst/2), -(2*w+l)*(latConst*np.sqrt(3))/2, 0])
            for coord in newcoords: 
                arms = np.vstack((arms, coord))
    mirrorCell = -arms
    arms[:,1] *= -1
    arms = np.vstack((arms, mirrorCell))
    arms = np.unique(arms, axis=0)
    
    body = np.empty((0,3))
    bodyl = length
    for w in range(width):
        for bl in range(bodyl):
            newcoords = basis + np.array([w*(3*latConst/2), -(2*bl+w%2)*(latConst*np.sqrt(3))/2, 0])
            for coord in newcoords:
                body = np.vstack((body,coord))
    body += np.array([-(width//2)*(3*latConst/2), ((width//2)%2)*latConst*np.sqrt(3)/2, 0])
    junction = np.empty((0,3))
    junction = np.vstack((arms, body))
    junction = np.round(junction, 5)
    junction = np.unique(junction, axis=0)
    return junction

def hexagon(orientation="a", center = np.array([0,0,0])):
    angle = np.pi/3
    rotMatrix = np.array([[np.cos(angle), -np.sin(angle), 0], \
            [np.sin(angle), np.cos(angle), 0],[0, 0, 1]])
    basis = []
    for i in range(6):
        rotMatrix = np.array([[np.cos(i*angle), -np.sin(i*angle), 0], \
                        [np.sin(i*angle), np.cos(i*angle), 0],[0, 0, 1]])
        rotatedcoord = np.dot(rotMatrix, np.array([latConst,0,0]))
        basis.append(rotatedcoord)
    if orientation == "z":
        for i,j in enumerate(basis):
            basis[i] = rotate(basis[i], np.pi/2)
    basis += center
    return np.asarray(basis)

def ribbon(length,width,orientation = "a", start = np.array([0,0,0]),build = False):
    basis = hexagon(orientation,start)
    if orientation == "a":
        transVec = np.array([3*latConst,np.sqrt(3)*latConst,0])
    else:
        transVec = np.array([np.sqrt(3)*latConst, 3*latConst, 0])
    base = np.empty((0,3))
    for i in range(length):
        for j in range(width):
            newcoords = np.array([i*transVec[0], j*transVec[1],0]) + basis
            base = np.vstack([base,newcoords])
    if build is False:
        base = np.round(base,4)
    base = np.unique(base, axis=0)
    return base
    
def triangle(side, orientation="a"):
    if orientation == "a":
        transVec = np.array([3*latConst/2,3*np.sqrt(3)*latConst/2,0])
    else:
        transVec = np.array([np.sqrt(3)*latConst/2, 3*latConst/2, 0])
    base = ribbon(side,1,orientation)
    atoms = base
    for i in range(1,side):
        newlayer = ribbon(side-i,1,orientation) + np.array([i*transVec[0], i*transVec[1], 0])
        atoms = np.vstack([atoms,newlayer])
    xcm = 0
    ycm = 0
    for i in atoms:
        xcm += i[0]/len(atoms)
        ycm += i[1]/len(atoms)
    cm = np.array([xcm,ycm,0])
    atoms += -cm
    if orientation == "z":
        for i,j in enumerate(atoms):
            atoms[i] = rotate(atoms[i],np.pi/2)
    atoms = np.round(atoms, 2)
    atoms = np.unique(atoms, axis=0)
    return atoms

def flake(side, orientation="a"):
    if orientation == "a":
        transVec = np.array([[3*latConst/2,3*np.sqrt(3)*latConst/2,0],[3*latConst/2,-3*np.sqrt(3)*latConst/2,0]])
    else:
        transVec = np.array([[np.sqrt(3)*latConst/2, 3*latConst/2, 0],[np.sqrt(3)*latConst/2, -3*latConst/2, 0]])
    base = hexagon(orientation)
    atoms = np.empty((0,3))
    for i in range(side):
        for j in range(side):
            tmp = base + i*transVec[0] + j*transVec[1]
            atoms = np.vstack([atoms,tmp])
    removeList = []
    for index,atom in enumerate(atoms):
        cutoff = transVec[0][1]*((side+1*int(orientation == "z"))/2)
        if atom[1] > cutoff or atom[1] < -cutoff:
            removeList.append(index)
    for index in removeList[::-1]:
        atoms = np.delete(atoms,index, axis=0)
    xcm=0
    for atom in atoms:
        xcm += atom[0]
    xcm = xcm/len(atoms)
    cm = np.array([xcm,0,0])
    atoms -= cm
    if orientation == "z":
        for i,j in enumerate(atoms):
            atoms[i] = rotate(atoms[i],np.pi/2)
    atoms = np.round(atoms,4)
    atoms = np.unique(atoms,axis=0)
    return atoms

def crosssheet(length,width):
    base1 = ribbon(length,width,orientation="a")
    base2 = ribbon(length,width,orientation="z")
    xcm1 = sum([coord[0] for coord in base1])/len(base1)
    ycm1 = sum([coord[1] for coord in base1])/len(base1)
    cm1 = np.array([xcm1,ycm1,0])
    base1 -= cm1
    xcm2 = sum([coord[0] for coord in base2])/len(base2)
    ycm2 = sum([coord[1] for coord in base2])/len(base2)
    cm2 = np.array([xcm2,ycm2,0])
    base2 -= cm2
    for i,j in enumerate(base2):
        base2[i] = rotate(base2[i],np.pi/2)
    cross = np.concatenate((base1,base2),axis=0)
    cross = np.round(cross,2)
    cross = np.unique(cross,axis=0)
    return cross

def junction(length,width,orientation="a"):
    base = ribbon(length,width,orientation)
    if orientation is "a":
        base -= np.array([(width%2-1)*3*latConst/2,sum([coord[1] for coord in base])/len(base),0])
    else:
        base -= np.array([(width%2-1)*np.sqrt(3)*latConst/2,sum([coord[1] for coord in base])/len(base),0])
    arm1 = base*np.array([-1,1,1])
    for i,j in enumerate(arm1):
        arm1[i] = rotate(arm1[i], -np.pi/3)
    arm2 = arm1*np.array([1,-1,1])
    body = np.concatenate((base,arm1,arm2),axis=0)
    if orientation is "z":
        for i,j in enumerate(body):
            body[i] = rotate(body[i],-np.pi/2)
    body = np.round(body,2)
    body = np.unique(body,axis=0)
    return body
    
def indexspace(cell):
    indices = [(int(np.round(atom[0]/(3*latConst/2))), int(np.round(atom[1]/(0.5*latConst*np.sqrt(3)))))\
               for atom in cell]
    return indices

def fluxtubes(cell,sigma,B):
    tubes = []
    indices = []
    for atom in cell.atoms:
        for neighbor in atom.neighbors:
            if abs(atom.coord[1]-neighbor.coord[1]) < 0.05:
                x = (atom.coord[0]+neighbor.coord[0])/2
                y = (atom.coord[1]+neighbor.coord[1])/2 -latConst*np.sqrt(3)/2
                rsq = x**2 + y**2
                tmp = [(atom.Spaceindex,neighbor.Spaceindex),np.round(B*np.exp(-rsq/(2*sigma**2)),2)]
                if tmp not in tubes:
                    tubes.append(tmp)
                    indices.append(atom.MUCindex[1])
    tubez = copy.deepcopy(tubes)
    for i in range(len(tubez)):
        for j in range(i):
            if tubez[i][0][0][0] == tubes[j][0][0][0] and tubez[i][0][0][1] > tubes[j][0][0][1]:
                tubez[i][1] += tubes[j][1]
    return tubez, indices

""" Main object we build our lattice with """
class Carbon(object):
    @initializer
    def __init__(self, coord, Rvec=np.zeros(3), UCindex=[[0,0],0], SCindex=[[0,0],0], MUCindex=[[0,0],0], neighbors=[]):
        pass
    def dump(self):
        print(repr(self.__dict__))
    def translate(self, translationVec):
        self.coord += translationVec
    def __str__(self):
        return 'Carbon ' + str(self.MUCindex[1]) + ' at ' + str(np.round(self.coord,2))+ " with UCindex " +\
        str(self.UCindex) + " and SCindex " +str(self.SCindex) + '\n' + 'with ' \
        + 'neighbors ' + '\n' + NeighborStr(self)

'''  Our main simulation cell composed of carbon objects '''
class SimulationCell:
    @initializer
    def __init__(self, N1=0, N2=0, n1=0, n2=0, Lattice="square", q=1, M=0, periodic = True):
        """ Initialization parameters determine the size of the 
        simulation cell that will be used """
        if Lattice is "custom":
            self.periodic = False
            self.populateHexagons(M)
            self.populateNeighbors(1.1*latConst, 1, 1)
        else:
            self.basisVec, self.UClatVec = getlatinfo(Lattice)
            self.populateCell(N1, N2, n1, n2)
            if Lattice is "flake":
                self.periodic = False
                self.MUClatVec = self.SClatVec
                self.makeFlake(N1, N2)
                self.populateNeighbors(1.1*latConst, 1, 1)
            else:
                if periodic:
                    self.duplicate(q,1)
                else:
                    for index, atom in enumerate(self.atoms):
                        atom.MUCindex = [[0,0], index]
                self.populateNeighbors(1.1*latConst, 1, 1)
                
#    def bar(self):
#        profile.runctx('self.populateNeighbors(1.5,1,1)', globals(), locals())
    
    def dump(self):
        print(repr(self.__dict__))
    
    def populateCell(self, N1, N2, n1, n2):
        cellRvec = n1*self.UClatVec[0] + n2*self.UClatVec[1]
        self.SClatVec = np.array([(N1-n1)*self.UClatVec[0], (N2-n2)*self.UClatVec[1]])
        self.atoms = [Carbon(self.basisVec[k]+i*self.UClatVec[0]+j*self.UClatVec[1], Rvec = cellRvec, \
        UCindex = [np.array([i,j]),k]) for i in range(N1) for j in range(N2)\
        for k in range(len(self.basisVec))]
        for index, atom in enumerate(self.atoms):
            atom.SCindex = [np.array([0,0]), index]
            
    def populateHexagons(self, M):
        angle = np.pi/3
        basis = []
        for i in range(6):
            rotMatrix = np.array([[np.cos(i*angle), -np.sin(i*angle), 0], \
                        [np.sin(i*angle), np.cos(i*angle), 0],[0, 0, 1]])
            rotatedcoord = np.dot(rotMatrix, np.array([latConst,0,0]))
            basis.append(rotatedcoord)
        rows = len(M)
        cols = len(M[0])
        tmpCell = []
        self.gridxy = []
        for row in range(rows):
            for col in range(cols):
                transVec = np.array([col*(3*latConst/2), (rows-row+(col%2)/2)*(latConst*np.sqrt(3)), 0])
                if M[row][col] == 1:
                    newcoords = basis + transVec
                    for coord in np.round(newcoords,4):
                        tmpCell.append(coord)
                self.gridxy.append([transVec,[row,col],M[row][col]])
        hexagons = np.unique(tmpCell, axis=0)
        self.atoms = [Carbon(coord, MUCindex = [[0,0],index]) for index, coord in enumerate(hexagons)]
        
    def populateWlatticepts(self, array):
        self.periodic = False
        self.atoms = []
        self.atoms = [Carbon(coord, MUCindex = [[0,0],index]) for index, coord in enumerate(array)]
        self.indexspace()
        self.populateNeighbors(1.1*latConst, 1, 1)
        self.Lattice = "custom"
        
    def duplicate(self, M1, M2):
        newCell = []
        index = 0
        for m1 in range(M1):
            for m2 in range(M2):
                for atom in self.atoms:
                    tmpAtom = copy.deepcopy(atom)
                    tmpAtom.translate(m1*self.SClatVec[0]+m2*self.SClatVec[1])
                    tmpAtom.UCindex[0] += np.array([m1*self.N1, m2*self.N2])
                    tmpAtom.MUCindex = [np.array([0,0]),index]
                    tmpAtom.SCindex[0] = [m1,m2]
                    newCell.append(tmpAtom)
                    index += 1
        self.MUClatVec = np.array([M1*self.SClatVec[0], M2*self.SClatVec[1]])
        self.atoms = newCell
        
    def translate(self, translationVec):
        """ Simple translation method to translate all the objects in the cell """
        for atom in self.atoms:
            atom.translate(translationVec)
            atom.Rvec += translationVec
            
    def remove(self, removearray):
        del self.atoms[removearray]
        for index, atom in enumerate(self.atoms):
            atom.MUCindex[1] = index
        self.populateNeighbors(1.1*latConst, 1, 1)
            
    def makeFlake(self, N1, N2):
        xlowercut = (N1/2-1)*self.UClatVec[0][0] + latConst/2
        xuppercut = (3*N1/2)*self.UClatVec[0][0] - latConst
        removeList = []
        for atom in self.atoms:
            if atom.coord[0] < xlowercut or atom.coord[0] > xuppercut:           
                removeList.append(atom.SCindex[1])
        for item in removeList[::-1]:
            del self.atoms[item]
        for index, atom in enumerate(self.atoms):
            atom.SCindex[1] = index
            atom.MUCindex = atom.SCindex
            
    def rotate(self, angle, pivot = np.array([0,0,0])):
        """ Rotates the atoms in the cell around a given point in space
        by an axis parallel to z axis """
        rotMatrix = np.array([[np.cos(angle), -np.sin(angle), 0], \
            [np.sin(angle), np.cos(angle), 0],[0, 0, 1]])
        self.translate(-pivot)
        for atom in self.atoms:
            atom.coord = np.dot(rotMatrix, atom.coord)
        self.translate(pivot)
    
    def indexspace(self):
        for atom in self.atoms:
            atom.Spaceindex = [int(np.round(atom.coord[0]/(3*latConst/2))), int(np.round(atom.coord[1]/(0.5*latConst*np.sqrt(3))))]
    
    def resetNeighbors(self):
        for atom in self.atoms:
            atom.neighbors = []

    def populateNeighbors(self, rCut, N1, N2):
        """ Replicates the main cell and for every object in the main cell,
        creates neighbors based on a distance condition , rCut : cutoff radius"""
        self.resetNeighbors()
        superCell = []
        self.indexspace()
        if not self.periodic:
            superCell = copy.deepcopy(self.atoms)
        else:
            for i in range(-N1, N1+1):
                for j in range(-N2, N2+1):
                    tmpCell = copy.deepcopy(self)
                    tmpCell.translate(i*self.MUClatVec[0] + j*self.MUClatVec[1])
                    for atom in tmpCell.atoms:
                        atom.UCindex[0] += np.array([i*self.N1*self.q, j*self.N2]) ## our assumption is that the MUC is generated by duplicating in the first lattice vector direction
                        atom.MUCindex[0] = [i,j]
                        superCell.append(atom)
        for atom in self.atoms:
            atom.neighbors = []
            for neighbor in superCell:
                separationVec = neighbor.coord - atom.coord
                separationDist = dot(separationVec,separationVec)**0.5
                if separationDist < rCut and separationDist > 0.1:
                    tmpAtom = copy.copy(neighbor)
                    tmpAtom.sepDist = separationDist
                    tmpAtom.sepVec = separationVec
                    tmpAtom.sepAngle = np.arctan2(separationVec[1], separationVec[0])
                    atom.neighbors.append(tmpAtom)
        self.neighborCell = superCell
        
    def __str__(self):
        atoms = ''
        for atom in self.atoms:
            atoms += atom.__str__() + '\n'*(self.atoms.index(atom) != len(self.atoms)-1)
        return atoms


def solveSecularSO(lattice, kpath, orthogonal = True, alpha=0):
    """ For each kpoint in kpath, creates Hamiltonian and overlap matrix elements
    using the 4x4 interaction matrix and the neighbor list for each atom in lattice """
    Natoms = len(lattice.atoms)
    bands = np.empty((0,Natoms), float)
    for kpoint in kpath:
        H = np.zeros((Natoms,Natoms), dtype = complex)
        S = np.zeros((Natoms,Natoms), dtype = complex)
        for atom in lattice.atoms:
            for neighbor in atom.neighbors:
#                blochPhase = np.exp(np.complex(0, np.dot(kpoint,neighbor.sepVec)))
                blochPhase = np.exp(np.complex(0, np.dot(kpoint,neighbor.Rvec-atom.Rvec)))
                PP = GetPeierlsPhase(lattice, atom, neighbor, alpha)
                H[atom.MUCindex[1]][neighbor.MUCindex[1]] += Vpi*blochPhase*PP/(latConst**2)
                S[atom.MUCindex[1]][neighbor.MUCindex[1]] += Spi*blochPhase*PP/(latConst**2)
        S += np.eye(Natoms)
        H += -V2s*np.eye(Natoms)
        if orthogonal:
            S = np.identity(Natoms)
            energies, eigvecs = la.eigh(H,S)
        else:
            energies, eigvecs = la.eigh(H,S)
#        energies = np.sort(np.real(energies))  ## NOT NEEDED SINCE EIGH SORTS EIGVALS
        energies = np.real(energies)
        bands = np.vstack([bands, energies])
    return H, S, bands, eigvecs

def solveSecularAO(lattice, kpath, alpha=0):
    """ For each kpoint in kpath, creates Hamiltonian and overlap matrix elements
    using the 4x4 interaction matrix and the neighbor list for each atom in lattice """
    Natoms = len(lattice.atoms)
    bands = np.empty((0,4*Natoms), float)
    for kpoint in kpath:
        H = np.zeros((4*Natoms,4*Natoms), dtype = complex)
        S = np.zeros((4*Natoms,4*Natoms), dtype = complex)
        for atom in lattice.atoms:
            for neighbor in atom.neighbors:
                blochPhase = np.exp(np.complex(0, np.dot(kpoint,neighbor.Rvec-atom.Rvec)))
                PP = GetPeierlsPhase(lattice, atom, neighbor, alpha)
                hm, sm = intMatrix(neighbor.sepAngle, neighbor.sepDist, orbitals = "all")
                H[4*atom.MUCindex[1]:4*atom.MUCindex[1]+4, 4*neighbor.MUCindex[1]:4*neighbor.MUCindex[1]+4] += hm*blochPhase*PP
                S[4*atom.MUCindex[1]:4*atom.MUCindex[1]+4, 4*neighbor.MUCindex[1]:4*neighbor.MUCindex[1]+4] += sm*blochPhase*PP
        S += np.eye(4*Natoms)
        H += np.diag(np.tile([V2s, V2p, V2p, V2p], Natoms))
        energies, eigvecs = la.eigh(H,S)
        energies = np.sort(np.real(energies))
        bands = np.vstack([bands, energies])
    return H, S, bands, eigvecs

gamma = [np.array([0,0,0])]

def butterfly(latticetype,N1,N2,qfactor,orbitals):
    SimCell = SimulationCell(N1, N2, Lattice = latticetype, q = qfactor)
    totBands = []
    Htot = []
    eigfuncs = []
    Stot = []
    plt.close()
    plt.figure(figsize=(12,9))
    for i in range(0,SimCell.q+1):
        if orbitals == "s":
            H,S,band,eigs = solveSecularSO(SimCell, gamma, alpha = i/SimCell.q)
        else:
            H,S,band,eigs = solveSecularAO(SimCell, gamma, alpha = i/SimCell.q)
        totBands.append(band)
        Htot.append(H)
        Stot.append(S)
        eigfuncs.append(eigs)
        print(i)
    for i in range(len(totBands[0][0])-1):
        xx = [item/SimCell.q for item in list(range(SimCell.q+1))]
        plt.plot(xx,[bands[0][i] for bands in totBands],'r.',markersize='1')
    plt.xlabel("p/q (unitless, magnetic field strength)")
    plt.ylabel("Energy (eV)")
    plt.xlim((0,1))
    figname = "bands"+str(N1)+"-"+str(N2)+"-"+str(SimCell.q)+".png"
    plt.savefig(figname,bbox_inches='tight', dpi = 300)
    return Htot, totBands, SimCell, eigfuncs

start = time.time()

Htot, ads, cell, eigfuncs = butterfly('PG',1,2,47,orbitals="s")

end = time.time()

print(end-start)
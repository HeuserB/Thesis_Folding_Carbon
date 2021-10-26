#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow
import matplotlib as mpl
from matplotlib.patches import Arc
import matplotlib.patches as mpatches
import subprocess
import pathlib

sys.path.append("../functions")

from Application_functions import *
from functions_folding import *
from Unfolding import Unfolding
atoms = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 28, 29, 30, 31],dtype=int)
pattern_start = "Input\s*orientation:\s*-*\s*Center\s*Atomic\s*Atomic\s*Coordinates\s*\(Angstroms\)\s*Number\s*Number\s*Type\s*X\s*Y\s*Z\s*-*"
pattern_start = "orientation:\s*-*\s*Center\s*Atomic\s*Atomic\s*Coordinates\s*\(Angstroms\)\s*Number\s*Number\s*Type\s*X\s*Y\s*Z\s*-*"
pattern_end = "\s+-+\s+"
NA = np.newaxis


# In[2]:


def list_files(directory, extension=".log"):
    file_list = []
    for file in os.listdir(directory):
        if file.endswith(extension):
            file_list.append(os.path.join(directory, file))
    return file_list


# In[3]:


def read_functional(txt):
    keyword = re.split("Done:\s*E\(",txt)[1]
    keyword = re.split("\)\s*=\s*",keyword)[0]
    
    if bool(re.search("Opt=",txt)) == False:
        functional = re.split("R",keyword)[-1]
        functional = re.sub(r'([A-Z])\1', lambda pat: pat.group(1).lower(), str(functional))
        basis_set = re.split("Standard\sbasis:\s+",txt)[1]
        basis_set = re.split("\s",basis_set)[0]

    else:
        tmp = re.split("Opt=",txt)[1]
        tmp = re.split("\s",tmp)[1]
        functional, basis_set = re.split("/",tmp)
    keyword ="E\(" + keyword + "\) =\s+"
    return functional, basis_set, keyword


# In[4]:


def read_energies(txt):
    start = "B3LYP\)\s+=\s+"
    end = "\s+A.U.\safter"
    energies = re.compile(start, re.IGNORECASE).split(txt)[1:]
    E = ""
    for energy in energies:
            tmp = re.split(end, energy)[0]
            E += tmp + ","
    E = np.fromstring(E,sep=',')
    return E


# In[5]:


def read_atoms(txt):
    tmp = re.split("Multiplicity\s=\s[0-9]",txt)[-1]
    tmp = re.split("\s+AtmSel",tmp)[0]
    atoms = re.sub("[^a-zA-Z]", '', tmp)
    atoms = re.findall('[A-Z][^A-Z]*', atoms)
    return atoms


# In[6]:


def read_geometries(txt, pattern_start=pattern_start, pattern_end=pattern_end):
    geometries = re.split(pattern_start,txt)[1:]
    geometries_array = []
    for geometry in geometries:
        tmp = re.split(pattern_end,geometry)[0]
        tmp = re.sub("[0-9]+\s+[0-9]+\s+0\s+","",tmp)
        tmp_np = np.fromstring(tmp,sep=' ').reshape(-1,3)
        geometries_array.append(tmp_np)
    return np.array(geometries_array)[1:]


# In[7]:


def form_datasets(directory, title, output_directory = ""):
    file_list = list_files(directory)
    # create a list of all the dataset that need to be created, with functionals, halogen, initial angle and basis set
    geometry_list = []
    functionals = [[],[]]
    basis_sets = [[],[]]
    radii,  E_inits, E_finals = [[],[]], [[],[]], [[],[]]
    failed_list = []
    init = True
    pattern = re.compile("Normal termination of Gaussian")
    
    for file in file_list:
        print(file)
        f = open(file,'r')
        txt = f.read()
        f.close()

        if bool(re.search(pattern,txt)) == True:
            state = 0
        else:
            state = 1

        E = read_energies(txt)
        E_init, E_final = E[0], E[-1]
        geometries = read_geometries(txt)
        d_CC = np.linalg.norm(geometries[-1][34] - geometries[-1][36])
        geometries = np.vstack([geometries[0][NA,...], geometries[-1][NA,...]])

        if init == True:
            geometry_list.append([np.empty([1,2,geometries.shape[-2], geometries.shape[-1]])])
            geometry_list.append([np.empty([1,2,geometries.shape[-2], geometries.shape[-1]])])
        init = False        
        radii[state] = np.append(radii[state],d_CC)
        E_inits[state] = np.append(E_inits[state],E_init)
        E_finals[state] = np.append(E_finals[state],E_final)
        geometry_list[state] = [np.append(geometry_list[state][0], geometries[NA,...], axis=0)]
    
        
    filename = output_directory + title + ".h5"    
    hdf_file = h5py.File(filename, 'w')
    
    state = 0
    id_sort = np.argsort(radii[state])
    print(id_sort)
    radii[state] = radii[state][id_sort]
    E_inits[state] = E_inits[state][id_sort]
    E_finals[state] = E_finals[state][id_sort]
    geometry_list[state][0] = geometry_list[state][0][1:,...]
    geometry_list[state][0] = geometry_list[state][0][id_sort]
        
    hdf_file.create_dataset("E_init", data=E_inits[state])
    hdf_file.create_dataset("E_final", data=E_finals[state])
    hdf_file.create_dataset("geometries", data=geometry_list[state][0])
    hdf_file.create_dataset("d_CC", data=radii[state])
        
        
    state = 1
    id_sort = np.argsort(radii[state])
    radii[state] = radii[state][id_sort]
    E_inits[state] = E_inits[state][id_sort]
    E_finals[state] = E_finals[state][id_sort]
    geometry_list[state][0] = geometry_list[state][0][1:,...]
    geometry_list[state][0] = geometry_list[state][0][id_sort]
        
    hdf_file.create_dataset("E_init_pending", data=E_inits[state])
    hdf_file.create_dataset("E_final_pending", data=E_finals[state])
    hdf_file.create_dataset("geometries_pending", data=geometry_list[state][0])
    hdf_file.create_dataset("d_CC_pending", data=radii[state])

    hdf_file.close()

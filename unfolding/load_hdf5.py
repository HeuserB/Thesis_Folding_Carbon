#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
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

sys.path.append("../thesis-carbon-folding/functions")
sys.path.append("../thesis-carbon-folding/Unfolding/")

#sys.path.append("../Unfolding")
#sys.path.append("../functions")
#sys.path.append("../Unfolding/Application_functions")
#sys.path.append("../../results/hdf5/")
#sys.path.append("../../fullerene-unfolding/")

# In[2]:


def load_h5(filename, remove_zeros=True):
    hf = h5py.File(filename, 'r')
    d_CC = np.array(hf.get("d_CC"))
    E_init = np.array(hf.get("E_init"))
    E_final = np.array(hf.get("E_final"))
    d_CC_pending = np.array(hf.get("d_CC_pending"))
    E_init_pending = np.array(hf.get("E_init_pending"))
    E_final_pending = np.array(hf.get("E_final_pending"))
    functional = np.array(hf.get("functional"))
    basis_set = np.array(hf.get("basis_set"))
    geometries = np.array(hf.get("geometries"))
    geometries_pending = np.array(hf.get("geometries_pending"))
    hf.close()
    if remove_zeros:
        mask = np.where(E_final!=0.)
        print("The calculations for the CC-distances: %sA yielded no result" %str(d_CC[mask]))
        d_CC, E_init, E_final, geometries = d_CC[mask], E_init[mask], E_final[mask], geometries[mask]
    return d_CC, E_init, E_final, geometries, d_CC_pending, E_init_pending, E_final_pending, geometries_pending
# %%
def merge(E, E_pending, Geo, Geo_pending, d_CC, d_CC_pending):
    d_CC_merged = np.concatenate([d_CC, d_CC_pending])
    pending = np.concatenate([np.zeros_like(d_CC,dtype=int), np.ones_like(d_CC_pending,dtype=int)])
    E_merged = np.concatenate([E, E_pending])
    Geo_merged = np.concatenate([Geo, Geo_pending])

    ids = np.argsort(d_CC_merged)
    E_merged = E_merged[ids]
    Geo_merged = Geo_merged[ids]
    d_CC_merged = d_CC_merged[ids]
    pending = pending[ids]
    return E_merged, Geo_merged, d_CC_merged, pending
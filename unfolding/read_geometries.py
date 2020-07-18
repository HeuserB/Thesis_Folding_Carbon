#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import re
import os
import sys
import seaborn as sns
import h5py

sys.path.append("../../thesis-carbon-folding/functions/")
from functions_folding import angle_vec


# In[2]:


def list_files(directory="results/", extension=".log"):
    file_list = []
    for file in os.listdir(directory):
        if file.endswith(extension):
            file_list.append(os.path.join(directory, file))
    return file_list


# In[4]:


def read_geometry(file):
    f = open(file, "r")
    txt = f.read()
    f.close()
    pattern = re.compile("Normal termination of Gaussian")
    if bool(re.search(pattern,txt)) == False:
        return None
    tmp = re.split("0,1\\\\",txt)[-1]
    tmp = re.split("\\\\\\\\Version",tmp)[0]
    tmp = re.sub('[A-Z]', '', tmp)
    tmp = re.sub("\s","",tmp)
    tmp = re.sub('[a-z]', '', tmp)
    tmp = tmp[1:]
    tmp = re.sub("\\\\",'',tmp)
    tmp = np.fromstring(tmp,sep=',').reshape(-1,3)
    return tmp


# In[ ]:





import numpy as np
import re
import h5py
NA = np.newaxis

def read_unfolding(filename):
    hf = h5py.File(filename, 'r')
    graph_unfolding_array = hf.get('graph_unfolding')
    graph_unfolding_faces_array = hf.get('graph_unfolding_faces')
    graph_faces_array = hf.get('graph_faces')
    dual_unfolding_array = hf.get('dual_unfolding')
    vertices_final = hf.get('vertices_final')
    vertices_final = np.array(vertices_final,dtype=np.float64).reshape(-1,3)
    bonds_toBe = np.array(hf.get('bonds_toBe'))
    lengths_toBe = np.array(hf.get('lengths_toBe'))
    angles_f = np.array(hf.get('angles_f'),dtype=np.float64)
    opt_geom = np.array(hf.get('opt_geom'),dtype=np.float64)
    halogene_positions = np.array(hf.get('halogene_postitions'))
    neighbours = np.array(hf.get('neighbours'))
    
    graph_unfolding = []
    dual_unfolding = []
    graph_unfolding_faces = []
    graph_faces = []
    
    count = 0
    for array, graph in zip([dual_unfolding_array,graph_unfolding_array, graph_unfolding_faces_array, graph_faces_array],[dual_unfolding,graph_unfolding, graph_unfolding_faces, graph_faces]):
        for i in range(len(array)):
            tmp = []
            for j in range(len(array[i])):
                if count < 2:
                    if array[i][j] != i:
                        tmp.append(array[i][j])
                else:
                    if not np.any(np.array(tmp) == array[i][j]):
                        tmp.append(array[i][j])
            graph.append(tmp)
        count += 1 
    hf.close()
    return dual_unfolding, graph_unfolding, graph_unfolding_faces, vertices_final, bonds_toBe, lengths_toBe, angles_f, opt_geom, halogene_positions, neighbours, graph_faces

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

def fit_plane(points):
    N = len(points)
    CM = (np.sum(points,axis=0)/N)
    M = points - CM[NA,:]
    U, S, V = np.linalg.svd(M)
    n = V[np.argmin(S)]
    ev = np.min(S)
    err = np.mean(np.abs((M@n)))
    d = CM@n
    return n, err, ev, d

def fit_all_planes(unfolding):
    geometry = unfolding.vertex_coords
    normals, errors, evals = np.zeros([len(unfolding.graph_unfolding_faces),3]), np.zeros([len(unfolding.graph_unfolding_faces)]), np.zeros([len(unfolding.graph_unfolding_faces)])
    ds = np.zeros([len(unfolding.graph_unfolding_faces)])
    for j, face in enumerate(unfolding.graph_unfolding_faces):
        points = geometry[face]
        n, err, ev, d = fit_plane(points)
        normals[j], errors[j], evals[j], ds[j] = n, err, ev, d
    return normals, errors, evals, ds
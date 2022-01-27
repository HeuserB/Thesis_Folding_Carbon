#!/usr/bin/python3
import sys, numpy as np
from numpy.lib.function_base import angle
from functions_folding import *
from geometry_functions import *
import data.C120D6_fat as data
from vedo import show, Spheres, Mesh, printc
from plot_vedo import plot_unfolding_vedo, MainWindow
from PyQt5 import Qt

bond_angles  = np.array([108.,120.])*np.pi/180
bond_lengths = np.array([1.458,1.401])

if(len(sys.argv)<2):
    print(f"Syntax: {sys.argv[0]} <unfolding_filename.npz> <unfolding_number:0>")
    sys.exit(-1)
    
unfolding_filename = sys.argv[1]
unfolding_number = 0
if(len(sys.argv)>=3): 
    unfolding_number = int(sys.argv[2])

f = np.load(unfolding_filename,allow_pickle=True)    
arcpos      = f['Arcpos'][unfolding_number]
pentagon_ix = f['Pentagon_ix'][unfolding_number]
path        = f['Paths'][unfolding_number]
isomer      = dict(zip(f['isomer_keys'][:], f['isomer_values'][:]))
root_node   = path[0,0,0]

print(f"Loaded unfoldings for isomer {isomer['name']}.")

dual_neighbours  = isomer['dual_neighbours']
cubic_neighbours = isomer['cubic_neighbours']
hexagons         = isomer['hexagons']
pentagons        = isomer['pentagons']

unfolding_subgraph = arcpos_to_unfolding(dual_neighbours,arcpos)
faces = faces_from_hp(dual_neighbours, hexagons, pentagons)

print(f"Root face is {faces[root_node]}, dual vertex {root_node}")

tree, hinges, connected_hinges = minimal_spanning_tree(unfolding_subgraph, root_node, faces)

tree, affected_vs, hinges, connected_hinges = hinges_traversed(unfolding_subgraph, faces, root_node)

N  = len(cubic_neighbours)
Nf = len(dual_neighbours)
gg = {u:unfolding_subgraph[u] for u in range(Nf)}
tt = {u:tree[u] for u in range(Nf) if tree[u] != []}
ff = {f:faces[f] for f in range(Nf)}

print(f"subgraph: {gg}")
print(f"faces: {ff}")
print(f"tree: {tt}\n")
print(f"hinges[0]: {hinges[0]}\n"
      f"hinges[1]: {hinges[1]}\n")
#print(f'face 9 contains atoms {faces[9]} : face 20 contains atoms {faces[20]}\n')
planar_geometry = draw_vertices_unfolding(unfolding_subgraph,faces,root_node,bond_angles,bond_lengths)

unfolding_normals = np.zeros((Nf,3),dtype=float)

pent_id, hex_id = face_type(faces)

hexagons, pentagons = [], []
for face in faces:
    if len(face) == 5:
        pentagons.append(face)
    elif len(face) == 6:
        hexagons.append(face)
hexagons, pentagons = np.array(hexagons), np.array(pentagons)

unfolding_normals[pent_id] = mean_normal(planar_geometry, pentagons)
unfolding_normals[hex_id] = mean_normal(planar_geometry, hexagons)

unfolding_faceids = [u for u in range(Nf) if unfolding_subgraph[u] != [] ] 
unfolding_faces = [faces[u] for u in unfolding_faceids]

plot_unfolding_vedo(planar_geometry,unfolding_faces)


X = isomer['points_opt']
closed_normals = np.zeros((Nf,3),dtype=float)
closed_normals[pent_id] = mean_normal(X, pentagons)
closed_normals[hex_id] = mean_normal(X, hexagons)
angles_final = [angle_vec(closed_normals[u],closed_normals[v],degrees=False) for u,v in hinges[0]]

angles_final = calculate_final_angles(X, unfolding_subgraph, hinges)

#for hinge in range(len(hinges[0])):
#for hinge in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
#    update_transform(planar_geometry, hinge, hinges, affected_vs, angles_final[hinge])

#fig = plot_unfolding(planar_geometry,faces,unfolding_faces)
#plt.show()

app = Qt.QApplication(sys.argv)
window = MainWindow(planar_geometry, unfolding_faces, hinges, affected_vs, angles_final)
app.aboutToQuit.connect(window.onClose)
app.exec_()

#angles_final = calculate_final_angles(closed_vertices, graph_unfolding_faces, hinges)
print(angles_final)

# For the folding up to work we will need:
# 
# dual_unfolding, = planar_geometry
# graph_unfolding_faces, = unfolding_subgraph
# graph_faces, = faces
# graph_unfolding, = !!! Subgraph of the unfolding 
# graph, = dual_neighbours
# halogen_positions=halogen_positions, 
# root_node=0, 
# bonds_toBe=bonds_toBe, 
# angles_f=angles_f

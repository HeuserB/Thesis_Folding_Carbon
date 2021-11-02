#!/usr/bin/python3
import numpy as np
from functions_folding import *
import data.C120D6_fat as data

bond_angles  = np.array([108.,120.])*np.pi/180
bond_lengths = np.array([1.458,1.401])

f = np.load("C120D6_fat-output.npz")

arcpos      = f['Arcpos'][0]
pentagon_ix = f['Pentagon_ix'][0]
path        = f['Paths'][0]

root_node = path[0,0,0]

unfolding_subgraph = arcpos_to_unfolding(data.dual_neighbours,arcpos)

faces = faces_from_hp(data.dual_neighbours, data.hexagons,data.pentagons)

tree, hinges, connected_hinges = minimal_spanning_tree(unfolding_subgraph, root_node, faces)

tree, affected_vs, hinges, connected_hinges = hinges_traversed(unfolding_subgraph, faces, root_node)

gg = {u:unfolding_subgraph[u] for u in range(62)}
tt = {u:tree[u] for u in range(62) if tree[u] != []}
#print(f"subgraph: {gg}")
#print(f"tree: {tt}\n\n")
#print(f"hinges[0]: {hinges[0]}\n\n"
#      f"hinges[1]: {hinges[1]}\n")
print(f'face 9 contains atoms {faces[0]} : face 20 contains atoms {faces[20]}\n')
#planar_geometry = draw_vertices_unfolding(unfolding_subgraph,faces,root_node,bond_angles,bond_lengths)



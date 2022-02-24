'''
This module includes all functions that are related to the graph theory part of the unfolding
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import copy, deepcopy
import re
import h5py
from matplotlib import cm
import sys
from geometry_functions import *

#from numpy.lib.shape_base import get_array_wrap

NA = np.newaxis

#bonding_lengths = np.array([1.458,1.401]) 
#bond_angles = np.radians(np.array([108.,120.]))
#k = np.array([5,6]) *100

def internal_faces(dual_graph, arcpos):
    Nf = len(dual_graph)
    
    internal_arc = np.zeros((Nf,6),bool)
    
    for u in range(Nf):
        for j in range(len(dual_graph[u])):
            xu,xv = arcpos[u][j]    # u->v
            v = dual_graph[u][j]

            rj = dual_graph[v].index(u)
            rxv, rxu = arcpos[v][rj] # v->u

            internal_arc[u,j] = ((xv == rxv) & (xu == rxu)).all()

    number_internal_arcs = np.sum(internal_arc.astype(int),axis=1)
    return number_internal_arcs

def unfolding_faces(dual_graph, arcpos, pent_ix, number_internal_arcs):

    unfolding_face = (number_internal_arcs==6) # Hexagons that have all arcs internal are in the unfolding
    
    # Which pentagons are included in the unfolding? Only the ones that have exactly one edge split and
    # a maximum of two adjacent neighbour faces included in the unfolding. 
    for p in pent_ix:
        too_many_adjacent_neighours = False
        for j in range(5):
            neighbours = dual_graph[p]
            s,t,r = neighbours[j], neighbours[(j+1)%5], neighbours[(j+2)%5]

            too_many_adjacent_neighours |= (unfolding_face[s] & unfolding_face[t] & unfolding_face[r])

        unfolding_face[p] = (number_internal_arcs[p]==4) & ~too_many_adjacent_neighours
#        print(f"pentagon {p} -> unfolding_face: {unfolding_face[p]}")

    return unfolding_face


# graph: dual graph subgraph of faces included in unfolding: dual nodes for faces not included have []-entries
# root:  root face, node in dual graph
# faces: atoms in faces (cubic graph node ids)
def minimal_spanning_tree(graph,root,faces):
    '''
    INPUT:
    graph:  list[list[int]]; list of list of the graph, which face is neighobour of which face 
    root:   int; index of the root face to create MST
    faces:  list[list[int]]; list of list, which atom indices make up which face

    OUTPUT:
    tree:  list[list[int]]; list of lists of the face subgraph in a spanning tree manner
    hinges:  list[list[[int_a,int_b],...] , list[[int_c,int_d],...]] list of two lists: first is the hinges as lists of parent hinge 
             int_a and child int_b and the second list as the corresponding right-handed vertex connection int_c and int_d 
    connected_hinges: 

    This function will return the faces between which we have hinges in a spanning tree manner
    '''

    ### classic MST algorithm part
    if root >= len(graph): 
        print("Please enter a valid root node!")
        return
    queue = []
    hinges = []
    visited = [False] * len(graph)
    queue.append(root)
    tree = [ [] for i in range(len(graph)) ]
    connected_hinges = [ [] for i in range(len(graph)) ]
    visited[root] = True
    hinge = 0

    while queue:
            # Take a vertex from the queque
            s = queue.pop(0)
            # Get all adjacent vertices of the 
            # dequeued vertex s. If a adjacent 
            # has not been visited, then mark it 
            # visited and queue it 
            for v in graph[s]:
                if visited[v] == False:
                    tree[s].append(v)
                    queue.append(v)
                    visited[v] = True
                    hinges.append([s,v])
                    connected_hinges[s].append(hinge)
                    hinge += 1


    hinges_traversed = []
    for i,j in hinges:
        hinge = []
        flip = False
        found_first_vertex = False

        ### Test some stuff for simplification
        intersection = set(faces[i]) & set(faces[j])
        #print(f"The intersection of face {i} ({faces[i]}) and face {j} ({faces[j]}) are: {intersection}")
        ###

        # Iterate through all vertices of a parent face i
        for u in faces[i]:
            found_vertex = False
            # Iterate through all vertices of the child face j
            for v in faces[j]:
                # If parent and child face share the vertex add it to the hinge axis and set found_first_vertex and found_vertex to True
                if u == v:
                    hinge.append(u)
                    found_first_vertex = 1
                    found_vertex = True
                    break
             # Is the axis complete ?
            if len(hinge) == 2:
                    break
             # If the parents face vertex is not found in the chuld face
            if found_vertex == False:
                 # But the first vertex has been found     
                if found_first_vertex == 1:
                    # Vertices in between have been skipped and the axis is not in clockwise order -> flip it!
                    flip = True
            
        if flip == True:
            hinges_traversed.append(hinge[::-1])
        else:
            hinges_traversed.append(hinge)
            
    hinges = [hinges, hinges_traversed]
    return tree, hinges, connected_hinges

def spanning_tree(graph_orig, root, faces, duplicates = False):
    if root >= len(graph_orig): 
        print("Please enter a valid root node!")
        return
    MST = minimal_spanning_tree(graph_orig, root)
    graph = deepcopy(graph_orig)
    queue = []
    hinges = []
    queue.append(root)
    tree = [ [] for i in range( len(graph)) ]
    connected_hinges = [ [] for i in range(len(graph)) ]
    hinge = 0 
    while queue:
            # Take a vertex from the queque
            s = queue.pop(0)
            
            # Get all adjacent vertices of the 
            # dequeued vertex s. If a adjacent 
            # has not been visited, then mark it 
            # visited and queue it 
            for i in graph[s]:
                tree[s].append(i)
                queue.append(i)
                graph[i].remove(s)
                hinges.append([s,i])
                if duplicates == True:
                    if depth_in_tree(MST,i, root) == depth_in_tree(MST,s, root):
                        tree[i].append(s)
                connected_hinges[s].append(hinge)
                hinge += 1

    hinges_traversed = []
    for i,j in hinges:
        #print("Hinge: %s - %s\n" %(i,j))
        hinge = []
        flip = False
        tmp = 0
        for u in faces[i]:
            search = False
            #print("Looking for vertex %s in face %s" %(u,j))
            for v in faces[j]:
                if u == v:
                    #print("Found!\n")
                    hinge.append(u)
                    tmp = 1
                    search = True
                    break
            #print("Length of hinge is %s" %len(hinge))
            if len(hinge) == 2:
                    #print("Hinge is complete!\n")
                    #print(hinge)
                    break
                    
            if search == False:        
                #print("We did not find vertex %s in face %s" %(u,j))
                if tmp == 1:
                    #print("We found vertex %s in face %s but vertex %s is not there" %(hinge[0], j, u))
                    flip = True
            
        if flip == True:
            hinges_traversed.append(hinge[::-1])
        else:
            hinges_traversed.append(hinge)
            
    hinges = [hinges, hinges_traversed]

    return [tree, hinges, connected_hinges]


def tree_depth(tree,root):
    d = 0
    for c in tree[root]:
        d = max(d, tree_depth(tree,c)+1)
    return d

def hinge_order(tree, root, dual_hinges):
    order = [[] for i in range(tree_depth(tree,root))]
    for i, hinge in enumerate(dual_hinges):
        p,c   = hinge
        depth = depth_in_tree(tree,p,root)
        order[depth] += [i]

    return order

def depth_in_tree(tree, node, root):
    depth = 1
    if node == root:
        return 0
    test_node = root
    this_level = [ i for i in tree[root]]
    next_level = []
    for i in this_level:
        for j in tree[i]:
            next_level.append(j)

    while test_node != node:
        if len(this_level) == 0:
            depth += 1
            this_level = next_level
            next_level = []
            for i in this_level:
                for j in tree[i]:
                    next_level.append(j)
        else:
            test_node = this_level.pop(0)
    return depth

def get_rotational_axis(hinge_0, hinges, coordinates_3D):
    axis_cart = coordinates_3D[hinges[1][hinge_0][0]] - coordinates_3D[hinges[1][hinge_0][1]] 
    return  - axis_cart

def true_angle(a,b,degree = False, epsilon=1e-10):
    same = np.sum(np.abs(a - b) <= epsilon, axis=-1)
    angle = np.zeros(a.shape[:-1])
    angle[np.where(same == 3)] = 0.
    angle[np.where(same != 3)] = angle_vec(a[np.where(same != 3)], b[np.where(same != 3)],degrees=degree)
    return angle

def all_child_nodes(MST, node, faces):
    # returns all children of a given node
    
    child_nodes = [node]
    queue = []
    for i in MST[node]:
        child_nodes.append(i)
        queue.append(i)
    while queue:
        tmp = MST[queue.pop(0)]
        for i in tmp:
            child_nodes.append(i)
            queue.append(i)
    # now we want the face indices for the child nodes
    child_vertices = faces[node].copy()
    for i in child_nodes:
        vertices = faces[i]
        for j in vertices:
            new = True
            for k in child_vertices:
                if j == k:
                    new = False
            if new == True:
                child_vertices.append(j)

    return child_vertices

def child_nodes(MST, node):
    child_nodes = []
    queue = []
    for i in MST[node]:
        child_nodes.append(i)
        queue.append(i)
    while queue:
        tmp = MST[queue.pop(0)]
        for i in tmp:
            child_nodes.append(i)
            queue.append(i)
    return child_nodes

def affected_children(tree, hinges, faces, root):
    aff_children = [False] * len(hinges[0])
    affected_vertices = [[]] * len(hinges[0])
    #tree_d, _, _ = spanning_tree(graph, root, faces, duplicates = True)
    
    for hinge_id in range(len(hinges[0])):
        hinge = hinges[0][hinge_id]
        children = child_nodes(tree, hinge[1])
        children.insert(0,hinge[1])
        children = list(set(children))
        aff_children[hinge_id] = children
        tmp = []
        for face in children:
            for vertex in faces[face]:
                tmp.append(vertex)
        affected_vertices[hinge_id] = list(set(tmp))
    
    '''
    ### OLD STUFF ###
    # delete duplicates if their hinge has the same depth
    max_depth = len(hinges[0])
    affected_depth = []
    for i in range(max_depth):
        affected_depth.append([])

    for i in range(len(hinges[0])):
        hinge = hinges[0][i]
        depth = depth_in_tree(tree, hinge[1], root)
        for j in affected_vertices[i]:
            if j in affected_depth[depth]:
                affected_vertices[i].remove(j)
            else:
                affected_depth[depth].append(j)
    '''

    # delete duplicate vertices from the affected vertices
    max_depth = len(hinges[0])
    affected_depth = []
    for i in range(max_depth):
        affected_depth.append([]) # create a list of lists which contains all vertices that are affected by hinges at a certain depth

    for i in range(len(hinges[0])): # for all hinges 
        hinge = hinges[0][i]
        depth = depth_in_tree(tree, hinge[1], root) # first get their depth
        for j in affected_vertices[i]: # check for all affected vertices by this hinge if it has already been affected (1) by a hinge with higher level (2) and remove it from the vertices if that is the case (3)
            for d in np.arange(depth+1): # (2)
                #print(f"Currently working on depth: {d}")
                if j in affected_depth[d]: # (1)
                    affected_vertices[i].remove(j) #(3)
                    #print(f"Found vertex {j} and removed it moving on to next vertex")
                    break
                else:
                 affected_depth[depth].append(j)

    return affected_vertices

def hinges_traversed(graph, faces, root):
    
    # returns all the hinges as a list of two lists with the two faces and the dual vertices
    # the traversed hinges representing the vertices forming a hinge are in the correct order
    # meaning having the same chirality

    #tree, hinges, connected_hinges = spanning_tree(graph, root, faces)
    tree, hinges, connected_hinges = minimal_spanning_tree(graph, root, faces)
    
    affected_vertices = affected_children(tree, hinges, faces, root)

    return [tree, affected_vertices, hinges, connected_hinges]

def update_transform(coordinates_3D, hinge_0, hinges, affected_children, delta_phi):
    # This function will update all child node vertices which follow the given parent node
    # it will require the traversed hinges forming the rotational axis to be in the correct
    # manner, meaning right-handed or left-handed self-consitentley
    
   
    # the vertices which are connected to hinge_0 are connecting_faces
    # Note: this includes the two vertices forming the rotational axis
    # which will not be changed during a rotation as the connecting vectors are: (0,0,0) 
    affected = affected_children[hinge_0]

    #print(affected)
    
    # calculate all connecting vectors in the reference frame of the rotational axis
    origin = coordinates_3D[hinges[1][hinge_0][0]]
    reference_vertices = coordinates_3D[affected] - origin

    #print(origin)
    #print(reference_vertices)
    
    # calculate the rotational axis
    rot_axis =  get_rotational_axis(hinge_0, hinges,coordinates_3D)
    
    # caluculate the rotational matrix
    rot_matrix = rotation_matrix(rot_axis, delta_phi)
    
    #rotate and update the 3D coordinates
    new_vertices =  np.dot(rot_matrix,reference_vertices.T).T
    coordinates_3D[affected] = origin + new_vertices
    print(coordinates_3D)
    
    #print(new_reference_vertices)

def triangulate_polygone(polygones,num_of_vertices):
    """
    Return a set of triangulated faces for all polygones and return the indexes of the triangulation
    The created midpoints will allways be appended at the end of the vertices_extended (optimise later)
    """

    faces = np.empty([1,3],dtype=int)
    for i in range(len(polygones)):
        tmp = np.array(polygones[i])
        face = np.concatenate([tmp[:,None],np.hstack([tmp[1:],tmp[0]])[:,None],np.repeat(num_of_vertices + i,len(tmp))[:,None]],axis=1)
        faces = np.concatenate([faces,face])
    return faces[1:]

def unique_vertices(faces):
    vertices = np.array([0],dtype=np.int8)
    for face in faces:
        for vertex in face:
            vertices = np.append(vertices,vertex)
    return np.unique(vertices)
   
def draw_face(dual_planar, hinges, face, h,bond_angles, bonding_lengths):    
    '''
    This function draws a face either hexagon or pentagon on the two dimensional plane from a given mother hinge by rotating around it 
    '''
#    print(f"hinges[0]: {hinges[0]} \nhinges[1]: {hinges[1]}")
    u,v = hinges[0][h]          # hinges[0] = dual_hinges
    a,b = hinges[1][h]          # hinges[1] = cubic_hinges
    
    xa, xb = dual_planar[a], dual_planar[b]
    r      = xa-xb
    
    #print(f"Drawing face {v}: {face} connected by hinge {h}: {a,b}/{u,v}")
    
    shape_index = len(face)-5

    # look where the beginning of the hinge is located in the face 
    # use the beginning because the hinge and the child face are oriented differetly
    try:
        hinge_offset = face.index(a)
    except:
        print(f"Error: hinge vertices {hinge_vertices} are not part of face {face}")
        sys.exit()

    # roll the face by the id
    missing_atoms = np.roll(face,-hinge_offset-1)[:-2]
    #print(f"missing_atoms = {missing_atoms}")
    
    # successively create the new vertex points of the face starting with the hinge vector
    # theta is the angle between the old vectors, namley the pentagon or heaxon angles
    theta =  np.pi - bond_angles[shape_index]
    length = bonding_lengths[shape_index]

    r = (r / np.linalg.norm(r)) * length

    for c in missing_atoms:
        # rotate the vector connecting the two previous two hinges around the bonding angle
        r = rotate_vector(r,theta)
  
        # and add the rotated vector on the last vertex
        xc = xa + r
        dual_planar[c] = xc

        xb = xa
        xa = xc
        r = xa - xb

    return dual_planar

def draw_root_face(dual_planar,faces,root_node, bond_angles, bonding_lengths):
    
    face = faces[root_node]
    if len(face) == 5:
        index = 0
    elif len(face) == 6:
        index = 1
    else:
        print("Root face is neither pentagon nor hexagon")

    theta_0 = np.pi / len(face)
    length = bonding_lengths[index]
    dual_planar[face[0]][0] = length * 0.5
    dual_planar[face[0]][1] = (length * 0.5) / np.tan(theta_0)

    r  = dual_planar[face[0]] / np.linalg.norm(dual_planar[face[0]]) *  bonding_lengths[index]

    theta = bond_angles[index]

    r = rotate_vector(r, (np.pi / 2. - theta_0) + np.pi - theta)

    for i in range(1,len(face)):
        vertex = face[i]
        vertex_old = dual_planar[face[i-1]]
        vertex_new = vertex_old + r

        dual_planar[vertex] = vertex_new

        r = vertex_new - vertex_old
        r = rotate_vector(r,np.pi - theta)

    return dual_planar


def draw_vertices_unfolding(dual_graph, faces, root_node, bond_angles, bonding_lengths):
    num_of_vertices = len(unique_vertices(faces))
    
    dual_planar = np.zeros([num_of_vertices,2],dtype=np.float64)
    dual_planar = draw_root_face(dual_planar, faces, root_node, bond_angles, bonding_lengths)

#    print(f"root_node = {root_node}, root_face = {faces[root_node]}")
    
    tree, affected_children_tree, hinges, connected_hinges = hinges_traversed(dual_graph, faces, root_node)

    dual_hinges, cubic_hinges = hinges    

    # print(f"dual hinges:  {hinges[0]}\n"
    #       f"cubic hinges: {hinges[1]}\n")

    for h in range(len(dual_hinges)):
        u,v = dual_hinges[h]
        dual_planar = draw_face(dual_planar, hinges, faces[v], h, bond_angles, bonding_lengths)

    dual_planar = np.concatenate([dual_planar,np.zeros([dual_planar.shape[0],1])],axis=1)
    return dual_planar

def update(self):
    if len(self.unfolded_hinges) == 0:
        return      
          # this function should finish one hinge after the other

        # pick a random hinge fron the unfinished hinges
        #update_hinge = np.random.choice(self.unfolded_hinges)

    step = np.random.normal(self.step_size[self.active_hinge],scale = 0.01 * np.abs(self.step_size[self.active_hinge]))

    direction = - np.sign(self.angles_f[self.active_hinge] - self.angles[self.active_hinge])

        # rotate the vertex around the chosen hinge
    update_transform(self.vertices, self.active_hinge, self.hinges, self.affected_children, 
        delta_phi = step *  direction)

        # update the current angles
    self.angles[self.active_hinge] += step * direction


    self.vertices[self.num_of_vertices:] = self.vertices[self.graph_unfolding_faces].mean(axis=1)

    self.m1.setMeshData(
        vertexes=self.vertices, faces=self.faces, faceColors = self.color
        )
            
    deviation = np.abs(self.angles_f[self.active_hinge] - self.angles[self.active_hinge])
            
            
    if deviation < self.minimal_deviation_angles:
        print("Finished hinge %s" %self.active_hinge)
        self.unfolded_hinges.remove(self.active_hinge)
        if len(self.unfolded_hinges) != 0:
            self.active_hinge = self.unfolded_hinges[0]
        else:
            sys.exit(self.app.exec_())

def hex_and_pents(graph_unfolding_faces):
    pentagons = []
    hexagons = []
    for face in graph_unfolding_faces:
        if len(face) == 5:
            pentagons.append(face)
        elif len(face) ==6:
            hexagons.append(face)
    return np.array(pentagons), np.array(hexagons)

def make_graph_array(graph_unfolding, graph, halogen_positions, neighbours = 3):
    ### Take the uncomplete graph which is missing some of the bonds and complete it by adding hydrogens and halogens at the given positions ###
    ### 
    graph_array = np.zeros([len(graph_unfolding),neighbours],dtype=int)
    periphery = 0
    hydrogens = 0
    halogens = 0
    periphery_type = []
    hydrogen_positions = np.zeros(len(graph_unfolding),dtype=int)
    periphery_graph = []
    parent_atom = []
    ### Go through the whole graph and all its neighbours and see which ones are not in the graph of the unfolding ###
    for vertex in range(len(graph)):
        for j, neighbour in enumerate(graph[vertex]):  
            ### If they are not in the graph, set the corresponding neighbour to a hydrogen or halogen ###
            if (neighbour in graph_unfolding[vertex]) == False:
                if halogen_positions[vertex] == 0:
                    hydrogen_positions[vertex] = 1
                    # If we want the periphery sorted by halogens and hydrogens 
                    # graph_array[vertex,j] = len(graph_unfolding) + len(np.where(halogen_positions != 0)[0]) + hydrogens
                    graph_array[vertex,j] = len(graph_unfolding) + periphery
                    periphery_type.append(0)
                    hydrogens += 1
                else:
                    # If we want the periphery sorted by halogens and hydrogens 
                    #graph_array[vertex,j] = len(graph_unfolding) + halogens
                    graph_array[vertex,j] = len(graph_unfolding) + periphery
                    halogens += 1
                    periphery_type.append(1)
                ### In order to not double calculate the bond potentials and derivatives, set the peripherey array ###
                ### via which we can easily access the bond gradient in regard to the halogen/hydrogen-carbon bond ### 
                periphery_graph.append([vertex, j])
                periphery += 1
                parent_atom.append(vertex)
            else:
                graph_array[vertex,j] = graph[vertex, j]


    return graph_array, periphery, np.array(hydrogen_positions), np.array(periphery_graph), np.array(periphery_type), np.array(parent_atom)

def plot_graph(unfolding, savefig=False,filename=None):
    dual_planar = unfolding.vertex_coords
    midpoints = unfolding.midpoints
    fig, ax = plt.subplots()
    ax.scatter(dual_planar[:,0],dual_planar[:,1])
    for i in range(len(unfolding.graph_unfolding)):
        for j in unfolding.graph_unfolding[i]:
            tmp = np.stack([unfolding.vertex_coords[i,:-1],unfolding.vertex_coords[j,:-1]]).T
            ax.plot(tmp[0],tmp[1],tmp[2],'b-',lw=0.5)
    for i, txt in enumerate(np.arange(dual_planar.shape[0])):
        ax.annotate(txt,dual_planar[i][:2])
    ax.scatter(midpoints[:,0],midpoints[:,1])
    ax.axis('equal');
    if savefig == True:
        fig.savefig(filename)
    return fig

def plot_unfolding(dual_planar, faces, dual_subgraph, savefig=False, filename=None):
    fig, ax = plt.subplots()
    ax = fig.add_subplot(111, projection='3d')

    midpoints = np.zeros([len(dual_subgraph),3],dtype=np.float64)
    for faceid, face in enumerate(dual_subgraph):
        #        print(f'Drawing face {faceid} with vertices{face}')
        #midpoint = self.vertex_coords[self.graph_faces[face]].mean(axis=0)
        midpoint = np.mean(dual_planar[face],axis=0)
        midpoints[faceid] = midpoint
        for u, v in zip(face, np.roll(face,-1)):
            tmp = np.stack([dual_planar[u],dual_planar[v]]).T
            ax.plot(tmp[0],tmp[1],tmp[2],'b-',lw=0.5)

    ax.scatter(dual_planar[:,0],dual_planar[:,1],dual_planar[:,2])

    #for face in dual_subgraph:
    #    for u, v in zip(face, np.roll(face,-1)):
    #       tmp = np.stack([dual_planar[u],dual_planar[v]]).T
    #        ax.plot(tmp[0],tmp[1],tmp[2],'b-',lw=0.5)

    for i, txt in enumerate(np.arange(dual_planar.shape[0])):
        #ax.annotate(txt,dual_planar[i][:2])
        ax.text(*dual_planar[i], txt)

    for i, txt in enumerate(np.arange(midpoints.shape[0])):
        #ax.annotate(txt,midpoints[i][:2],color='r')
        ax.text(*midpoints[i], txt,color='r')

    ax.scatter(midpoints[:,0],midpoints[:,1],midpoints[:,2])
    #ax.axis('equal');
    if savefig == True:
        fig.savefig(filename)
    return fig

def plot_graph_3D(graph):
    X = graph.vertex_coords
    #%matplotlib notebook
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0],X[:,1],X[:,2], color="r")
    
    for vertex in range(len(graph.graph_unfolding)):
        for neighbour in graph.graph_unfolding[vertex]:
            line = np.vstack([graph.vertex_coords[vertex],graph.vertex_coords[neighbour]]).T
            ax.plot(line[0],line[1],line[2] )
    plt.show()
    return fig, ax

def plot_graph_H(unfolding, savefig=False, filename=None, gauss_numbering=False, double_hydrogens=np.array([])):
    dual_planar = unfolding.vertex_coords
    midpoints = unfolding.midpoints
    fig, ax = plt.subplots(figsize=(10,8))
    rot_ax = np.array([0,0,1])
    ax.scatter(dual_planar[:,0],dual_planar[:,1])
    for i in range(len(unfolding.graph_unfolding)):
        for j in unfolding.graph_unfolding[i]:
            tmp = np.stack([unfolding.vertex_coords[i,:-1],unfolding.vertex_coords[j,:-1]]).T
            ax.plot(tmp[0],tmp[1],'k-',lw=1.5)
    ax.scatter(dual_planar[:,0],dual_planar[:,1],color="blue",s = 200, label="Carbon")
    tag = dual_planar.shape[0]
    if gauss_numbering == True:
        tag += 1
    count_H = 0
    count_F = 0
    for i in range(len(unfolding.graph_unfolding)):
            
            if len(unfolding.graph_unfolding[i]) == 2:
                con_1 = unfolding.vertex_coords[unfolding.graph_unfolding[i][0]] - unfolding.vertex_coords[i]
                con_2 = unfolding.vertex_coords[unfolding.graph_unfolding[i][1]] - unfolding.vertex_coords[i]
                vec = (con_1 + con_2) / 2.0

                if (i in double_hydrogens):
                    vec = - vec / np.sqrt(np.sum(vec ** 2)) * 1.09
                    mat_1 = rotation_matrix(rot_ax,0.625)
                    vec_1 = np.dot(mat_1,vec.T).T
                    mat_2 = rotation_matrix(rot_ax,-0.625)
                    vec_2 = np.dot(mat_2,vec.T).T
                    vec_1 = unfolding.vertex_coords[i] + vec_1
                    vec = unfolding.vertex_coords[i] + vec_2
                    tmp_1 = np.stack([unfolding.vertex_coords[i,:-1],vec_1[:-1]]).T
                    tmp_2 = np.stack([unfolding.vertex_coords[i,:-1],vec[:-1]]).T
                    ax.plot(tmp_1[0],tmp_1[1],'k-',lw=1.5)
                    ax.scatter(vec_1[0], vec_1[1],color="green",s = 200,label="Hydrogen" if count_H == 0 else "")
                    ax.annotate(tag,vec_1[:2],size=25)
                    tag += 1
                    count_H += 1
                    ax.plot(tmp_2[0],tmp_2[1],'k-',lw=1.5)
                    ax.scatter(vec[0], vec[1],color="green",s = 200)
                    ax.annotate(tag,vec[:2],size=25)
                    tag += 1 

                elif (i in unfolding.halogen_parent_atom):
                    vec = vec / np.sqrt(np.sum(vec ** 2)) * 1.35
                    vec = unfolding.vertex_coords[i] - vec
                    tmp = np.stack([unfolding.vertex_coords[i,:-1],vec[:-1]]).T
                    ax.plot(tmp[0],tmp[1],'k-',lw=1.5)
                    ax.scatter(vec[0], vec[1],color="red",s = 200,label="Flouride" if count_F == 0 else "")
                    ax.annotate(tag,vec[:2],size=25)
                    tag += 1
                    count_F += 1
                else:
                    vec = vec / np.sqrt(np.sum(vec ** 2)) * 1.09
                    vec = unfolding.vertex_coords[i] - vec
                    tmp = np.stack([unfolding.vertex_coords[i,:-1],vec[:-1]]).T
                    ax.plot(tmp[0],tmp[1],'k-',lw=1.5)
                    ax.scatter(vec[0], vec[1],color="green",s = 200,label="Hydrogen" if count_H == 0 else "")
                    ax.annotate(tag,vec[:2],size=25)
                    tag += 1
                    count_H += 1

    if gauss_numbering==True:                
        for i, txt in enumerate(np.arange(1,1 + dual_planar.shape[0])):
            ax.annotate(txt,dual_planar[i][:2],size=25)

    else:                
        for i, txt in enumerate(np.arange(dual_planar.shape[0])):
            ax.annotate(txt,dual_planar[i][:2],size=25)
    plt.legend(fontsize="x-large")
    #ax.scatter(midpoints[:,0],midpoints[:,1])
    ax.axis('equal');
    ax.axis("off")
    if savefig == True:
        fig.savefig(filename)
    return fig

def neighbours_on_face(node, faces):
    face = 0
    for _ in range(len(faces)):
        if (node in faces[face]) == True:  
            break
        face +=1
    
    face = faces[face]
    next_on_face = face[(np.where(np.array(face)==node)[0][0] + 1)%len(face)]
    prev_on_face = face[(np.where(np.array(face)==node)[0][0] - 1)]
    return prev_on_face, next_on_face

def restart_header(header):
    header = re.split("#",header)[0] + "# Restart\n#" + re.split("#",header)[1] 
    return header

def xyz_to_string(coordinates):
    tmp = ""
    for j in coordinates:
                tmp += " "
                tmp += str(np.round(j,9))
    tmp += "\n"
    return tmp

def write_gaussfile(unfolding, header, double_hydrogens=np.array([],dtype=int), freeze=False, connectivity=False, writeFile=False, rotate=False, phis=[0.,0.], axes=None, atoms=None, filename="OutFile.com", freezelist=[], halogen=0, interpolated_angles = None):
        name = re.split("_initialise",filename)[0]
        name = re.split("/", name)[-1]
        #title_init = filename + "-initialise"
        #name = re.split("/", title_init)[-1]
        text = "%chk=/tmp/Gau-" + name + ".chk\n" + "%rwf=/tmp/Gau-" + name + ".rwf\n"

        header= deepcopy(header)
        text += header

        # write the C coordinates
        for idC, coords in enumerate(unfolding.vertex_coords[:unfolding.n_carbon]):
            text += " C "
            if freeze ==  True:
                if idC in freezelist:
                    text += "-1 "
                text += xyz_to_string(coords)
            else:
                # add some noise to avoid singular angles
                text += xyz_to_string(coords + np.random.normal(loc=0.,scale=1e-3,size=3)) 

        # give each C that only has two neighbours an H
        hydrogen_list = [None] * len(unfolding.graph_unfolding)
        num_of_hydrogen = 0
        for i in range(len(unfolding.graph_unfolding)):
            if len(unfolding.graph_unfolding[i]) == 2:
                hydrogen_list[i] = len(unfolding.graph_unfolding) + 1 + num_of_hydrogen
                num_of_hydrogen += 1
                con_1 = unfolding.vertex_coords[unfolding.graph_unfolding[i][0]] - unfolding.vertex_coords[i]
                con_2 = unfolding.vertex_coords[unfolding.graph_unfolding[i][1]] - unfolding.vertex_coords[i]
                vec = (con_1 + con_2) / 2.0
                #print(unfolding.graph_unfolding[i][0])

                if (i in double_hydrogens):
                    hydrogen_list[i] = [hydrogen_list[i], hydrogen_list[i] + 1]
                    num_of_hydrogen += 1
                    vec = - vec / np.sqrt(np.sum(vec ** 2)) * 1.09
                    ax = unfolding.vertex_coords[unfolding.graph_unfolding[i][0]] - unfolding.vertex_coords[unfolding.graph_unfolding[i][1]]
                    mat_1 = rotation_matrix(ax,0.925)
                    vec_1 = np.dot(mat_1,vec.T).T
                    mat_2 = rotation_matrix(ax,-0.925)
                    vec_2 = np.dot(mat_2,vec.T).T
                    vec_1 = unfolding.vertex_coords[i] + vec_1
                    vec = unfolding.vertex_coords[i] + vec_2
                    text += " H "
                    # add some noise to avoid singular angles
                    text += xyz_to_string(vec_1+ np.random.normal(loc=0.,scale=1e-3,size=3))
                    text += " H "

                elif (i in unfolding.halogen_parent_atom):
                    vec = - vec / np.sqrt(np.sum(vec ** 2)) * [1.35,1.76][halogen]
                    if rotate == True:
                        if i in atoms:
                            if np.any(unfolding.bonds_toBe == i) == True:
                                distance = np.linalg.norm(np.diff(unfolding.vertex_coords[unfolding.bonds_toBe[np.where(unfolding.bonds_toBe == i)[0]]],axis=1))
                                tmp = interpolated_angles(distance, halogen)
                                vec = vec / np.sqrt(np.sum(vec ** 2)) * tmp[1]
                            #ax = unfolding.vertex_coords[axes[i][1]] - unfolding.vertex_coords[axes[i][0]]
                            ax = unfolding.vertex_coords[neighbours_on_face(i,unfolding.graph_unfolding_faces)[0]] - unfolding.vertex_coords[neighbours_on_face(i,unfolding.graph_unfolding_faces)[1]]
                            mat = rotation_matrix(ax, tmp[0])
                            vec = np.dot(mat,vec.T).T
                    vec = unfolding.vertex_coords[i] + vec
                    text += [" F "," Cl "][halogen]

                else:
                    vec = - vec / np.sqrt(np.sum(vec ** 2)) * 1.09
                    if rotate == True: 
                        if i in atoms:
                            if np.any(unfolding.bonds_toBe == i) == True:
                                distance = np.linalg.norm(np.diff(unfolding.vertex_coords[unfolding.bonds_toBe[np.where(unfolding.bonds_toBe == i)[0]]],axis=1))
                                tmp = interpolated_angles(distance,halogen)
                                vec =  vec / np.sqrt(np.sum(vec ** 2)) * tmp[3]
                            #ax = unfolding.vertex_coords[axes[i][1]] - unfolding.vertex_coords[axes[i][0]]
                            ax = unfolding.vertex_coords[neighbours_on_face(i,unfolding.graph_unfolding_faces)[0]] - unfolding.vertex_coords[neighbours_on_face(i,unfolding.graph_unfolding_faces)[1]]
                            #print(i, neighbours_on_face(i,unfolding.graph_unfolding_faces), tmp[2])
                            mat = rotation_matrix(ax, tmp[2])
                            vec = np.dot(mat,vec.T).T
                    vec = unfolding.vertex_coords[i] + vec
                    text += " H "
                text += xyz_to_string(vec + np.random.normal(loc=0.,scale=1e-3,size=3))
        text += "\n"

        # and the connectivity table
        unique_graph = deepcopy(unfolding.graph_unfolding)
        if connectivity == True:
            for i in range(len(unique_graph)):
                text += " "
                text += str(i+1)
                if len(unique_graph[i]) > 0: 
                    for j in unique_graph[i]:
                        text += " "
                        text += str(j+1)
                        text += " 1.0"
                        unique_graph[j].remove(i)
                        if hydrogen_list[i] != None:
                            if type(hydrogen_list[i]) == list:
                                for hydrogen in hydrogen_list[i]:
                                    text += " "
                                    text += str(hydrogen)
                                    text += " 1.0"
                            else:
                                text += " "
                                text += str(hydrogen_list[i])
                                text += " 1.0"
                    text += "\n"
                else:
                    if hydrogen_list[i] != None:
                        if type(hydrogen_list[i]) == list:
                                for hydrogen in hydrogen_list[i]:
                                    text += " "
                                    text += str(hydrogen)
                                    text += " 1.0"
                                text += "\n"
                        else:
                            text += " "
                            text += str(hydrogen_list[i])
                            text += " 1.0"
                            text += "\n"
            else:
                text +="\n" 


            for i in range(len(hydrogen_list)):
                text += " "
                text +=  str(len(unfolding.graph_unfolding) + 1 + i)
                text += "\n"

        if writeFile == True:
            filename += ".com"
            with open(filename, 'w+') as outfile:
                outfile.write(text)
            outfile.close()
        else:
            return text

def write_gaussfile_restart(unfolding, header, double_hydrogens=np.array([],dtype=int), freeze=False, connectivity=False, writeFile=False, rotate=False, phis=[0.,0.], axes=None, atoms=None, filename="OutFile.com", freezelist=[], halogen=0, interpolated_angles = None):
    header_restart =  restart_header(header)
    filename_init = filename + "_initialise"
    write_gaussfile(unfolding, header, double_hydrogens, freeze, connectivity, writeFile, rotate, phis, axes, atoms, filename_init, freezelist, halogen, interpolated_angles=interpolated_angles)
    write_gaussfile(unfolding, header_restart, double_hydrogens, freeze, connectivity, writeFile, rotate, phis, axes, atoms, filename, freezelist, halogen, interpolated_angles=interpolated_angles)

def load_h5(filename):
    hf = h5py.File(filename, 'r')
    d_CC = np.array(hf.get("d_CC"))
    E_init = np.array(hf.get("E_init"))
    E_final = np.array(hf.get("E_final"))
    CCF = np.array(hf.get("CCF"))
    CCH = np.array(hf.get("CCH"))
    d_HCs = np.array(hf.get("d_HCs"))
    d_FCs = np.array(hf.get("d_FCs"))
    functional = np.array(hf.get("functional"))
    basis_set = np.array(hf.get("basis_set"))
    angle = np.array(hf.get("angle"))
    halogen = np.array(hf.get("halogen"))
    hf.close()
    return d_CC, E_init, E_final, CCF, CCH, d_HCs, d_FCs, basis_set, functional, halogen, angle

def plot_graph_with_periphery(unfolding, savefig=False, filename=None):
    dual_planar = unfolding.vertex_coords
    #midpoints = unfolding.midpoints
    fig, ax = plt.subplots()
    colors = ["k"]*unfolding.n_carbon + ["r"] * unfolding.n_halogen + ["b"] * unfolding.n_hydrogen
    ax.scatter(dual_planar[:,0],dual_planar[:,1],color=colors)
    for i in range(unfolding.n_carbon):
        for j in unfolding.graph_unfolding[i]:
            tmp = np.stack([unfolding.vertex_coords[i,:-1],unfolding.vertex_coords[j,:-1]]).T
            ax.plot(tmp[0],tmp[1],'b-',lw=0.5) 
    for i, txt in enumerate(np.arange(dual_planar.shape[0])):
        ax.annotate(txt,dual_planar[i][:2])
    #ax.scatter(midpoints[:,0],midpoints[:,1])
    ax.axis('equal');
    if savefig == True:
        fig.savefig(filename)
    return fig

def split_norm(X, axis=-1):
    R = np.sqrt(np.sum(X*X, axis = axis));
    D = X/R[...,NA]
    return R, D

def edge_displacements(X, neighbours):
    Xab = X[neighbours] - X[:len(neighbours),NA]       # Displacement vectors Xv-Xu (n x d x 3)
    return split_norm(Xab)

def edge_displacements_periphery(X, neighbours, n_carbon):
    Xab = X[neighbours] - X[n_carbon:,NA]
    return split_norm(Xab)

def set_colors(graph_faces):
    color = []
    colors = cm.rainbow(np.linspace(0, 1, len(graph_faces)))
    for i in range(len(graph_faces)):
        for j in range(len(graph_faces[i])):
            color.append(colors[i])
    color = np.array(color)
    return color
    
def init_face_right(graph, faces):
    ## check the combinations of three vertices and see if they belong to a pentagon or hexagon ##
    right_face = np.ones_like(graph, dtype=int) * (-1)
    vert_a, vert_b, vert_c = np.repeat(np.arange(len(graph))[...,NA],3,-1), graph, np.roll(graph,-1,axis=-1)
    corners = np.concatenate([vert_a[...,NA],vert_b[...,NA],vert_c[...,NA]], axis=-1)
    for face in faces:
        for j in range(graph.shape[0]):
            for i in range(graph.shape[1]):
                if ((corners[j][i][0] in face) and (corners[j][i][1] in face) and (corners[j][i][2] in face) ) == True:
                    right_face[j,i] = len(face) - 5
    return right_face

def repulsion_matrix(graph):
    ### indices of all the other atoms a node is not bond to or itself ###
    matrix = np.zeros([graph.shape[0],graph.shape[0] - 1 - graph.shape[1]], dtype=int)
    for i in range(graph.shape[0]):
        tmp = np.arange(graph.shape[0])
        tmp = np.delete(tmp,i)
        for j in graph[i]:
            tmp = np.delete(tmp, np.where(tmp == j)[0][0])
        matrix[i] = tmp
    return matrix

def repulsion_matrix_periphery(graph_periphery, n_carbon):
    periphery = graph_periphery.shape[0]
    periphery_matrix = np.zeros([periphery, periphery + n_carbon - 2], dtype=int)
    for i in range(n_carbon, n_carbon + periphery):
        tmp = np.arange(n_carbon + periphery)

        ### Delete the perihpery atom ###
        tmp = np.delete(tmp, i)

        ### And the periphery parent atom ###
        tmp = np.delete(tmp, np.where(tmp == graph_periphery[i - n_carbon][0])[0][0])
        periphery_matrix[i - n_carbon] = tmp
    return periphery_matrix

#
def replace_periphery_constants(graph_unfolding_array, N, periphery_type,spring_lengths,spring_constants,out_of_plane_k, periphery_bond_lengths):
    
    for u, neighbours_u in enumerate(graph_unfolding_array):
        for i, v in enumerate(neighbours_u):
            if v>= N:           # v is a periphery atom (hydrogen or halogen)
                # TODO: Once we want to make simulation more physical, periphery bonds should be physically appropriate instead of same as C-C.
                spring_lengths[u][i] = periphery_bond_lengths[periphery_type[v-N]]                
#                spring_constants[u][i] = 0 # NB: Switches off periphery for debugging
                print(f"Setting optimal bond length of {u}--{v} to {spring_lengths[u][i]}")                
#                out_of_plane_k[v] = 0
                

# # Replace cut bonds with bonds to periphery atoms: the hydrogen and halogens
# def remove_bonds(graph, bonds_removed, bond_k, angle_k, out_of_plane_k, spring_lengths, halogen_positions):

#     ### For each bond that is removed, four angle force constants have to be set to zero ###
#     """
#     a     b
#      \   /
#       (u)
#        |
#       (v)
#      /   \
#     d     c
#     """
#     ### When removing bond u-v, the two angles <(vub) and <(aub) as well as the two angles <(ijc) and <(cjd) have to be set to zero ###
#     ### Those can be easily found by looking through the points around a node, which form an angle ###
#     ### These are always given by the point (vert_a), its neighbour (vert_b) and the rotated neighbours (vert_c) and are thereby clearly defined ###
#     vert_a, vert_b, vert_c = np.repeat(np.arange(len(graph))[...,NA],3,-1), graph, np.roll(graph,1,axis=-1)
#     corners = np.concatenate([vert_a[...,NA],vert_b[...,NA],vert_c[...,NA]], axis=-1)

#     ### Now we can look through all the corners (defined by the three vertices) around all points and if two of ###
#     ### those vertices are in the removed bonds list, we set the correspodning angular k to zero  ###
    
#     for u,v in bonds_removed:
#         print(f'Removing bond: {pair}')
#         if u in halogen_positions:
#             spring_lengths[u][np.where(graph[u] == v)[0]] = 1.35#,1.76]
#         else:
#             spring_lengths[u][np.where(graph[u] == v)[0]] = 1.09

#         if v in halogen_positions:
#             spring_lengths[v][np.where(graph[v] == u)[0]] = 1.35#,1.76]
#         else:
#             spring_lengths[v][np.where(graph[v] == u)[0]] = 1.09

#         ### For a pair of removed bonds, set the force constant that correspond to that bond to zero ###
#         bond_k[u][np.where(graph[u] == v)[0]] = 0.
#         bond_k[v][np.where(graph[v] == u)[0]] = 0.

#         ### Set the out of plane constants for all three vertices around the nodes to zero ### 
#         out_of_plane_k[u] *= 0.
#         out_of_plane_k[v] *= 0.

#         ### Go through all nodes ###
#         for j in range(graph.shape[0]):
#             ### and all neighbours ###
#             for i in range(graph.shape[1]):
#                 ### if the two vertices, whichs bond is removed is in the corner (part of an angle) its force constant is set to 0 ###
#                 if (u in corners[j,i]) and (v in corners[j,i]) == True:
#                     angle_k[j,i] = 0.

def add_periphery(affected_children, parent_atom, n_carbon):
    ### This functions takes the affected children list and adds all the periphery atoms to the list in which the parent atoms appear ###
    affected_children_periphery = deepcopy(affected_children)
    for i in range(len(parent_atom)):
        for j in range(len(affected_children)):
            if parent_atom[i] in affected_children[j]:
                affected_children_periphery[j].append(n_carbon + i)
    return affected_children_periphery

def face_type(faces):
    # For a list of vertices forming a face
    # return two arrays, of which face is a 
    # pentagon and which is a hexagon 
    hex_id = []
    pent_id = []
    for i in range(len(faces)):
        if len(faces[i]) == 6:
            hex_id.append(i)
        elif len(faces[i]) == 5:
            pent_id.append(i)
    hex_id = np.array(hex_id)
    pent_id = np.array(pent_id)
    return pent_id, hex_id


# Map translating dual arc to the cubic node corresponding to the triangle it's part of
def make_arc2cubic(isomer):
    arc2cubic = {}
    triangles = isomer['triangles']
    # Assign cubic nodes to dual triangles according to order triangles in isomer.triangles
    for i, triangle in enumerate(triangles):
        for j in range(len(triangle)):
            arc2cubic[triangle[j], triangle[(j+1)%len(triangle)]] = i

    return arc2cubic

# Map translating dual arc to cubic arc
def make_darc2carc(isomer,arc2cubic):
    darc2carc = {}
    dg = isomer['dual_neighbours']

    for u, nu in enumerate(dg):
        for v in nu:
            a = arc2cubic[u,v]
            b = arc2cubic[v,u]
            darc2carc[u,v] = (a,b)

    return darc2carc

def unfolding_dual_graph(isomer,arcpos):
    def unique_source(A):
        return len(set([tuple(a[0]) for a in A]))==1

    def remove_node(g,u):
        nu = g[u]
        for v in nu:
            g[v].remove(u)
        g[u] = []
    
    uf_dg = deepcopy(isomer['dual_neighbours'])
    degrees = np.array([len(row) for row in uf_dg])    
    Nf    = len(uf_dg)
    
    for u in range(Nf):
        if not unique_source(arcpos[u][:degrees[u]]):
            remove_node(uf_dg,u)
    
    return uf_dg


def unfolding_bonds(uf_dg, isomer, arcpos):
    arc2cubic = make_arc2cubic(isomer)

    included_bonds = set()
    dg = isomer['dual_neighbours']
    
    for u, uf_f in enumerate(uf_dg):
        if(uf_f != []):            # f is fully included in unfolding
            f = dg[u]
            d = len(f)
            for i in range(d):
                a,b = arc2cubic[u,f[i]], arc2cubic[u,f[(i+1)%d]]
                included_bonds.add( (min(a,b), max(a,b)) )

    uf_g  = isomer['cubic_neighbours'] # Full cubic graph
    ufi_g = deepcopy(uf_g).tolist()    # Bonds included in unfolding
    ufb_g = deepcopy(uf_g).tolist()    # Bonds broken in unfolding

    for a, neighbours_a in enumerate(uf_g):
        for b in neighbours_a:
            if (min(a,b),max(a,b)) in included_bonds:
                ufb_g[a].remove(b)
            else:
                ufi_g[a].remove(b)                

    return ufi_g, ufb_g

def make_faces(dg, arc2cubic):       
    faces = [];

    for u, nu in enumerate(dg):
        f = []
        for v in nu:
            a = arc2cubic[u,v]
            f += [a]
#        print(f"face {u} has dual neighbours {nu}. These become the atoms: {f}")            
        faces += [f]
    return faces

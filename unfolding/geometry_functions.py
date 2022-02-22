'''
This module contains all functions that are related to the 3D geometry
'''

import numpy as np
from numpy.lib.shape_base import get_array_wrap
NA = np.newaxis 

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    INPUT:
    axis: 3d array of the exis around which the rotation is applied
    theta: angle in radians of the applied rotation

    OUTPUT:
    3 by 3 np.array with the rotation matrix 
    """
    # get the axis as a 64 bit float for higher precision
    axis = np.asarray(axis,dtype=np.float64)
    axis = axis / np.sqrt(np.dot(axis, axis))

    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]],dtype=np.float64)

def rot_2d(theta):
    '''
    Returns the rotation matrix for a rotation around the z-axis 

    INPUT:
    theta: angle of rotation in radians

    OUTPUT: 
    2 by 2 np.array with the rotation matrix 
    '''
    return np.array([[np.cos(theta), np.sin(theta)],[- np.sin(theta), np.cos(theta)]])

def rotate_vector(vec,phi):
    M_rot = rot_2d(phi)
    return np.matmul(M_rot,vec)


def angle_vec(a,b,degrees = True):
    c = np.sum(a * b, axis = -1)
    tmp = c / (np.sqrt(np.sum(a**2, axis = -1)) * np.sqrt(np.sum(b**2, axis = -1)))
    if degrees == True:
        return np.degrees(np.arccos(tmp))
    else:
        return np.arccos(tmp)

def Eisenstein_to_Carth(x_eis,unit_dist = 1):
    '''
    Return the Eisenstein coordinates of two dimensional input vertices
    INPUT:

    OUTPUT:

    '''
    omega = np.exp(1j * 2. * np.pi / 6)
    tmp = x_eis[:,0] + x_eis[:,1] * omega
    x_cart = np.concatenate([np.array([np.real(tmp),np.imag(tmp)]).T, np.zeros_like(x_eis[:,0])[:,None]], axis=1)
    return (x_cart * unit_dist)

def vote_on_normal(normals):
    '''
    Given a set of normal vectors find the mean direction and multiply each normal vector with the sign to make them point in the same direction
    INPUT:
    normals: a set of normal vectors as np.array 

    OUTPUT:
    a set of normal vectors pointing in the direction of the majority of the vectors
    '''

    # find a non-zero normal vector
    non_zero = np.argmax((normals**2).sum(axis=-1), axis=-1)
    #print(f"Non zero normal vector is {normals[non_zero]} and has index: {non_zero}\n")

    #print(normals[np.arange(len(normals)),non_zero][:,np.newaxis,:])

    # pick the corresponding vectors to the 
    # indices and repeat them for each vertex 
    # in the face
    non_zero_normals = np.repeat(normals[np.arange(len(normals)),non_zero][:,np.newaxis,:], normals.shape[1], axis=1)
        
    # identify which of the normal vectors 
    # point in the same direction as the 
    # reference normal by looking at the 
    # sign of the dot product
    signs = np.sign(np.sum(non_zero_normals * normals, axis=-1))
    signs[np.where(signs == 0)] = 1

    # determine the average sign to see 
    # whether the reference normal vector 
    # points in the wrong direction
    change_sign = np.sign((signs.sum(axis=-1)))
    change_sign[np.where(change_sign == 0)] = 1

    # return an array of +/- 1 that each 
    # normal vector has to be multiplied with 
    # in order for all of them to point in 
    # the same direction
    return change_sign[...,NA] * signs

def mean_normal(vertices, faces):
    '''
    # Return the mean normal vector for each face, given the vertices and the face array either hexagons or pentagons
    INPUT:

    OUTPUT:

    '''

    vec_left = vertices[faces] - vertices[np.roll(faces, -1, axis = -1)]
    vec_right = vertices[np.roll(faces, 1, axis = -1)] - vertices[faces]

    normals = np.cross(vec_left, vec_right)
    normals = normals / np.linalg.norm(normals, axis = -1)[...,NA]
    #print(f"Shape of normal vectors is {normals.shape}")

    mean_normal = 1/faces.shape[-1] * np.sum(normals * vote_on_normal(normals)[...,NA], axis = -2) 

    return mean_normal

def detect_minimal_distance(geometry,min_length,title):
    n_atoms = geometry.shape[0]
    distances = np.zeros([n_atoms,n_atoms])
    for atom_a in range(n_atoms):
        distances[atom_a,atom_a] = 1e10
        for atom_b in range(atom_a):
            length = np.sqrt(np.sum((geometry[atom_a] - geometry[atom_b])**2))
            distances[atom_a,atom_b] = length
            distances[atom_b,atom_a] = length
    min_atom_a, min_atom_b = np.where(distances < min_length)
    min_atom_a, min_atom_b = min_atom_a[:len(min_atom_a)//2], min_atom_b[:len(min_atom_b)//2]
    txt=''
    if len(min_atom_a) !=0:
        for min_dist_ID in range(len(min_atom_a)):
            txt+=(f"Minimal distance in file {title}: r={distances[min_atom_a[min_dist_ID],min_atom_b[min_dist_ID]]} encountered between Atoms {min_atom_a[min_dist_ID]} and {min_atom_b[min_dist_ID]}!\n")
            print(f"Minimal distance in file {title} r={distances[min_atom_a[min_dist_ID],min_atom_b[min_dist_ID]]} encountered between Atoms {min_atom_a[min_dist_ID]} and {min_atom_b[min_dist_ID]}!\n")
    return txt

def calculate_final_angles(closed_vertices, graph_unfolding_faces, hinges):
    final_midpoints = np.zeros([len(graph_unfolding_faces),3])

    for faceId, face in enumerate(graph_unfolding_faces):
        final_midpoints[faceId] = np.mean(closed_vertices[face],axis=-2)

    hinge_midpoints = np.sum(closed_vertices[[tuple(hinges[1])]],axis=-2) / 2.

    # calculate the vectors from the hinge midpoint to the face midpoint of the two faces which form the hinge
    hinge_legs = final_midpoints[[tuple(hinges[0])]] - np.repeat(hinge_midpoints[:,np.newaxis,:],2,axis=-2)

    # calculate the angles between those two leg-vectors
    angles = angle_vec(hinge_legs[:,0,:],hinge_legs[:,1,:],degrees=False)
    return angles
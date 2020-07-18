import numpy as np
from functions_folding import split_norm
NA = np.newaxis

def harmonic_FC(atom_1, atom_2):
    ### H, F, Cl, C = 0, 1, 2, 3 ###
    ### for 0,0 return H-H FC for 0,3 H-C FC ###
    parameters =  np.array([[0.354, 4.5280 ,0.712], [0.668, 10.874 ,1.735], [1.044, 8.564, 2.348],  [0.732, 5.343,1.912]])
    A, B = parameters[atom_1], parameters[atom_2]
    r_a, x_a, z_a, r_b, x_b, z_b = A[:,0], A[:,1], A[:,2], B[:,0], B[:,1], B[:,2] 
    r_EN = r_a*r_b * (np.sqrt(x_a) - np.sqrt(x_b))**2 / (x_a*r_a + x_b*r_b)
    r_ab = r_a + r_b + r_EN
    return 664.12 * (z_a * z_b) / r_ab

def harm_pot(R, length, k):
    Ui = 0.5 * k * (R - length)**2
    return Ui

def grad_harm_pot(D, R, length, k):
    dE = - (R - length)[...,NA] * D * k[...,NA]
    return dE

def angle_spring_pot(vertices, k, graph, angles):
    left = vertices[:,np.newaxis,:] - vertices[graph]
    right =  vertices[:,np.newaxis,:] - vertices[np.roll(graph,-1,axis=1)]
    phi = np.arccos(np.sum(left * right,axis=-1) / (np.sqrt(np.sum(left**2,axis=-1)) * np.sqrt(np.sum(right**2,axis=-1))))
    Ui = 0.5 * k * (phi - angles)**2
    U = np.sum(Ui)
    return U

def morse(R, r_e, De, k):
        a = np.sqrt(k / (2 * De))
        return De * (1 - np.exp(- a * np.abs(R - r_e)))**2

def morse_grad(R, D, r_e, De, k):
        a = np.sqrt(k / (2 * De))   
        return -2 * De[...,NA] * (1 - np.exp(- a * np.abs(R - r_e)))[...,NA] * (np.exp(- a * np.abs(R - r_e)) * a)[...,NA] * D

def grad_angle_spring_pot(D, R, angles, k):
    D_left, R_left = np.roll(D,-1,axis=-2), np.roll(R, -1, axis=-1)
    arccos_dr = - 1 / np.sqrt(1 - np.sum(D * D_left, axis=-1)**2) 
    
    #nominator = R * np.roll(R, -1, axis=-1) * (-1 ) *( D + np.roll(D,-1,axis=-2)) - np.sum(R[...,NA] * D, axis=-1) * (- np.roll(R, -1, axis=-1) * D - R * np.roll(D,-1,axis=-2))
    nominator = (R * R_left * (-1 ))[...,NA] * \
                (R[...,NA] * D + R_left[...,NA] * D_left) -\
                (R[...,NA] * D * R_left[...,NA] * D_left) *\
                (- R_left[...,NA] * D - R[...,NA] * D_left)

    denominator =  ( R * R_left )[...,NA] ** 2

    return  k[...,NA] * np.arccos((np.sum(D * D_left, axis=-1)))[...,NA] * arccos_dr[...,NA] * nominator / denominator

def grad_cos_angle(D_right, R_right, angles, k):
    D_left, R_left = np.roll(D_right, 1,axis=-2), np.roll(R_right, 1, axis=-1)
    cos_phi = np.sum(D_right * D_left, axis=-1)

    ### Functional derivatives of the outer function ###
    f1 = k * (cos_phi - np.cos(angles))

    ### how a change in the left/right node affectes the angle ###
    contrib_from_left = f1[...,NA] * 1 / R_left[...,NA] * (D_right - D_left * cos_phi[...,NA])
    contrib_from_right = f1[...,NA] * 1 / R_right[...,NA] * (D_left - D_right * cos_phi[...,NA])

    contrib_from_center = - (contrib_from_left + contrib_from_right)
    return contrib_from_center, contrib_from_right, contrib_from_left

def out_of_plane_gradient(D, R, k):
    ### For nomenclature of the vectors see Thesis Chapter: "Potentials and Gradients" ###
    ### Define the local coordinate system for the tetrahedron with point j at the top ###

    angles = np.ones_like(k) * np.pi

    D_left, R_left = np.roll(D, 1, axis = -2), np.roll(R, 1, axis = -1)
    D_right, R_right = np.roll(D, -1, axis = -2), np.roll(R, -1, axis = -1)

    ### Unit vector perpendicular to nj and mj ###
    u = np.cross( - D_right, D_left, axis = -1)
    u = u / np.linalg.norm(u,axis=-1)[...,NA]
    
    ### Projection of jk on the njm plane as the second axis###
    w = D - u * np.sum(D * u, axis =-1)[...,NA]
    w = w / np.linalg.norm(w, axis=-1)[...,NA]
    
    ### Complete the local coordinate system with a vector orthogonal to both u and w ###
    v = np.cross(w, u, axis = -1)
    v = v / np.linalg.norm(v, axis=-1)[...,NA]
    
    ### In order to place the phantom atom i we need to define the value f  which determines the proportion along nm at which i lies###
    f = np.sum(- D_right * v, axis = -1) / ( np.sum(- D_right * v, axis = -1) + np.sum( D_left * v, axis = -1) )
    
    rji = D_left + (1 - f)[...,NA] * (D_right - D_left)
    
    R_ji, D_ji = split_norm(rji)
    cos_gamma = np.sum(D_ji * D, axis=-1)
    
    f1 = k * (cos_gamma - np.cos(angles))
    
    ### Treat as a regular angular spring ###
    ### This need to be added to the gradient of vertices[graph] = right neighbour ###

    contrib_from_k = f1[...,NA] * 1 / R[...,NA] * (D_ji - D * cos_gamma[...,NA])
    
    ### This is the contribution from the phantom point i ###
    contrib_from_i = f1[...,NA] * 1 / R_ji[...,NA] * (D - D_ji * cos_gamma[...,NA])
    
    ### Contribution from the central node is given as -(sum of the neighbours) ###
    contrib_from_j = - (contrib_from_k + contrib_from_i)
    
    ### In order to get the derivative as a function of the points m and n we have to apply the chain rule ###
    contrib_from_n = contrib_from_i * (1 - f)[...,NA]
    contrib_from_m = contrib_from_i * f[...,NA]
    
    return contrib_from_j, contrib_from_k, contrib_from_m, contrib_from_n
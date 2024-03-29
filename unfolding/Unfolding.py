import sys
import numpy as np
#import pyqtgraph.opengl as gl
from matplotlib import cm




sys.path.append('../functions')
from functions_folding import *
from potentials import *
from geometry_functions import *


class Unfolding(object):
    def __init__(self, dual_unfolding, graph_unfolding_faces, graph_faces, graph_unfolding, graph, halogen_positions, root_node, bonds_toBe, angles_f, position_final):

        self.dual_unfolding         = dual_unfolding
        self.graph_unfolding        = graph_unfolding
        self.graph                  = graph
        self.root_node              = root_node  ### Index of the root face
        self.n_carbon               = len(graph_unfolding)
        
        ### Define the bonds which have to be formed and are therfore removed from the graph ###

        if not isinstance(bonds_toBe, np.ndarray):
            n_removed_bonds     = sum([bonds != [] for bonds in bonds_toBe]) // 2 
            bonds               = np.empty([n_removed_bonds,2],dtype=int)
            bond_id             = 0
            for node_id, v in enumerate(bonds_toBe):
                if v != []:
                    if node_id not in bonds:
                        bonds[bond_id][0]   = node_id
                        bonds[bond_id][1]   = v[0]
                        bond_id             += 1
            self.bonds_toBe = bonds                         
        else:
            self.bonds_toBe = bonds_toBe

        if halogen_positions is None:
            halogen_positions = np.zeros(self.n_carbon,dtype=int)
            #for i in self.bonds_toBe.flatten():
                #halogen_positions[int(i)]    = 1

        
        self.n_halogen              = len(np.where(halogen_positions != 0)[0]) 
        self.graph_unfolding_array,\
            self.n_periphery,\
            self.hydrogen_positions,\
            self.graph_periphery,\
            self.periphery_type,\
            self.parent_atom        = make_graph_array(self.graph_unfolding, self.graph, halogen_positions, neighbours=3)
        self.n_hydrogen             = self.n_periphery - self.n_halogen
        self.graph_unfolding_faces  = graph_unfolding_faces
        self.graph_faces            = graph_faces
        self.vertices_final         = None
        if halogen_positions is not None:
            self.halogen_positions      = halogen_positions
        self.halogen_parent_atom    = np.where(self.halogen_positions!= 0)[0]
        self.hydrogen_parent_atom   = np.where(self.hydrogen_positions!= 0)[0]

        print(f'Halogene position : {self.halogen_positions}')
        print(f'length of halogene position: {len(self.halogen_positions)}')
        np.random.seed(12)  ### Define a random seed

        ##### CONSTANTS ####
        self.periphery_atomic_masses = np.array([1.,18.99, 35.453]) # H,F,Cl

        ### Define the mean vacuum bond parameters ###
        self.bond_angles = np.radians(np.array([108.,120.]))
        self.bond_lengths = np.array([1.458,1.401]) # hp,hh TODO: Include pp
        self.force_constants = {'bond':  np.array([390.,450.,260.]) * 6.022e8, # hp,hh,pp; (6.022e8) # given on Lukas Wirtz p. 126
                                'angle': np.array([100.,100.]) * 6.022e8    # h, p
                                };
        self.periphery_bond_lengths = np.array([1.09,1.35,1.76]) # H,F,Cl
        
        ### Dissociation energy and bond distacne matrix in the order H-F-Cl-C
        self.D_E = 4.184 * np.array([[436.002, 568.6, 431.8, 310.], [568.6, 156.9, 250.54, 485.] , [431.8, 250.54, 242.580, 351.], [310., 485., 351., 607.]])
        self.r_e = np.array([[0.74, 0.92, 1.27, 1.09], [0.92, 1.42, 1.648, 1.35] , [1.37, 1.648, 1.99 ,1.77], [1.09, 1.35, 1.77, 1.44]])

        ### initialise the faces around one cubic node ###
        # TODO: This seems to be face_right from data
        self.right_face = init_face_right(self.graph, self.graph_faces)

        ### Intitialise the optimal angles ###
        self.angle = np.ones([self.n_carbon,3],dtype=np.float64)


        ### Initialise the displacement unit vectors that get updated each step and their lengths ###
        # self.D_carbon = np.zeros([self.n_carbon, 3, 3]) # displacement unit vectors
        # self.R_carbon = np.zeros([self.n_carbon, 3])    # displacement lengths
        
        # self.D_unfolding = np.zeros([self.n_carbon, 3])
        # self.R_unfolding = np.zeros([self.n_carbon])
        
        #self.D_hydrogen = np.zeros([self.n_hydrogen, 3, 3])
        #self.R_hydrogen = np.zeros([self.n_hydrogen, 3])

        ### Initialise the positions ###
        self.vertex_coords = np.zeros([self.n_carbon + self.n_periphery,3])

        ### Define the masses fo the integration ###
        self.m = np.ones([len(self.vertex_coords)]) * 12.011
        ### Set the masses of the halogens and hydrogens ###
        self.m[self.n_carbon:] = self.periphery_atomic_masses[self.periphery_type]

        self.v = np.zeros([len(self.m),3]) # velocities
        self.a = np.zeros([len(self.m),3]) # accelerations
        self.dt = 1e-8  ### set the timestep for the integration

        ### Initiate the views for halogens and hydrogens ###
        ### They should never be redefined but only ### 
        ### treated with additions and substractions ###

        #self.halogens = self.vertex_coords[self.halogen_parent_atom]# + self.n_carbon]
        #self.hydrogens = self.vertex_coords[self.hydrogen_parent_atom]# + self.n_carbon]
        self.periphery_vertices = self.vertex_coords[self.n_carbon:]

        # TODO: Also in the input data
        self.pentagons, self.hexagons = hex_and_pents(graph_unfolding_faces)
        self.pentagon_normals = np.zeros([self.pentagons.shape[0],3])
        self.hexagon_normals = np.zeros([self.hexagons.shape[0],3])
        self.face_type = face_type(self.graph_unfolding_faces) # returns (pentagon_ids, hexagon_ids): dual node/face id's for all faces gathered by face degree

        ### Define the indices needed for the coulomb repulsion ###
        # NB: We wait with Coulomb forces until later; Turning them off for now
        #self.inverse_graph = repulsion_matrix(self.graph) 
        #self.inverse_graph_periphery = repulsion_matrix_periphery(self.graph_periphery, self.n_carbon)

        ### Draw initial vertices from the graph ###
        self.vertex_init =  draw_vertices_unfolding(self.dual_unfolding, self.graph_unfolding_faces, self.root_node, self.bond_angles, self.bond_lengths)         # Initial positions of every atom
        self.vertex_coords[:self.n_carbon] = self.vertex_init                                                                                                        # Current coordinates of carbon skeleton

        ### Give the halogens and hydrogens the coordinates of their parent atom ###
        #self.halogens += self.vertex_coords[self.halogen_parent_atom]
        #self.hydrogens += self.vertex_coords[self.hydrogen_parent_atom]
        self.periphery_vertices += self.vertex_coords[self.parent_atom]

        ### Move the halogens and hydrogens by taking the average of the connecting vectors, their C father atom is bond to and flipping it by substracting it from their father atom ###
        #tmp_1 = np.sum(self.vertex_coords[self.graph_unfolding_array[self.halogen_parent_atom]] - self.halogens[:,np.newaxis], axis=-2) / 2
        #tmp_2 = np.sum(self.vertex_coords[self.graph_unfolding_array[self.hydrogen_parent_atom]] - self.hydrogens[:,np.newaxis], axis=-2) / 2
        tmp_3 = np.sum(self.vertex_coords[self.graph_unfolding_array[self.parent_atom]] - self.periphery_vertices[:,np.newaxis], axis=-2) / 2

        #tmp_1 = tmp_1 / np.sqrt(np.sum(tmp_1 ** 2, axis=-1))[:,np.newaxis]
        #tmp_2 = tmp_2 / np.sqrt(np.sum(tmp_2 ** 2, axis=-1))[:,np.newaxis]
        tmp_3 = tmp_3 / np.sqrt(np.sum(tmp_3 ** 2, axis=-1))[:,np.newaxis]

        #self.halogens -= self.periphery_bond_lengths[self.halogen_positions[np.where(self.halogen_positions != 0)[0]]][:,np.newaxis] * tmp_1
        #self.hydrogens -= 1.09 * tmp_2
        self.periphery_vertices -= self.periphery_bond_lengths[self.periphery_type][...,NA] * tmp_3

        ### Define the tree and the hinges for the Minimal Spanning Tree ###
        self.tree, self.affected_children, self.hinges, self.hinges_conected = hinges_traversed(dual_unfolding, graph_unfolding_faces, self.root_node)
        self.face_normals = np.zeros([len(self.graph_unfolding_faces), 3])

        print(f"MST = {self.tree}, root node: {self.root_node}")
        
        ### Initialise the hinge angles as pi = Flat precursor molecule ###
        self.hinge_angles = np.ones(len(self.hinges[0])) * np.pi
        #self.update_hinge_angles()

        ### Define the hinges, which have not yet reached their optimal value (all) in a list so they are ordered by ther depth from the root node ###
        #self.open_hinges = [[0,1,2],[3,4,5,6,7,8],[9,10,11],[12,13,14]] # Hardcoded values for Scott precursor
        #self.all_hinges  = [[0,1,2],[3,4,5,6,7,8],[9,10,11],[12,13,14]] # Hardcoded values for Scott precursor
        self.open_hinges = hinge_order(self.tree, self.root_node, self.hinges[0])

        print(f"open_hinges = {self.open_hinges}")
        

         ### Final angles for the hinges ###
        if angles_f is not None:
            self.angles_f = angles_f
            
        elif position_final is not None:
            Nf = len(self.graph)
            pent_id, hex_id = face_type(self.graph_faces)
            closed_normals = np.zeros((Nf,3),dtype=float)

            hexagons, pentagons = [], []
            for face in self.graph_faces:
                if len(face) == 5:
                    pentagons.append(self.graph_faces)
                elif len(face) == 6:
                    hexagons.append(self.graph_faces)
            hexagons, pentagons = np.array(hexagons), np.array(pentagons)
            print(f'Pentagon Ids: {pent_id}')
            # TODO: Look at this when we start closing
            #closed_normals[pent_id] = mean_normal(position_final, pentagons)
            #closed_normals[hex_id] = mean_normal(position_final, hexagons)
            #angles_final = [angle_vec(closed_normals[u],closed_normals[v],degrees=False) for u,v in self.hinges[0]]

            #angles_final = calculate_final_angles(position_final, self.dual_unfolding, self.hinges)

        ### Define the steps to close the hinges ###
        #self.num_of_steps = 20000
        #self.angle_steps = np.linspace(np.pi, self.angles_f,self.num_of_steps + 1)
        #self.step_size =  (np.pi - self.angles_f) / self.num_of_steps

        self.stage = 0  ### Set the stage to zero # TODO: This should probably be controlled from the outside?


        ### Intialise the mesh for the GL application ###
        #self.midpoints = np.zeros([len(self.graph_faces),3],dtype=np.float64)
        #self.vertex_mesh = np.zeros([self.n_carbon + len(self.graph_faces), 3],dtype=np.float64)
        #self.color = set_colors(self.graph_faces)
        #self.faces = triangulate_polygone(self.graph_faces,self.n_carbon)

        self.midpoints          = np.zeros([len(self.graph_unfolding_faces),3],dtype=np.float64)
        self.vertex_mesh        = np.zeros([self.n_carbon + len(self.graph_unfolding_faces), 3],dtype=np.float64)
        self.color              = set_colors(self.graph_unfolding_faces)
        subfaces                = []
        for face in self.graph_unfolding_faces:
            if len(face) > 0:   
               subfaces.append(face) 
        self.faces              = triangulate_polygone(subfaces,self.n_carbon)
        
        self.stepsize = 1e-3

        #self.vertex_coords[:self.n_carbon] += np.random.normal(0,1,[self.n_carbon,3])

        self.update_displacements()
        self.init_springs()
        self.update_mesh()
        self.angle_constants = self.force_constants['angle'][self.right_face]

        ### Define the out of plane bending constants ###
        self.out_of_plane_constants = np.ones_like(self.graph, dtype=np.float64)  * 1e11
        print(f'Bonds to be: {self.bonds_toBe}')
        print(f'Halogene parent atom: {self.halogen_parent_atom}')
        replace_periphery_constants(self.graph_unfolding_array,self.n_carbon,self.periphery_type,self.spring_lengths, self.spring_constants,self.out_of_plane_constants,self.periphery_bond_lengths)
#        remove_bonds(self.graph, self.bonds_toBe, self.spring_constants, self.angle_constants, self.out_of_plane_constants, self.spring_lengths, self.halogen_parent_atom)

        ### define the index array to add the force to ###
        self.ix = np.repeat(self.graph_unfolding_array[...,NA],3,-1)
        self.iy = np.repeat(np.arange(self.graph.shape[1])[NA,NA,...],self.n_carbon,0)
        self.iz = np.repeat(np.array([[[0,1,2]]]),self.n_carbon,0)
        self.scale=1e-1
#        self.coulomb = repulsion_constant = 2.10935 * 1e5 * 36.# NB: Coulomb forces switched off for now

        ###
        ### Set the outmost three hinges to a negative step so they get opened during the process 
        ### this is only to move the carbon atoms closer together for the additional PCCP calculations
        #self.hinges[1][-3:] = np.flip(self.hinges[1][-3:],axis=1)
        ###
        ###

    def init_springs(self):
        ### Set all the sprig information to Hexagon-hexagon values ###
        ### Intitialise the spring constants and the optimal lenths ###
        # self.spring_lengths[pent_mask] = self.force_constants['bond'][0]        
        # self.spring_lengths[hex_mask]  = self.force_constants['bond'][1]

        self.spring_lengths = np.ones([self.n_carbon,3],dtype=np.float64)   * self.bond_lengths[1]
        self.spring_constants = np.ones([self.n_carbon,3],dtype=np.float64) * self.force_constants['bond'][1]
        
        # change all spring lengths and constants of the springs next to a pentagon
        for pentagon in self.graph_faces:
            if (len(pentagon) == 5):
                for vertex in range(5):
                    neighbours = self.graph[pentagon[vertex]]
                    for j in range(3):
                        if (neighbours[j] in pentagon) == True:
                            ### If the neighbour of a vertex, which is part of a pentagon is also part of the same pentagon ###
                            ### change the spring constant and length to the pentagon values ###
                            self.spring_constants[pentagon[vertex],j] = self.force_constants['bond'][0]
                            self.spring_lengths[pentagon[vertex],j]   = self.bond_lengths[0]

    def update_displacements(self):
        self.R_carbon,    self.D_carbon       = edge_displacements(self.vertex_coords, self.graph)
        self.R_unfolding, self.D_unfolding = edge_displacements(self.vertex_coords, self.graph_unfolding_array)
        #self.R_hydrogen, self.D_hydrogen = split_norm(self.vertex_coords[self.hydrogen_parent_atom] - self.hydrogens)

    def coulomb_force(self):
        ### Coulomb force of the carbons ### 
        R, D = edge_displacements(self.vertex_coords, self.inverse_graph)
        force_carbons = ( 1 / R**2 )[...,NA] * D * self.coulomb
        ### Coulomb force of the periphery ### 
        R_periphery, D_periphery = edge_displacements_periphery(self.vertex_coords, self.inverse_graph_periphery, self.n_carbon)
        force_periphery =  (1 / R_periphery**2 )[...,NA] * D_periphery * self.coulomb
        ### Combined coulomb force ###
        #print(force_carbons.shape)
        #print(force_carbons.shape)
        total = np.concatenate([np.sum(force_carbons, axis=-2), np.sum(force_periphery, axis=-2)])
        return total

    def update_mesh(self):
        for face in range(len(self.graph_unfolding_faces)):
            #midpoint = self.vertex_coords[self.graph_faces[face]].mean(axis=0)
            midpoint = self.vertex_coords[self.graph_unfolding_faces[face]].mean(axis=0)
            self.midpoints[face] = midpoint
        
        self.vertex_mesh[:self.n_carbon] = self.vertex_coords[:self.n_carbon]
        self.vertex_mesh[self.n_carbon:] = self.midpoints

    def update_force_bond(self):
        grad_pot = grad_harm_pot(self.D_unfolding, self.R_unfolding, self.spring_lengths, self.spring_constants)
        
        ### Calulate the gradient for all the carbons and take the negative gradient from the parent atoms for the periphery ###

        #ida = np.repeat(self.graph_periphery[...,NA], 3, -1)[:,0,:]
        #idb = np.repeat(self.graph_periphery[...,NA], 3, -1)[:,1,:]
        #grad_periphery = - grad_pot[ida, idb][:,0,:]


        ### Calculate the periphery to parent atom Morse potential ###
        # TODO: Work out how to do this properly (if we want more accurate FF for periphery)
        # R_per, D_per =  split_norm(self.periphery_vertices - self.vertex_coords[self.parent_atom])
        # k = harmonic_FC(3* np.ones_like(self.parent_atom),self.periphery_type)
        # D_E = self.D_E[3* np.ones_like(self.parent_atom), self.periphery_type]
        # r = self.r_e[3* np.ones_like(self.parent_atom), self.periphery_type]

        # grad_periphery = - morse_grad(R_per, D_per, r, D_E, k) 

        periphery_mask =self.graph_unfolding_array >= self.n_carbon
        periphery_ix   =self.graph_unfolding_array[periphery_mask] - self.n_carbon

        grad_periphery = -grad_pot[periphery_mask]

        return grad_pot, grad_periphery #np.concatenate([np.sum(grad_pot, -2), grad_periphery])

    def update_force_angle(self):
        ### Calculate the gradient in regard to the three angles around a node ###
        center, right, left = grad_cos_angle(self.D_unfolding, self.R_unfolding, self.bond_angles[self.right_face], self.angle_constants)
        grad = np.concatenate([center, np.zeros([self.n_periphery,3,3])])
        ### add the gradient in regard to the neighbour on the right to the neighbout on the right ###
        #center[self.ix, self.iy, self.iz] += right

        ### Try if it works with the periphery ###
        grad[self.ix, self.iy, self.iz] += right
        ### add the gradient in regard to the neighbour on the left to the neighbout on the left ###
        #center[np.roll(self.ix, 1, axis=1), self.iy, self.iz] += left

        ### Try if it works with the periphery ###
        grad[np.roll(self.ix, 1, axis=1), self.iy, self.iz] += left
        return grad #center
        
    def update_out_of_plane(self):
        contrib_j, contrib_k, contrib_m, contrib_n = out_of_plane_gradient(self.D_unfolding, self.R_unfolding, self.out_of_plane_constants)
        grad = np.concatenate([contrib_j, np.zeros([self.vertex_coords.shape[0] - self.n_carbon,3,3])])
        
        ### Add to the neighbour ###
        #contrib_j[self.ix, self.iy, self.iz] += contrib_k

        ### Add to the left ###
        #contrib_j[np.roll(self.ix, 1, axis=1), self.iy, self.iz] += contrib_n

        ### Add to the right ###
        #contrib_j[np.roll(self.ix, -1, axis=1), self.iy, self.iz] += contrib_m

        ### Try if it works with the periphery ###
        grad[self.ix, self.iy, self.iz] += contrib_k
        grad[np.roll(self.ix, 1, axis=1), self.iy, self.iz] += contrib_n
        grad[np.roll(self.ix, -1, axis=1), self.iy, self.iz] += contrib_m
        
        return grad #contrib_j

    def collect_gradient(self):
        grad_pot, grad_periphery = self.update_force_bond()
        grad_pot_dist = np.concatenate([np.sum(grad_pot, axis=-2), grad_periphery])        

        grad_pot_angle = self.update_force_angle()
        grad_out_of_plane = self.update_out_of_plane()
#        coulomb = self.coulomb_force() # NB: Coulomb forces switched off for now
#        grad = np.sum(grad_pot_angle, axis = -2) + np.sum(grad_out_of_plane, axis = -2) + grad_pot_dist # + 0*coulomb # NB: Coulomb forces switched off for now
        grad = grad_pot_dist + np.sum(grad_pot_angle, axis=-2)
        #freeze = np.array([15,21,22,28,29,35])
        #grad[freeze] *= 0.

        # self.a = - (1 / self.m)[...,NA] * grad


        return grad

    def velocity_verlet(self):
        v_t_half = self.v + 0.5 * self.a * self.dt
        self.vertex_coords = self.vertex_coords + v_t_half * self.dt
        self.update_displacements()
        self.collect_gradient()
        self.v = self.v + 0.5 * (self.a) * self.dt
        n = - self.vertex_coords[0]
        #self.vertex_coords = self.vertex_coords + np.repeat(n[np.newaxis,:],self.num_of_vertices,axis=0)

    def update(self):
        #self.disturb()
        for _ in range(50):
            self.velocity_verlet()
            #self.update_displacements() 
            #grad = self.collect_gradient()
            #self.vertex_coords[:self.n_carbon] -= self.stepsize * grad * 10
            #self.vertex_coords -= self.stepsize * grad * 10
        
        # update the mesh
        self.update_mesh()

    def disturb(self):
        self.vertex_coords[:self.n_carbon] += np.random.normal(0,self.scale,[self.n_carbon,3])
        self.scale *= 0.9

    def update_hinge_angles(self):
        self.hinge_angles = angle_vec(self.face_normals[np.array(self.hinges[0])[:,0]], self.face_normals[np.array(self.hinges[0])[:,1]] , degrees=False)

    def close_unfolding(self):
            #for active_hinge in self.open_hinges[0]:
        for hinges in self.open_hinges:
            for active_hinge in hinges:
                #step_size = self.angles_hinge[active_hinge] - self.angle_steps[self.stage + 1][active_hinge]
                step_size = self.step_size[active_hinge]
                affected_children_periphery = add_periphery(self.affected_children, self.parent_atom, self.n_carbon)
                update_transform(self.vertex_coords, active_hinge, self.hinges, affected_children_periphery, delta_phi = step_size)
                #self.update_face_normals()
                #self.update_hinge_angles()
            
            #self.stage += 1
            #if self.stage == self.num_of_steps:
                #self.open_hinges.pop(0)
            #    if len(self.open_hinges) > 0:
            #        self.stage = 0

            self.update_mesh()
            #for face in range(len(self.graph_unfolding_faces)):
            #    midpoint = self.vertex_coords[self.graph_unfolding_faces[face]].mean(axis=0)
            #    self.midpoints[face] = midpoint
            #self.vertex_mesh = np.concatenate([self.vertex_coords,self.midpoints])

    def enforce_distance(self):
        return
    
    def optimise_geometry(self, delta = 1.):
        self.update_displacements()
        self.E = harm_pot(self.R_unfolding, self.spring_lengths, self.spring_constants) # + angle_spring_pot(self.vertex_coords, 
        print(f"E = {np.sum(self.E)}")
        
        dG = self.collect_gradient()


        
        step = - delta * dG
        self.vertex_coords += step
        return

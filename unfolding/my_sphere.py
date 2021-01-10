import pyqtgraph.opengl as gl
import numpy as np

class my_sphere(gl.GLMeshItem):
    def __init__(self, radius=1., rows=4, cols=8, offset=True):
        verts = np.empty((rows+1, cols, 3), dtype=float)
        
        ## compute vertexes
        phi = (np.arange(rows+1) * np.pi / rows).reshape(rows+1, 1)
        s = radius * np.sin(phi)
        verts[...,2] = radius * np.cos(phi)
        th = ((np.arange(cols) * 2 * np.pi / cols).reshape(1, cols)) 
        if offset:
            th = th + ((np.pi / cols) * np.arange(rows+1).reshape(rows+1,1))  ## rotate each row by 1/2 column
        verts[...,0] = s * np.cos(th)
        verts[...,1] = s * np.sin(th)
        verts = verts.reshape((rows+1)*cols, 3)[cols-1:-(cols-1)]  ## remove redundant vertexes from top and bottom
        

        ## compute faces
        faces = np.empty((rows*cols*2, 3), dtype=np.uint)
        rowtemplate1 = ((np.arange(cols).reshape(cols, 1) + np.array([[0, 1, 0]])) % cols) + np.array([[0, 0, cols]])
        rowtemplate2 = ((np.arange(cols).reshape(cols, 1) + np.array([[0, 1, 1]])) % cols) + np.array([[cols, 0, cols]])
        for row in range(rows):
            start = row * cols * 2 
            faces[start:start+cols] = rowtemplate1 + row * cols
            faces[start+cols:start+(cols*2)] = rowtemplate2 + row * cols
        faces = faces[cols:-cols]  ## cut off zero-area triangles at top and bottom
        
        ## adjust for redundant vertexes that were removed from top and bottom
        vmin = cols-1
        faces[faces<vmin] = vmin
        faces -= vmin  
        vmax = verts.shape[0]-1
        faces[faces>vmax] = vmax
        super().__init__(vertexes=verts, faces=faces, smooth=True, drawEdges=False, shader='shaded')
        self.my_verts = verts
        self.vertexes = verts
        self.faces = faces
        #print(gl.__file__)

    def translate(self, dr):
        new_verts = self.my_verts + dr
        #print(new_verts)
        self.setMeshData(vertexes=new_verts, faces=self.faces)

class molecule(list):
    def __init__(self, n_atoms):
        self.atoms = [None] * n_atoms
        self.radii = 0.2 * np.ones(n_atoms)
        for i in range(n_atoms):
            self.atoms[i] = my_sphere()
    
    def update(self, coordinates):
        for i in len(range(self.atoms)):
            self.atoms[i].translate(coordinates[i]) 
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import time
import ui_main
from ui_scan import *
from Application_functions import *
from make_hdf5 import *
from load_hdf5 import *
from Unfolding_new import Unfolding
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication,  QFileDialog, QWidget, QMessageBox, QInputDialog, QShortcut
from PyQt5.QtGui import QIcon, QHBoxLayout, QVBoxLayout, QPushButton, QCheckBox, QSpinBox, QKeySequence, QGridLayout, QLabel
#from PyQt5.Qt3DExtras import QSphereMesh
from PyQt5.QtOpenGL import *
from pyqtgraph.Qt import QtCore

import pyqtgraph.opengl as gl
from pyqtgraph import PlotWidget, plot
import pyqtgraph
from my_sphere import my_sphere


try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QApplication.translate(context, text, disambig)


class Window(QWidget):
    def __init__(self, unfolding, parent=None):
        super(Window, self).__init__(parent)
        self.listCheckBox = []
        self.listLabel = []
        self.unfolding = unfolding

        for hinge in unfolding.all_hinges:
            self.listCheckBox.append(str(hinge))
            self.listLabel.append('')

        grid = QGridLayout()

        for i, v in enumerate(self.listCheckBox):
            self.listCheckBox[i] = QCheckBox(v)
            self.listLabel[i] = QLabel()
            grid.addWidget(self.listCheckBox[i], i, 0)
            grid.addWidget(self.listLabel[i],    i, 1)
            if unfolding.all_hinges[i] in unfolding.open_hinges: 
                self.listCheckBox[i].setChecked(True)
            else: 
                self.listCheckBox[i].setChecked(False)

        self.button = QPushButton("Update hinges")
        self.button.clicked.connect(self.checkboxChanged)
        self.labelResult = QLabel()

        grid.addWidget(self.button,     10, 0, 1,2)     
        grid.addWidget(self.labelResult,11, 0, 1,2)  
        self.setLayout(grid)
        self.setWindowTitle('Hinge Selection')

    def checkboxChanged(self):
        self.labelResult.setText("")
        open_hinges = []
        for i, v in enumerate(self.listCheckBox):
            self.listLabel[i].setText("True" if v.checkState() else "False")
            self.labelResult.setText("{}, {}".format(self.labelResult.text(),
                                                     self.listLabel[i].text()))
            if v.checkState():
                open_hinges.append(self.unfolding.all_hinges[i])
        self.unfolding.open_hinges = open_hinges
        self.close()
                

class Unfolding_App(QMainWindow, ui_main.Ui_MainWindow):
    
    def __init__(self, parent=None):
        self.unfolding = None
        self.mesh = None
        self.status = False
        self.filename = "/home/ben/Nextcloud/Documents/KU/Master_thesis/thesis-carbon-folding/C_20_data/C60data.h5"
        self.root = 0

        pyqtgraph.setConfigOption('background', 'w') #before loading widget
        super(Unfolding_App, self).__init__(parent)
        self.setupUi(self)
        #self.btnUp.clicked.connect(self.update)
        ### To close continuously
        self.btnUp.clicked.connect(self.close_unfolding)
        self.spinStep.valueChanged.connect(self.step_changed)
        self.btnFin.clicked.connect(self.show_final_position)
        self.btnInit.clicked.connect(self.show_init_position)
        self.btnSelHinge.clicked.connect(self.selectHinges)
        self.btnClose.clicked.connect(self.close_unfolding)
        self.energy_Plot.plotItem.showGrid(True, True, 0.7)
        self.sc_close_uf = QShortcut(QKeySequence("Ctrl+N"), self)
        self.sc_close_uf.activated.connect(self.close_unfolding)
        self.hinge_step = 200
        
        self.sc_export_uf = QShortcut(QKeySequence("Ctrl+E"), self)
        self.sc_export_uf.activated.connect(self.exportUnfolding)

        self.molecule = None
        
    def openFileNameDialog(self):
        if self.filename != None and self.root != None:
            #dual_unfolding, graph_unfolding, graph_unfolding_faces, vertices_final, bonds_toBe, lengths_toBe, angles_f, opt_geom, halogene_positions, graph, graph_faces = 
            dual_unfolding, graph_unfolding, graph_unfolding_faces, vertices_final, bonds_toBe, lengths_toBe, angles_f, opt_geom, halogen_positions, neighbours, graph_faces = read_unfolding(self.filename)
            self.unfolding = Unfolding(dual_unfolding, graph_unfolding_faces, graph_faces, graph_unfolding, neighbours, halogen_positions=halogen_positions, root_node=0, bonds_toBe=bonds_toBe, angles_f=angles_f)
            #self.unfolding.load_position(opt_geom)
            self.show_unfolding()
        else:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","*.h5", options=options)
            if fileName:
                try:
                    dual_unfolding, graph_unfolding, graph_unfolding_faces, vertices_final, bonds_toBe, lengths_toBe, angles_f, opt_geom, halogene_positions, graph = read_unfolding(fileName)
                    root = self.chooseRoot(graph_unfolding)
                    self.unfolding = Unfolding(dual_unfolding, graph_unfolding_faces, graph_faces, graph_unfolding, neighbours, halogen_positions=halogen_positions, root_node=0, bonds_toBe=bonds_toBe,angles_f=angles_f)
                    self.show_unfolding()
                    self.root = root
                    self.filename = fileName
                except:
                    QMessageBox.about(self, "Error loading file", "The data could not be loaded. Please check the integrity of your file!")

    def exportUnfolding(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","Gaussian Files (*.com)", options=options)
        file = open(fileName,'w')
        text = self.unfolding.write_positions()
        file.write(text)
        file.close()
    
    def update_geometry(self, geometry):
        self.unfolding.vertex_coords = geometry
        self.unfolding.update_mesh()
        self.mesh.setMeshData(
            vertexes=self.unfolding.vertex_mesh, faces=self.unfolding.faces, faceColors = self.unfolding.color)
        for i in range(len(self.unfolding.vertex_coords)):
                self.molecule[i].translate(self.unfolding.vertex_coords[i])


    def loadGeometry(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()","","Log Files (*.log)", options=options)
        geometry = read_geometry(fileName)
        self.update_geometry(geometry)


    def writeNP_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","Text Files (*.txt)", options=options)
        np.savetxt(fileName,self.unfolding.vertex_coords)

    def chooseRoot(self, graph_unfolding):
        i, okPressed = QInputDialog.getInt(self, "Root node","Please enter a root node:", 0, 0, len(graph_unfolding), 1)
        if okPressed:
            return i

    def step_changed(self):
        self.hinge_step = self.spinStep.value()
        #self.label.setText("Current Value Is : " + str(spinValue))

    def show_unfolding(self):
        if self.unfolding:
            self.mesh = gl.GLMeshItem(
            vertexes=self.unfolding.vertex_mesh,
            faces=self.unfolding.faces, faceColors = self.unfolding.color, smooth=False, drawEdges=True, shader='shaded')
            self.mesh.setGLOptions('additive')
            self.gl_widget.addItem(self.mesh)
            self.molecule = [None] * len(self.unfolding.vertex_coords)
            for i in range(len(self.unfolding.vertex_coords)):
                self.molecule[i] = my_sphere(radius=0.2)
                self.gl_widget.addItem(self.molecule[i])
                self.molecule[i].translate(self.unfolding.vertex_coords[i])
            self.spinStep.setValue(int(self.unfolding.stepsize))
        else:
            QMessageBox.about(self, "Error drawing Unfolding", "Please define an unfolding to draw!")            

    def show_final_position(self):
        if self.mesh:
            self.mesh.setMeshData(
            vertexes=self.unfolding.vertex_mesh_final, faces=self.unfolding.faces, faceColors = self.unfolding.color)
            self.unfolding.vertex_coords = self.unfolding.vertices_final
        else:
            QMessageBox.about(self, "Error", "Please define a mesh to set the final data!")    

    def show_init_position(self):
        dual_unfolding, graph_unfolding, graph_unfolding_faces, vertices_final, bonds_toBe, lengths_toBe, angles_f, opt_geom, halogene_positions, graph = read_unfolding(self.filename)
        self.unfolding = Unfolding(dual_unfolding, graph_unfolding_faces, graph_unfolding, graph, self.root, bonds_toBe, lengths_toBe, angles_f,halogene_positions, vertices_final=vertices_final)
        self.unfolding.load_position(opt_geom)
        self.mesh.setMeshData(
            vertexes=self.unfolding.vertex_mesh, faces=self.unfolding.faces, faceColors = self.unfolding.color)
        #if self.mesh:
        #    self.unfolding.init_vertices()
        #    self.mesh.setMeshData(
        #    vertexes=self.unfolding.vertex_mesh, faces=self.unfolding.faces, faceColors = self.unfolding.color)
        #else:
        #    QMessageBox.about(self, "Error", "Please define a mesh to set the initial data!")                

    def selectHinges(self):
        self.Window = Window(unfolding = self.unfolding)
        self.Window.show()


    def close_unfolding(self):
        for _ in range(self.hinge_step):
            self.unfolding.close_unfolding()

        self.mesh.setMeshData(
                vertexes=self.unfolding.vertex_mesh, faces=self.unfolding.faces, faceColors = self.unfolding.color)
        for i in range(len(self.unfolding.vertex_coords)):
                self.molecule[i].translate(self.unfolding.vertex_coords[i])
        
        if self.chkIntegrate.isChecked():
            QtCore.QTimer.singleShot(1, self.close_unfolding) # QUICKLY repeat

    def update(self):
        #t1 = time.clock()
        if self.mesh != None :

            self.unfolding.update()
            self.unfolding.close_unfolding()
            self.unfolding.update_hinge_angles()
            self.mesh.setMeshData(
            vertexes=self.unfolding.vertex_mesh, faces=self.unfolding.faces, faceColors = self.unfolding.color)
            #points = self.unfolding.E_kin_hist.shape[0] #number of data points

            for i in range(len(self.unfolding.vertex_coords)):
                self.molecule[i].translate(self.unfolding.vertex_coords[i])

            #X = np.arange(points)
            #Y1 = self.unfolding.E_kin_hist
            #Y2 = self.unfolding.E_spring_hist
            #Y3 = self.unfolding.E_rep_hist
            #pen1 = pyqtgraph.mkPen(width = 1, color='r')
            #pen2 = pyqtgraph.mkPen(width = 1, color='b')
            #pen3 = pyqtgraph.mkPen(width = 1, color='g')
            #self.energy_Plot.plot(X,Y1,pen = pen1,clear = True)
            #self.energy_Plot.plot(X,Y2,pen = pen2,clear = False)
            #self.energy_Plot.plot(X,Y3,pen = pen3,clear = False)
            
        if self.chkIntegrate.isChecked():
            QtCore.QTimer.singleShot(1, self.update) # QUICKLY repeat

    def select_directory(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        directory = str(QFileDialog.getExistingDirectory(self, "Select Input Directory",options=options))
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","HDF5 files (*.h5)", options=options)
        #print(fileName)
        form_datasets(directory, title=fileName)
    
    def show_stage(self):
        #print(self.current_stage)
        self.update_geometry(self.Geo_merged[self.current_stage,-1])
        self.Active_step.setData([self.d_CC_merged[self.current_stage]], [self.E_merged[self.current_stage]])
        self.btnUp.setText(str(self.d_CC_merged[self.current_stage]))

    def load_hdf5_scan(self):
        self.openFileNameDialog()
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()","Select HDF5 file","HDF5 files (*.h5)", options=options)
        self.d_CC, self.E_init, self.E_final, self.geometries, self.d_CC_pending, self.E_init_pending, self.E_final_pending, self.geometries_pending = load_h5(fileName)
        self.E_merged, self.Geo_merged, self.d_CC_merged, self.pending = merge(self.E_final, self.E_final_pending, self.geometries, self.geometries_pending, self.d_CC, self.d_CC_pending)
        
        ### Center the geometries ###
        self.Geo_merged -= np.repeat(self.Geo_merged[:,:,0,:][:,:,NA,:],self.Geo_merged.shape[2],axis=2)
        self.current_stage = len(self.d_CC_merged) - 1
        
        points = self.E_final.shape[0]
        X = np.arange(points)
        self.E_final_line = self.energy_Plot.plot(self.d_CC, self.E_final, pen ='b', symbol ='x', symbolPen ='b', symbolBrush = 0.2, name ='Energy after optimisation')
        self.E_final_line_pending = self.energy_Plot.plot(self.d_CC_pending, self.E_final_pending, pen ='r', symbol ='x', symbolPen ='r', symbolBrush = 0.2, name ='Pending or failed')
        self.Active_step = self.energy_Plot.plot([self.d_CC_merged[self.current_stage]], [self.E_merged[self.current_stage]], pen =None, symbol ='o', symbolPen ='g', symbolBrush = 0.5, name ='Current stage')
        #self.update_geometry(self.geometries[self.current_stage,-1])
        self.show_stage()
        set_scan_ui(self)
        self.btnFin.clicked.connect(self.increase_d_CC)
        self.btnInit.clicked.connect(self.decrease_d_CC)
        self.btnUp.setText(str(self.d_CC_merged[self.current_stage]))

    def decrease_d_CC(self):
        self.current_stage -= 1
        if self.current_stage < 0:
            self.current_stage = 0
        self.show_stage()
        

    def increase_d_CC(self):
        self.current_stage += 1
        if self.current_stage > len(self.d_CC_merged) - 1:
            self.current_stage = len(self.d_CC_merged) - 1
        self.show_stage()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Unfolding_App()
    ex.show()
    ex.update()
    time.sleep(0.01)
    sys.exit(app.exec_())
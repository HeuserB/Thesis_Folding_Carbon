import sys
from PyQt5 import Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vedo import show, Spheres, Mesh, printc, Plotter, Picture, Text2D
import numpy as np
from functions_folding import update_transform

def plot_unfolding_vedo(planar_geometry,unfolding_faces,radii=1.0):
    pts = [(planar_geometry[i, 0], planar_geometry[i, 1], planar_geometry[i, 2]) for i in range(len(planar_geometry))]
    rads = [radii for i in range(len(planar_geometry))]  # radius=0 for y=0

    s1 = Spheres(pts, r=rads, c="lb", res=8)

    mesh = Mesh([planar_geometry, np.array(unfolding_faces)])
    mesh.backColor('violet').lineColor('black').lineWidth(2)
    labs = mesh.labels('id').c('black')

    # retrieve them as numpy arrays
    printc('points():\n', mesh.points(), c=3)
    printc('faces(): \n', mesh.faces(),  c=3)
    #show(mesh, labs, __doc__, viewup='z', axes=1)
    show([s1,mesh], labs, at=0, axes=2, interactive=True).close()

class MainWindow(Qt.QMainWindow):
    
    def __init__(self, planar_geometry=None, unfolding_faces=None, hinges=None, affected_vs=None, angles_final=None, radii=1.0, parent=None):

        Qt.QMainWindow.__init__(self, parent)
        self.frame = Qt.QFrame()
        self.layout = Qt.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.hinges = hinges
        self.planar_geometry = planar_geometry
        self.unfolding_faces = unfolding_faces
        self.radii = radii
        self.hinge_queque = list(np.arange(len(hinges[0])))
        self.affected_vs = affected_vs
        self.angles_final = angles_final


        # Create vedo renderer and add objects and callbacks
        self.vp = Plotter(qtWidget=self.vtkWidget)
        self.vp._timer_event_id = None
        self.cbid = self.vp.addCallback("key press", self.onKeypress)
        self.text2d = Text2D("Use slider to change contrast")

        #self.slider = Qt.QSlider(1)
        pts = [(self.planar_geometry[i, 0], self.planar_geometry[i, 1], self.planar_geometry[i, 2]) for i in range(len(self.planar_geometry))]
        rads = [self.radii for i in range(len(self.planar_geometry))]  # radius=0 for y=0

        self.spheres = Spheres(pts, r=rads, c="lb", res=8)

        self.mesh = Mesh([self.planar_geometry, np.array(self.unfolding_faces)])
        self.mesh.backColor('violet').lineColor('black').lineWidth(2)
        labs = self.mesh.labels('id').c('black')

        #button1 = QPushButton('Button-1', self)
        self.button1 = Qt.QPushButton('Close next hinge', self)
  
        self.layout.addWidget(self.vtkWidget)
        self.layout.addWidget(self.button1)
        #self.layout.addWidget(self.slider)

        #self.slider.valueChanged.connect(self.onSlider)
        self.button1.clicked.connect(self.on_click)

        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)
        self.vp.show([self.spheres,self.mesh], labs, at=0, axes=2, interactive=True)
        self.show()                                            # show the Qt Window


    def on_click(self):
        active_hinge = self.hinge_queque.pop(0)
        print(f'Self final angles are: {self.angles_final}')
        update_transform(self.planar_geometry,active_hinge, self.hinges, self.affected_vs, self.angles_final[active_hinge])
        self.mesh.points(self.planar_geometry)
        self.spheres.pos = self.planar_geometry
        print('Now closing hinge: ')
    
    def onSlider(self,value):
        self.imgActor.window(value*10) # change image contrast
        self.text2d.text(f"window level is now: {value*10}")
        self.vp.render()

    def onKeypress(self, evt):
        printc("You have pressed key:", evt.keyPressed, c='b')
        if evt.keyPressed=='q':
            self.vp.close()
            self.vtkWidget.close()
            exit()
            
    def onClose(self):
        self.vtkWidget.close()

if __name__ == "__main__":
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    app.aboutToQuit.connect(window.onClose)
    app.exec_()

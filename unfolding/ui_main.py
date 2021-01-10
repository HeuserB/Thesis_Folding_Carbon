from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication,  QFileDialog, QWidget, QMessageBox, QInputDialog
from PyQt5.QtGui import QIcon, QHBoxLayout, QVBoxLayout, QPushButton, QCheckBox, QSpinBox
import pyqtgraph.opengl as gl
from pyqtgraph import PlotWidget, plot
from pyqtgraph.Qt import QtCore


try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        self.resize(1920, 1080)
        self.status = False
        self.setAutoFillBackground(False)
        self.setDocumentMode(False)

        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))


        self.exitAct = QAction(QIcon('exit.png'), '&Exit', self)        
        self.exitAct.setShortcut('Ctrl+Q')
        self.exitAct.setStatusTip('Exit application')
        self.exitAct.triggered.connect(qApp.quit)

        self.loadAct = QAction(QIcon('exit.png'), '&Import Unfolding', self)        
        self.loadAct.setShortcut('Ctrl+L')
        self.loadAct.setStatusTip('Import Unfolding')
        self.loadAct.triggered.connect(self.openFileNameDialog)

        self.exportAct = QAction(QIcon('exit.png'), '&Export Unfolding', self)        
        self.exportAct.setShortcut('Ctrl+E')
        self.exportAct.setStatusTip('Export Unfolding')
        self.exportAct.triggered.connect(self.exportUnfolding)
        
        self.writeNPAct = QAction(QIcon('exit.png'), '&Write coordinates as np file', self)
        self.writeNPAct.setShortcut('Ctrl+W')
        self.writeNPAct.setStatusTip('Write coordinates')
        self.writeNPAct.triggered.connect(self.writeNP_file)

        self.loadGeometryAct = QAction(QIcon('exit.png'), '&Load geometry from .log file' , self)
        #self.loadGeometry.setShortcut('Ctrl+W')
        self.loadGeometryAct.setStatusTip('Load geometry')
        self.loadGeometryAct.triggered.connect(self.loadGeometry)

        self.select_directoryAct = QAction(QIcon('exit.png'), '&Select a directory to generate hdf5 files from' , self)
        self.select_directoryAct.setStatusTip('Select directory')
        self.select_directoryAct.triggered.connect(self.select_directory)

        self.load_HDF5Act = QAction(QIcon('exit.png'), '&Select a hdf5 files for the DFT data' , self)
        self.load_HDF5Act.setStatusTip('Select HDF5')
        self.load_HDF5Act.triggered.connect(self.load_hdf5_scan)

        self.statusBar()

        self.menubar = self.menuBar()
        self.fileMenu = self.menubar.addMenu('&File')
        self.fileMenu.addAction(self.exitAct)
        self.fileMenu.addAction(self.loadAct)
        self.fileMenu.addAction(self.exportAct)
        self.fileMenu.addAction(self.writeNPAct)
        self.fileMenu.addAction(self.loadGeometryAct)
        self.fileMenu.addAction(self.select_directoryAct)
        self.fileMenu.addAction(self.load_HDF5Act)


        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setSpacing(10)

        self.verticalLayout = QVBoxLayout()

        ### Containing the integration buttons
        self.horizontalLayout_2 = QHBoxLayout()


        ### Containing the inital/final position show
        self.finalPositionLayout = QHBoxLayout()


        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setCameraPosition(distance=20, elevation=40)
        self.gl_widget.setGeometry(0, 110, 1920, 1080)
        
        self.energy_Plot = PlotWidget()
        self.energy_Plot.setObjectName(_fromUtf8("energyPlot"))
        self.energy_Plot.setBackground('k')
        # set properties of the label for y axis 
        self.energy_Plot.setLabel('left', 'E(B3LYP)', units ='H') 
  
        # set properties of the label for x axis 
        self.energy_Plot.setLabel('bottom', 'CC-distance', units ='A') 
        
        # adding legend 
        self.energy_Plot.addLegend()

        self.horizontalLayout.addWidget(self.gl_widget, 4)

        ### Push button for sinle integration
        self.btnUp = QPushButton()
        self.btnUp.setObjectName(_fromUtf8("btnUp"))


        ### Check box for continous integration
        self.chkIntegrate = QCheckBox()
        self.chkIntegrate.setObjectName(_fromUtf8("chkIntegrate"))

        ### SpinBox to set the size of the integration step
        self.spinStep = QSpinBox()
        self.spinStep.setMinimum(0)
        self.spinStep.setMaximum(10000)
        self.spinStep.setObjectName(_fromUtf8("spinStep"))
       

        ### final position button
        self.btnFin = QPushButton()
        self.btnFin.setObjectName(_fromUtf8("btnFin"))

        ### initial position button
        self.btnInit = QPushButton()
        self.btnInit.setObjectName(_fromUtf8("btnPos"))

        ### select hinges button
        self.btnSelHinge = QPushButton()
        self.btnSelHinge.setObjectName(_fromUtf8("btnSelHinge"))

        ### close unfolding button
        self.btnClose = QPushButton()
        self.btnClose.setObjectName(_fromUtf8("btnClose"))

        ### add the buttons to the integration layout
        self.horizontalLayout_2.addWidget(self.spinStep)
        self.horizontalLayout_2.addWidget(self.btnUp)
        self.horizontalLayout_2.addWidget(self.chkIntegrate)

        ## add final position button to layout
        self.finalPositionLayout.addWidget(self.btnInit)
        self.finalPositionLayout.addWidget(self.btnFin)
        self.finalPositionLayout.addWidget(self.btnSelHinge)
        self.finalPositionLayout.addWidget(self.btnClose)



        ### add integration and final position layout and plot widget to right side layout
        self.verticalLayout.addLayout(self.horizontalLayout_2,1)
        self.verticalLayout.addLayout(self.finalPositionLayout,1)
        self.verticalLayout.addWidget(self.energy_Plot, 6)

        self.horizontalLayout.addLayout(self.verticalLayout, 1)


        self.widget = QWidget()
        self.widget.setLayout(self.horizontalLayout)
        self.setCentralWidget(self.widget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.setWindowTitle('Fullerene Unfolding')    

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.btnUp.setText("Integrate")
        self.chkIntegrate.setText("Keep integrating")
        self.btnFin.setText("Final position")
        self.btnInit.setText("Initial position")
        self.btnClose.setText("Close unfolding")
        self.btnSelHinge.setText("Select Hinges")
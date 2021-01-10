def set_scan_ui(Ui_MainWindow):
    ### Set the new names of the buttons ### 
    Ui_MainWindow.btnFin.setText("Larger CC-distance")
    Ui_MainWindow.btnInit.setText("Smaller CC-distance")

    ### Hide unnecessecary widgets ###
    Ui_MainWindow.spinStep.hide()
    #Ui_MainWindow.btnUp.hide()
    #Ui_MainWindow.horizontalLayout_2.hide()
    Ui_MainWindow.chkIntegrate.hide()
    Ui_MainWindow.btnSelHinge.hide()
    Ui_MainWindow.btnClose.hide()

    
    ### Connect the buttons to other functions ###
    Ui_MainWindow.btnFin.disconnect()
    Ui_MainWindow.btnInit.disconnect()
    Ui_MainWindow.btnFin.setShortcut('Ctrl+right')
    Ui_MainWindow.btnInit.setShortcut('Ctrl+left')
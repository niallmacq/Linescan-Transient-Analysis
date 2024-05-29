import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLineEdit, QFileDialog, QPushButton, QSpinBox, QVBoxLayout, QLabel, QTextEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import uic, QtCore
import tifffile
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT 
from matplotlib.figure import Figure
from os import path
from matplotlib.widgets import SpanSelector
from scipy import signal, ndimage
import pandas as pd
from PyQt5 import QtGui
import qimage2ndarray
import czifile
import getAPD90
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def analyse_global_trans(fn, prof, iistart, iistop, interval=0.0005, background=0.0):
    prof=prof[iistart:iistop]
    prof=getAPD90.BaselineRefit(prof)
    #analyse transients
    x, y = (np.arange(prof.size) * interval, prof/ndimage.gaussian_filter1d(prof,3).min())
    y=ndimage.uniform_filter1d(y,7)
    np.savetxt(fn.split('.')[0]+'trans.txt', np.array((x,y)))
    df = getAPD90.analyze_data(x,y)
    fig=plt.figure()
    plt.plot(x,y)
    plt.show()
    (icross, ythresh) = getAPD90.find_threshold_crossings(x, ndimage.uniform_filter1d(y,3), 0.5)
    (itrough, ipeak, istart, iend) = getAPD90.find_complete_transients(x, ndimage.uniform_filter1d(y,21), icross)
    for i in range(len(itrough)):
        xt = x[itrough[i]-1:iend[i]].copy()
        yt = y[itrough[i]-1:iend[i]].copy()
        xs = x[istart[i]]
        ys = y[istart[i]]
        pdict = getAPD90.analyze_transient(ndimage.uniform_filter1d(yt,21), xt, yt, xs, ys)
        xm1,ym1,xm,ym=getAPD90.calc_transient(i,xt, yt, xs, ys, pdict)
        interval=x[1]-x[0]
        xtemp=xt[int(pdict['TD40']/interval-itrough[i]):iend[i]]
        ytemp=yt[int(pdict['TD40']/interval-itrough[i]):iend[i]]
        fig=plt.figure()
        plt.plot(xt,yt)
        #fit exponential
        try:
            popt, pcov = curve_fit(func, np.float32(xtemp), np.float32(ytemp),p0=[ytemp[0],1/xtemp[-1],ytemp[-1]])
            fit=func(xtemp, *popt)
            plt.plot(xtemp,fit)
            pdict['a'],pdict['b'],pdict['c']=popt
            df.loc[i,'Rate Constant (s)']=popt[1]
        except:
            df.loc[i,'Rate Constant (s)']=np.NaN
        
        
        plt.plot(xm, ym, 'b+')
        
        fig_fn=fn.split('.')[0]+'trans%g.png'%(i+1)
        plt.title('Trans %g'%(i+1))
        plt.savefig(fig_fn)
        plt.close(fig)
    avdf=df.mean(axis=0)
    print (avdf)
    df.to_csv(fn.lower().split('.')[0]+'trans_params.csv')
    avdf.to_csv(fn.lower().split('.')[0]+'avtrans_params.csv')
    np.savetxt(fn.lower().split('.')[0]+'start-stop-bg.txt',np.array([iistart,iistop,background]))

'''def onselect(self):
            indmin, indmax = np.searchsorted(self.x, (self.xmin, self.xmax))
            indmax = min(len(self.x) - 1, indmax)

            thisx = self.x[indmin:indmax]
            thisy = self.y[indmin:indmax]
            self.line1.set_data(thisx, thisy)
            self.ax_a.set_xlim(thisx[0], thisx[-1])
            self.ax_a.set_ylim(thisy.min(), thisy.max())
            self.fig.canvas.draw()'''

def norm_im(im, start, stop, norm_length):
         bl = im[:,start:stop][:,:min(norm_length,im.shape[1])].mean(axis=1)
         nim = im/np.array([bl]*im.shape[1]).T
         return nim

def analyse(current_image, istart, istop, fps,sgf, gf, tz, sp, background=0 ):
    #smooth plot
    sm=ndimage.uniform_filter(ndimage.uniform_filter(current_image[istart:istop],(1,3)),(1,5))
    sg_im=np.array([signal.savgol_filter(ndimage.uniform_filter(sm[:,x]),sgf,3) for x in range(current_image.shape[1])])
    #interpolate plot
    pixel_time_interval = 1/np.float64(fps)*1000
    zoom_factor = pixel_time_interval*(1/tz)#zoom to get 0.5 ms pixels
    print(zoom_factor)
    zim=ndimage.zoom(sg_im-background,(1,zoom_factor))
    zim=ndimage.gaussian_filter(zim,(1,gf))#filter it a bit with a 2sigma gaussian...no temporal effects, spatial minimal
    prof=ndimage.gaussian_filter1d(zim.mean(axis=0),sp)
    peakind = signal.argrelmax(np.diff(prof),order=int(400/pixel_time_interval*zoom_factor))[0]#find peak indices
    if peakind[0]-int(40*zoom_factor)<0:
        peakind=np.delete(peakind,0)
    if peakind[-1]+int(140*zoom_factor)>len(prof):
        peakind=np.delete(peakind,-1)
    print(peakind)
    #stp=[np.argmin(zim.mean(axis=0)[i-int(40*zoom_factor):i+int(10*zoom_factor)]) for i in peakind]#start points
    #align on autocorrelation
    
    corr_array=[np.correlate(np.diff(zim.mean(axis=0)[peakind[0]-int(30*zoom_factor):peakind[0]-int(30*zoom_factor)+int(80*zoom_factor)]) ,np.diff(zim.mean(axis=0)[peakind[i]-int(40*zoom_factor):peakind[i]-int(40*zoom_factor)+int(140*zoom_factor)]),mode="full") for i in range(len(peakind))]
    c_arr=[corr_array[0].argmax()-corr_array[i].argmax() for i in np.arange(len(corr_array))]
    l=[peakind[i]+c_arr[i] for i in np.arange(4)]
    l=np.diff(np.array(l))
    print ('Detected intervals'+str((l*0.5/1000))+' s')
    
    #average transients
    #aa_im=[zim[:,peakind[i]-int(80*zoom_factor)+stp[i]:peakind[i]-int(80*zoom_factor)+stp[i]+int(80*zoom_factor)] for i in range(len(peakind))]#list of images
    aa_im=[zim[:,peakind[i]-int(30*zoom_factor)+c_arr[i]:peakind[i]-int(30*zoom_factor)+int(80*zoom_factor)+c_arr[i]] for i in np.arange(len(peakind))]#list of images
    #peakind[i]-int(80*zoom_factor):peakind[i]-int(80*zoom_factor)+int(120*zoom_factor)]
    [print (im.shape) for im in aa_im]
    for i in np.arange(len(aa_im)):
        if i == 0:
            isplit=aa_im[i].shape[1]
        else:
             if aa_im[i].shape[1]<isplit:
                  aa_im.pop(i)
    aa_nim=[norm_im(im,0,int(5*zoom_factor),im.shape[1]) for im in aa_im]
    #tifffile.imshow(np.array(aa_nim))
    
    

    av_im=np.array(aa_nim).mean(axis=0)
    nim_norm=(ndimage.gaussian_filter(av_im,1)-1)/(ndimage.gaussian_filter(av_im,1).mean(axis=0).max()-1)
    #tifffile.imshow(np.array(nim_norm))
    imax=nim_norm.mean(axis=0).argmax()
    a=np.argmax(np.diff(nim_norm[:,:]>0.5),axis=1)
    a=(a-np.argmin(nim_norm.mean(axis=0)))*0.5
    a_=np.where(a>45,np.nan,a)
    print(np.nanmedian(a_))
    print(np.nanstd(a_))
    return a, a_, nim_norm, np.nanmedian(a_), np.nanstd(a_), peakind,np.array(istart+c_arr+peakind)*0.5/1000, zoom_factor, prof


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("gui.ui",self)
        #define widgets
        self.cropped=False
        self.lineEdit_parameters_basic_pixel_width = self.findChild(QLineEdit, "lineEdit_parameters_basic_pixel_width")
        self.lineEdit_parameters_basic_fps = self.findChild(QLineEdit, "lineEdit_parameters_basic_fps")
        self.lineEdit_parameters_upstroke_threshold = self.findChild(QLineEdit, "lineEdit_parameters_upstroke_threshold")
        self.lineEdit_parameters_extracellular_space_threshold = self.findChild(QLineEdit, "lineEdit_parameters_extracellular_space_threshold")
        self.preprocessing_lineEdit_SG_width = self.findChild(QLineEdit, "preprocessing_lineEdit_SG_width")
        self.preprocessing_lineEdit_1dGaussFilter = self.findChild(QLineEdit, "preprocessing_lineEdit_1dGaussFilter")
        self.preprocessing_lineEdit_temporal_zoom = self.findChild(QLineEdit, "preprocessing_lineEdit_temporal_zoom")
        self.preprocessing_lineEdit_smooth_peaks = self.findChild(QLineEdit, "preprocessing_lineEdit_smooth_peaks")
        self.spinBox_parameters_crop_top = self.findChild(QSpinBox, "spinBox_parameters_crop_top")
        self.spinBox_parameters_crop_bottom = self.findChild(QSpinBox, "spinBox_parameters_crop_bottom")
        self.spinBox_parameters_crop_left = self.findChild(QSpinBox, "spinBox_parameters_crop_left")
        self.spinBox_parameters_crop_right = self.findChild(QSpinBox, "spinBox_parameters_crop_right")
        self.button_load_image = self.findChild(QPushButton, "button_load_image")
        self.button_analyze_data = self.findChild(QPushButton,"button_analyze_data")
        self.button_save_outputs = self.findChild(QPushButton,"button_save_outputs")
        self.button_crop = self.findChild(QPushButton,"button_crop")
        self.button_save_tiff = self.findChild(QPushButton,"button_save_tiff")
        self.label = self.findChild(QLabel,"label_23")
        self.layout = self.findChild(QVBoxLayout,"verticalLayout")
        self.textEdit = self.findChild(QTextEdit,"textEdit")
        
        # connect buttons
        self.button_load_image.clicked.connect(self.load_image)
        self.button_analyze_data.clicked.connect(self.analyse_im)
        self.button_crop.clicked.connect(self.crop_im)
        self.button_save_tiff.clicked.connect(self.save_tiff)

        # Actions
        self.loaded=False
        
        #show app
        self.show()

    def plot_im_and_profiles(self,  palongx, palongy, cx = "b", cy = "b"):
        #Add plotting canvas
        
        if self.loaded==False:
            self.fig = Figure()
            self.canvas = FigureCanvasQTAgg(self.fig)
            
            self.layout.addWidget(self.canvas)
            
            self.loaded==True
        self.cropped = False
        
        # Create 3 axes with specified grid dimensions
        self.ax_a = plt.subplot2grid((8,10), (2,8), colspan=2, rowspan=6,fig=self.fig)
        self.linea,= self.ax_a.plot(palongy, np.arange(len(palongy))[-1::-1], color=cy)
        self.ax_a.locator_params(tight=True, nbins=4)
        self.ax_a.xaxis.set_tick_params(labeltop="on", labelbottom='off')
        self.ax_a.yaxis.set_visible(False)
        self.ax_b = plt.subplot2grid((8,10), (2,0), colspan=8, rowspan=6, fig=self.fig)
        self.ax_b.imshow(self.current_image, cmap='gray')
        self.ax_b.axis('tight')# ax_b.
        self.ax_c = plt.subplot2grid((8,10), (0,0), colspan=8, rowspan=2,fig=self.fig)
        self.linec,=self.ax_c.plot(palongx, color=cx)
        self.ax_c.locator_params(tight=True, nbins=4)
        self.ax_c.xaxis.set_visible(False)
        self.ax_c.set_xlim(0, len(palongx))
        self.ax_a.set_ylim(0, len(palongy))
        self.loaded=True
    

    
    
    def crop_im(self):
        if self.loaded==True:
            self.spinBox_parameters_crop_bottom.setMinimum(self.spinBox_parameters_crop_top.value()+1)
            self.spinBox_parameters_crop_right.setMinimum(self.spinBox_parameters_crop_left.value()+1)
            self.crop_left = self.spinBox_parameters_crop_left.value()
            self.crop_right = self.spinBox_parameters_crop_right.value()
            #self.ax_b.axhspan(,np.int64(spinBox_parameters_crop_top.value), fc = "r", ec="none", alpha=0.4)
            if self.cropped==True:
                self.horline.set_ydata([np.int64(self.spinBox_parameters_crop_top.value())]*2)
                self.horline2.set_ydata([np.int64(self.spinBox_parameters_crop_bottom.value())]*2)
                self.vertline.set_ydata([np.int64(self.spinBox_parameters_crop_left.value())]*2)
                self.vertine2.set_ydata([np.int64(self.spinBox_parameters_crop_right.value())]*2)
            self.horline=self.ax_b.axhline(y=np.int64(self.spinBox_parameters_crop_top.value()), color='r', linestyle='-')
            self.horline2=self.ax_b.axhline(y=np.int64(self.spinBox_parameters_crop_bottom.value()), color='r', linestyle='-')
            self.vertline=self.ax_b.axvline(x=np.int64(self.spinBox_parameters_crop_left.value()), color='b', linestyle='-')
            self.vertine2=self.ax_b.axvline(x=np.int64(self.spinBox_parameters_crop_right.value()), color='b', linestyle='-')
            #self.ax_b.axvspan(210,215, fc = "b", ec="none", alpha=0.4)
            self.istart = self.spinBox_parameters_crop_top.value()
            self.istop = self.spinBox_parameters_crop_bottom.value()
            xdata= self.profx.copy()
            xdata[0:self.crop_left]=self.profx.mean()
            xdata[self.crop_right:]=self.profx.mean()
            ydata= self.profy.copy()
            ydata[0:self.istart]=self.profy.mean()
            ydata[self.istop:]=self.profy.mean()
            self.cropped=True
            #step = self.current_image.size / self.current_image.shape[0]
            im_cropped=self.current_image[self.istart:self.istop,self.crop_left:self.crop_right]
            #COLORTABLE=[]
            #for i in range(256): COLORTABLE.append(QtGui.qRgb(i/4,i,i/2))
            QI = qimage2ndarray.array2qimage(im_cropped,True)
            #QI.setColorTable(COLORTABLE)
            self.label.setPixmap(QtGui.QPixmap.fromImage(QI))
            

            

            self.linea.set_xdata(ydata)
            self.linec.set_ydata(xdata)
            self.fig.canvas.draw() 
            self.fig.canvas.flush_events()
    
    def load_image(self):
        self.settings = QtCore.QSettings("pyqt_settings.ini", QtCore.QSettings.IniFormat)
        file_path = self.settings.value("Paths/csvfile", QtCore.QDir.rootPath())
        fname = QFileDialog.getOpenFileName(self, 'Select an image file', filter="All files (*.*);;TIFF (*.tif) ;;CZI (*.czi)", initialFilter='CZI (*.czi)', directory=file_path)
        

        if path.exists(fname[0]):
            finfo = QtCore.QFileInfo(fname[0])
            self.settings.setValue("Paths/csvfile", finfo.absoluteDir().absolutePath())
            print(["Reading "+fname[0]])
            self.statusBar().showMessage("Reading "+fname[0])
            if fname[0].endswith(".czi"):
                img = czifile.imread(fname[0])
                if len(img.shape)>6:
                    img=img[:,:,:,:,:,:,:,0].reshape(img.shape[2],img.shape[6]) 
                else:       
                    img=img.reshape(img.shape[2],img.shape[6])
                from lxml import etree
                try:
                    czi = czifile.CziFile(fname[0])
                    czi_xml_str = czi.metadata()
                    czi_parsed = etree.fromstring(czi_xml_str)
                    fps=str(np.uint64(np.round(1/np.float32((czi_parsed.xpath("////Increment")[0].text))))) #rate of acquisition
                    pixel_width=str(np.float32(czi_parsed.xpath("//Value")[4].text)*1e6)#pixel sixe in microns
                    self.lineEdit_parameters_basic_fps.setText(str(fps))
                    self.lineEdit_parameters_basic_pixel_width.setText(str(pixel_width))
                except:
                    pass
            else:
                img = tifffile.imread(fname[0])
            self.current_image = img
            if self.spinBox_parameters_crop_right.valueChanged.connect(self.crop_im):
                self.spinBox_parameters_crop_right.valueChanged.disconnect()
            if self.spinBox_parameters_crop_top.valueChanged.connect(self.crop_im):
                self.spinBox_parameters_crop_top.valueChanged.disconnect()
            if self.spinBox_parameters_crop_bottom.valueChanged.connect(self.crop_im):
                self.spinBox_parameters_crop_bottom.valueChanged.disconnect()
            if self.spinBox_parameters_crop_left.valueChanged.connect(self.crop_im):
                self.spinBox_parameters_crop_left.valueChanged.disconnect()

            
            self.spinBox_parameters_crop_top.setValue(int(0))
            self.spinBox_parameters_crop_bottom.setValue(int(img.shape[0]))
            self.spinBox_parameters_crop_left.setValue(int(0))
            self.spinBox_parameters_crop_right.setValue(int(img.shape[1]))
            self.spinBox_parameters_crop_top.setMinimum(int(0))
            self.spinBox_parameters_crop_bottom.setValue(int(img.shape[0]))
            self.spinBox_parameters_crop_bottom.setMaximum(int(img.shape[0]))
            self.spinBox_parameters_crop_left.setValue(int(0))
            self.spinBox_parameters_crop_left.setMinimum(int(0))
            self.spinBox_parameters_crop_right.setValue(int(img.shape[1]))
            self.spinBox_parameters_crop_right.setMaximum(int(img.shape[1]))
            
            self.spinBox_parameters_crop_right.valueChanged.connect(self.crop_im)
            self.spinBox_parameters_crop_top.valueChanged.connect(self.crop_im)
            self.spinBox_parameters_crop_bottom.valueChanged.connect(self.crop_im)
            self.spinBox_parameters_crop_left.valueChanged.connect(self.crop_im)
            self.current_image_fname = fname[0]
            self.fps=np.int32(self.lineEdit_parameters_basic_fps.text())
            #plt.imshow(img, cmap="gray")
            #plt.title("Image read")
            #plt.grid(False)
            #plt.show()
            maxx = self.current_image.max(axis=0)
            maxy = self.current_image.max(axis=1)#[-1::-1]
            
            self.profx = self.current_image[:, :].mean(axis=0)
            self.profy = self.current_image.mean(axis=1)
            self.x = self.profy.copy()
            self.y = self.profy.copy()
            
            self.plot_im_and_profiles(self.profx, self.profy, cx="r", cy="b")           # Show the line profile regions using `axvspan` and `axhspan`
            self.statusBar().showMessage("Loaded "+fname[0])
            #self.ax_b.axhspan(145,150, fc = "r", ec="none", alpha=0.4)
            #self.ax_b.axvspan(210,215, fc = "b", ec="none", alpha=0.4)
            
            #pts=self.fig.ginput(2)
            
            #self.istart=np.array(pts,np.int32)[1,:].min()
            #self.istop=np.array(pts,np.int32)[1,:].max()
            #self.spinBox_parameters_crop_top.setValue(int(self.istart))
            #self.spinBox_parameters_crop_bottom.setValue(int(self.istop))
    
    def norm_im(im, start, stop, norm_length):
         bl = im[:,start:stop][:,:norm_length].mean(axis=1)
         nim = im/np.array([bl]*im.shape[1]).T
         return nim

    def analyse_im(self):
        self.istart=np.int64(self.spinBox_parameters_crop_top.value())
        self.istop=np.int64(self.spinBox_parameters_crop_bottom.value())
        self.fps=np.int64(self.lineEdit_parameters_basic_fps.text())
        self.crop_left=np.int64(self.spinBox_parameters_crop_left.value())
        self.crop_right=np.int64(self.spinBox_parameters_crop_right.value())
        sgf = np.int64(self.preprocessing_lineEdit_SG_width.text())
        gf  = np.float64(self.preprocessing_lineEdit_1dGaussFilter.text())
        tz  = np.float64(self.preprocessing_lineEdit_temporal_zoom.text())
        sp = np.int64(self.preprocessing_lineEdit_smooth_peaks.text())
        if self.cropped==True:  
             a, a_, nim_norm, mediana, stda, peakind, stp, zoom_factor, prof=analyse(self.current_image[:,self.crop_left:self.crop_right], self.istart, self.istop, self.fps,sgf, gf, tz, sp)
        else:
             a, a_, nim_norm, mediana, stda, peakind, stp, zoom_factor,prof=analyse(self.current_image, self.istart, self.istop, self.fps,sgf, gf, tz, sp)
        fig=plt.figure()
        plt.plot(a)
        plt.show()
        plt.savefig(self.current_image_fname.split('.')[0]+'profile_plot.png')
        tifffile.imwrite(self.current_image_fname.split('.')[0]+'norm_im.tif',nim_norm)
        np.savetxt(self.current_image_fname.split('.')[0]+'profile.csv',a)
        np.savetxt(self.current_image_fname.split('.')[0]+'trans_times.csv',stp)
        
        df= pd.DataFrame(columns=['mean T50 (ms)','median T50 (ms)', 'std T50 (ms)'])
        df.loc[len(df),:]=np.array([a.mean(), mediana, a.std()])
        df.to_csv(self.current_image_fname.split('.')[0]+'stats.csv')
        self.textEdit.setText('mean T50 (ms) '+str(a.mean())+'\n'+'median T50 (ms) '+str(mediana)+'\n'+'std T50 (ms) '+str(a.std()))
        analyse_global_trans(self.current_image_fname, prof,0,prof.size,0.0005,0)
    
    def save_tiff(self):
        file_path = self.settings.value("Paths/csvfile", QtCore.QDir.rootPath())
        fname = QFileDialog.getSaveFileName(self, 'Save image file', directory=file_path)
        tifffile.imwrite(fname[0],self.current_image[self.istart:self.istop,self.crop_left:self.crop_right])

        




                                         
matplotlib.use("QT5Agg")
app = QApplication(sys.argv)
UIWindow = MainWindow()
app.exec_()


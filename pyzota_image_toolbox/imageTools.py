"""Mix of functions used for image processing and data extraction."""
import numpy as np
from scipy import ndimage as ndi
import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator

from skimage.filters import *
from skimage.morphology import *
from skimage import measure
from skimage import segmentation
from skimage import io
from skimage import exposure
from skimage.feature import peak_local_max

from matplotlib.widgets import  RectangleSelector
from pylab import *

import tifffile as tif

def setup(inputFolder, outputFolder):
    '''
    Creates output folder if it doesnt exist and get input file named from input
    file.
    '''
    try:
        os.stat(outputFolder)
    except:
        os.mkdir(outputFolder)
    filenames = os.listdir(inputFolder+"/")
    return filenames

def createFolder(folderName):
    '''
    Crates folder if it doens not already exist.
    '''
    try:
        os.stat(folderName)
    except:
        os.mkdir(folderName)

def getFileNamesFromFolder(path):
    '''
    Retrives all names of files in folder
    '''
    filenames = os.listdir(path+"/")
    return filenames

def open(pathname):
    '''
    Returns numpy array of image at pathname.
    '''
    Image = plt.imread(pathname)
    Image_Array = np.array(Image)
    return(Image_Array)

def showMe(image, cmap=plt.cm.gray):
    '''
    Shows the given image at runtime
    '''
    plt.imshow(image, cmap=cmap)
    if len(np.shape(image)) < 2: plt.colorbar()
    plt.show()

def selectReigon(image):
    ''' Selects reigon of image with mouse clicks'''
    global mouseCoords
    mouseCoords = [-1,-1,-1,-1]
    def onselect(eclick, erelease):
        #print(' startposition : (%f, %f)' % (eclick.xdata, eclick.ydata))
        #print(' endposition   : (%f, %f)' % (erelease.xdata, erelease.ydata))
        #print(' used button   : ', eclick.button)
        global mouseCoords
        mouseCoords = [eclick.xdata, erelease.xdata, eclick.ydata, erelease.ydata]
        plt.close()

    def toggle_selector(event):
        print(' Key pressed.')
        if event.key in ['Q', 'q'] and toggle_selector.RS.active:
            print(' RectangleSelector deactivated.')
            toggle_selector.RS.set_active(False)
        if event.key in ['A', 'a'] and not toggle_selector.RS.active:
            print(' RectangleSelector activated.')
            toggle_selector.RS.set_active(True)
    fig = figure
    ax = subplot(111)
    ax.imshow(image)
    toggle_selector.RS = RectangleSelector(ax, onselect, drawtype='box')
    connect('key_press_event', toggle_selector)
    plt.show()
    print(mouseCoords)

def blurr(image,sigma=1.0,imageType='RGB'):
    '''
    Perofrms a gaussian blurr on Image with standard deviation sigma.
    '''
    blurredImage = np.zeros(np.shape(image))
    if imageType == 'RGB':
        for layer in range(np.shape(image)[2]):
            blurredImage[:,:,layer] =  gaussian_filter(image[:,:,layer],sigma=sigma)
    if imageType == "sequence":
        for im in range(np.shape(image)[2]):
            blurredImage[:,:,im] =  gaussian_filter(image[:,:,im],sigma=sigma)
    if imageType == "gray":
        blurredImage =  gaussian_filter(image,sigma=sigma)
    return(blurredImage)

def Threshold(Image,blockSize):
    '''
    Performs an adaptive threshold on Image. Threshold value is obtained by
    averaging pixels in each small reigon the size of which is given by
    blocksize.
    '''
    Threshold_Image = threshold_adaptive(Image, blockSize)
    return(Threshold_Image)

def GlobalThreshold(Image):
    '''
    Simple threshold of image using single threshold value over whole image.
    '''
    thresh = threshold_otsu(Image)
    Image = Image > thresh
    return(Image)

def Erode(Image,numberoftimes):
    '''
    Performs binary erosion of image a required number of times.
    '''
    for i in range(numberoftimes):
        Image = binary_erosion(Image)
    return(Image)

def Dilate(Image,numberoftimes):
    '''
    Performs binary dilation on Image a required number of times.
    '''
    for i in range(numberoftimes):
        Image = binary_dilation(Image)
    return(Image)

def DistanceTransform(Image):
    '''
    Takes distance transform of image.
    '''
    return(ndi.distance_transform_edt(Image))

def Label(Image):
    '''
    Labels each connected reigon in Image with a different number
    '''
    Labelled_Image = measure.label(Image, background=0)
    Labelled_Image,a,b = segmentation.relabel_sequential(Labelled_Image, offset=1)
    return(Labelled_Image)

def WaterShed(Image):
    '''
    Performs water shed transform on Image. Useful for segmentation of connected
    reigons.
    '''
    Distance_Transformed_Image = DistanceTransform(Image)
    Local_Maxima = peak_local_max(Distance_Transformed_Image, indices=False, footprint=np.ones((5, 5)),labels=Image)
    markers = measure.label(Local_Maxima)
    Watershed_Image = watershed(-Distance_Transformed_Image, markers, mask=Image)
    return(Watershed_Image)

def Skeletonize(Image):
    '''
    Reduces Image to a skeleton.
    '''
    return(skeletonize(Image))

def SiveArea(Image,smallest=0,largest=1E9):
    '''
    Removes connected reigons smller than smallest and larger than largest in
    Image.
    '''
    Image = np.copy(Image)
    BadReigons = []
    for reigon in measure.regionprops(Image):
        if reigon.area > largest or reigon.area < smallest:
            BadReigons.append(reigon.label)
    for i in BadReigons:
        Image[Image==i] = 0
    Image,a,b = segmentation.relabel_sequential(Image)
    return(Image)

def Fill(Image):
    '''
    Fills in whole in connected reigons in Image.
    '''
    Filled = ndi.binary_fill_holes(Image)
    return(Filled)

def ConstructOverlay(Image1,Image2):
    '''
    Takes 2 images and overlays them for disply.
    '''
    p2, p98 = np.percentile(Image1, (0, 100))
    stretched_Image = exposure.rescale_intensity(Image1, in_range=(p2, p98))
    zeros = np.zeros((np.size(Image1,0),np.size(Image1,0)), dtype = np.uint16)
    overlay = np.dstack((Image1*Image2,Image2,zeros))
    return overlay

def Save(Image,pathname):
    '''
    Saves image to pathname as png.
    '''
    io.imsave(pathname, Image)


def Compare(ImageArray,ColorBarArray=None, TitleArray=None,commonScaleBar=True):
    '''
    Places images side by side for comparsion.
    '''
    if TitleArray == None:
        TitleArray = ["A","B","C","D","E","F","G","H","I","J","K","L"]
    if ColorBarArray == None:
        ColorBarArray = range(1,len(ImageArray)+1)
    dimensionsArray = [(1,1),(1,2),(1,3),(2,2),(2,3),(2,3),(3,3),(3,3),(3,3),(4,3),(4,3),(4,3)]
    rows, columns = dimensionsArray[len(ImageArray)-1]
    i=0
    fig , ax = plt.subplots(nrows=rows,ncols=columns,figsize=(20,10))
    if len(ImageArray)<4:
        ax = np.vstack((ax,ax))
    for y in range(rows):
        for x in range(columns):
            if(i > len(ImageArray)-1):
                break
            if commonScaleBar:
                im4 = ax[y][x].imshow(ImageArray[i], cmap= "jet",aspect='auto', vmin=np.min(ImageArray),vmax=np.max(ImageArray))
            else:
                im4 = ax[y][x].imshow(ImageArray[i], cmap= "jet",aspect='auto')
            ax[y][x].axis('off')
            ax[y][x].set_title(TitleArray[i])
            if i+1 in ColorBarArray:
                divider4 = make_axes_locatable(ax[y][x])
                ax[y][x] = divider4.append_axes("right", size="5%", pad=0.05)
                cbar4 = plt.colorbar(im4, cax=ax[y][x])
            ax[y][x].xaxis.set_visible(False)
            i += 1
    plt.tight_layout()
    plt.show()
    #plt.savefig(FileName + ".png")

def Show():
    '''
    Shows image at run time.
    '''
    plt.show()

def SavePlot(filename):
    '''
    Saves plot to pathname as png.
    '''
    plt.savefig(filename + ".png")

def FolderCompare(FolderName):
    '''
    Compares all images in a folder on screen.
    '''
    filenames = os.listdir(FolderName)
    Images = []
    for file in filenames:
        Images.append(Open(os.path.join(FolderName,file)))
    Compare(Images)
    Show()

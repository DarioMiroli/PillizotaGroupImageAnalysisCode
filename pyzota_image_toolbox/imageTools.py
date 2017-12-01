"""Mix of functions used for image processing and data extraction."""
import numpy as np
from scipy import ndimage as ndi
from scipy import misc
import os
import platform
import matplotlib
if platform.system() == 'Linux':
    matplotlib.use("GTKAgg")
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

#Custom classes
from AnnotateImage import Annotate

def Setup(inputFolder, outputFolder):
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

def CreateFolder(folderName):
    '''
    Crates folder if it doens not already exist.
    '''
    try:
        os.stat(folderName)
    except:
        os.mkdir(folderName)

def GetFileNamesFromFolder(path,fileOnly = True):
    '''
    Retrives all names of files in folder
    '''
    filenames = os.listdir(path+"/")
    if fileOnly:
        return filenames
    else:
        folderAndNames = []
        for f in filenames:
            folderAndNames.append(os.path.join(path,f))
        return folderAndNames

def Open(pathname):
    '''
    Returns numpy array of image at pathname.
    '''
    Image = ndi.imread(pathname)
    Image_Array = np.array(Image)
    return(Image_Array)

def ShowMe(image, cmap=plt.cm.gray):
    '''
    Shows the given image at runtime
    '''
    plt.imshow(image, cmap=cmap)
    if len(np.shape(image)) < 2: plt.colorbar()
    plt.show()

def SelectReigon(image,bgColor='blue',title=""):
    ''' Selects reigon of image with mouse clicks'''
    plt.imshow(image)
    a = Annotate(bgColor)
    plt.title(title)
    plt.show()
    rects = a.getRects()
    del a
    return rects

def Crop(image,rectangle):
    '''Crops image using bounding box '''
    rectangle = np.asarray(rectangle,int)
    x1,x2,y1,y2 = MouseToImageCoords(rectangle)
    croppedImage = image[x1:x2,y1:y2]
    return(croppedImage)

def MouseToImageCoords(rectangle):
    ''' Converts mouse coordinates to image coordinates'''
    if rectangle[2] > rectangle[3]:
        x1 = rectangle[3]
        x2 = rectangle[2]
    else:
        x1 = rectangle[2]
        x2 = rectangle[3]
    if rectangle[0] > rectangle[1]:
        y1 = rectangle[1]
        y2 = rectangle[0]
    else:
        y1 = rectangle[0]
        y2 = rectangle[1]
    return x1,x2,y1,y2

def Blurr(image,sigma=1.0,imageType='RGB'):
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
    Labelled_Image = measure.label(Image, background=0,connectivity =2)
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
    return(skeletonize(Image)*1)

def FitSkeleton(Image,degree=2):
    ''' Returns image with fitted polynomial to given skeleton
    '''
    def computey(x,deg):
        y=0
        for n in range(deg+1):
            y += z[deg-n]*x**n
        return y

    coords = (np.where(Image>0))
    xs = coords[0]
    ys = coords[1]
    zs = z= np.polyfit(xs,ys,degree)
    width , height = Image.shape
    newImage = np.zeros((width,height))
    for x in range(width):
        y = computey(x,degree)
        y=int(y)
        if y>=height: y = height -1
        if y< 0: y = 0
        newImage[x,y] =1

    for y in range(height):
        p = np.poly1d(zs)
        roots = (p - y).roots
        for root in roots:
            if root >=width: root = 0
            if root < 0: root = 0
            #newImage[root,y] =1
    xdata = np.arange(0,width,0.1)
    ydata = [computey(x,degree) for x in xdata]
    ydata = np.clip(ydata,0,height)
    #ydata = width - ydata
    #xdata = width-xdata
    return newImage,xdata,ydata

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

def Centering(thresh_im):
    '''Finds parts of the object definitely associated with the individual image'''
    '''Useful for segmentation'''
    print('Removing Noise')
    kernel=np.ones((1,1),np.uint8)
    opening = cv2.morphologyEx(thresh_im,cv2.MORPH_OPEN,kernel, iterations=2)

    print('Finding background')
    #sure background area
    sure_bg=cv2.dilate(opening,kernel,iterations=1)
    print('Finding foreground')
    #finding sure foreground area
    dist_transform=cv2.distanceTransform(opening,cv2.cv.CV_DIST_L2,0)
    ret, sure_fg=cv2.threshold(dist_transform,0.99*dist_transform.max(),255,0)

    print('Obtaining unknown region...')
    #finding unknow region
    sure_fg = np.uint8(sure_fg)
    unknown=cv2.subtract(sure_bg,sure_fg)

    #Marker labelling
    print('Labelling Markers...')
    ret, markers = cv2.cv.ConnectedComponents(sure_fg)
    #Add one to all labels so that sure background is not 0 but 1
    markers=markers+1

    #Mark the unknown regions with 0
    markers[unknown==255] = 0

    markers=cv2.watershed(thresh_im,markers)
    thresh_im[markers == -1] = [255,0,0]
    return(thresh_im)

def Edges(img,numberoferosions):
    '''Find the edge of an object'''
    print('Finding outlines...')
    eroded=Erode(img,2)
    edges=img ^ eroded
    return(edges)

def skeleton_endpoints(skel):
    '''Find the endpoints of your skeleton'''
    # make out input nice, possibly necessary
    skel = skel.copy()
    skel[skel!=0]=1
    skel=np.uint8(skel)

    #apply the conversion
    kernel = np.uint8([[1,1,1],[1,10,1],[1,1,1]])
    src_depth= -1
    filtered = cv2.filter2D(skel,src_depth,kernel)

    #now look through to find the value of 11
    #this returns a mask of the endpoints,
    #but if you just want the coordinates you could simply
    #return np.where(filtered==11)
    out = np.zeros_like(skel)
    out[np.where(filtered==11)]=1
    return np.where(filtered==11)

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
    misc.imsave(pathname, Image)


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

def AniShow(images,delay=1,cmap='gray',title=""):
    plt.ion()
    for image in images:
        plt.clf()
        plt.title(title)
        plt.imshow(image,interpolation='none',cmap=cmap)
        plt.colorbar()
        plt.pause(delay)
    plt.close()
    plt.ioff()

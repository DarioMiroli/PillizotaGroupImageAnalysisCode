from __future__ import generators
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy import spatial as spatial

from skimage import filters
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage.morphology import *
from skimage import measure
from skimage import segmentation
from skimage import io
from skimage import exposure
from skimage.feature import peak_local_max
from PIL import Image
import tifffile as tif
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import cv2
import os
from matplotlib.widgets import  RectangleSelector
from pylab import *
from itertools import combinations


''' FUNCTIONS'''
'''All functions required to run the program'''
def Setup(InputFolder, OutputFolder):
    '''Create output folder if it doesn't exist'''
    try:
        os.stat(OutputFolder)
    except:
        os.mkdir(OutputFolder)
    filenames=os.listdir(InputFolder+"/")
    return filenames

def Open(pathname):
    ''' Returns numpy array of image at pathname'''
    Image=plt.imread(pathname)
    Image_Array=np.array(Image,dtype=np.unit16)
    return(Image_Array)

def Opentiff(pathname):
    '''Returns numpy array of tiff image at pathname'''
    im=Image.open(pathname)
    imarray=np.array(im)
    return(imarray)

def Blur(Image,sigma):
    '''Gaussian Blur with standard deviation of sigma on Image'''
    Blurred_Image=gaussian_filter(Image,sigma=sigma)
    return(Blurred_Image)

def Threshold(Image,blocksize):
    '''Adaptive threshold of Image.
    Takes average of pixels in small region given by blocksize'''
    Threshold_Image=threshold_adaptive(Image,blocksize)
    return(Threshold_Image.astype(float))

def Global_Threshold(Image):
    '''Global threshold of entire image with single value'''
    thresh=threshold_otsu(Image)
    Image=Image>thresh
    return(Image.astype(float))

def Thresh_Low_values(Image,threshold):
    '''Low variable thresholding
    Sets all pixels that are below threshold to zero and those above to 1 '''
    low_values_indices = Image < threshold  #where low values are low
    #print(np.amax(Image))
    Image[low_values_indices]=0 #set low values to 0
    high_values_indices = Image > threshold
    Image[high_values_indices]=1
    #print(np.amax(Image))
    #print (imarray)
    #img=Image.fromarray(imarray)
    return(Image)

def Erode(Image,numberoftimes):
    '''Binary erosion of image
    removes the outermost pixels from Image, iterated over numberoftimes'''
    for i in range(numberoftimes):
        Image=binary_erosion(Image)
    return(Image)

def Dilate(Image,numberoftimes):
    '''Binary dilation of image, iterated over numberoftimes'''
    for i in range(numberoftimes):
        Image=binary_dilation(Image)
    return(Image)

def DistanceTransform(Image):
    '''Distance transform of image'''
    return(ndi.distance_transform_edt(Image))

def Label(Image):
    '''Label connected regions of image with different number
    each individual segment is given a unique number'''
    Labelled_Image=measure.label(Image, background=0)
    Labelled_Image,a,b = segmentation.relabel_sequential(Labelled_Image, offset=1)
    return(Labelled_Image)

def WaterShed(Image,one):
    '''Watershed of image to allow segmentation of connected regions'''
    Distance_Transformed_Image=DistanceTransform(Image)
    Local_Maxima=peak_local_max(Distance_Transformed_Image,indices=False, footprint=np.ones((one,one)),labels=Image)
    markers=measure.label(Local_Maxima)
    Watershed_Image=watershed(-Distance_Transformed_Image,markers,mask=Image)
    return(Watershed_Image)

def Skeletonise(Image):
    '''Reduce image to skeleton'''
    return(skeletonize_3d(Image))

def SieveArea(Image,smallest=0,largest=1E9):
    '''remove connected areas that are smaller than smallest or larger than largest'''
    Image=np.copy(Image)
    BadRegions=[]
    for region in measure.regionprops(Image):
        if region.area > largest or region.area < smallest:
            BadRegions.append(region.label)
    for i in BadRegions:
        Image[Image==i]=0
    Image,a,b = segmentation.relabel_sequential(Image)
    return(Image)

def Fill(Image):
    '''Fill in connected regions within an image'''
    Filled = ndi.binary_fill_holes(Image)
    return(Filled)


def ConstructOverlay(Image1,Image2):
    '''Overlay two images to display'''
    p2,p98=np.percentile(Image1,(0,100))
    stretched_Image = exposure.rescale_intensity(Image1, in_range=(p2,p98))
    zeros=np.zeros((np.size(Image1,0)),dtype=np.uint16)
    overlay=np.dstack((Image1*Image2,Image2,zeros))
    return(overlay)

def Compare(ImageArray,ColorBarArray=None,TitleArray=None):
    '''Place images side-by-side for comparison'''
    if TitleArray == None:
        TitleArray=["I","II","III","IV","V","VI","VII","VIII","IX","X"]
    if ColorBarArray == None:
        ColorBarArray= range(1,len(ImageArray+1))
    dimensionsArray=[(1,1),(1,2),(1,3),(2,2),(2,3),(3,3),(3,3),(3,3)(3,3),(4,3),(4,3),(4,3)]
    rows,columns=dimensionsArray[len(ImageArray)-1]
    i=0
    fig, ax = plt.subplots(nrows=rows,ncols=columns,figsize=(20,10))
    if len(ImageArray)<4:
        ax=np.vstack((ax,ax))
    for y in range(rows):
        for x in range(columns):
            if (i>len(ImageArray)-1):
                break
            im4 = ax[y][x].imshow(ImageArray[i],cmap="jet",aspect='auto')
            ax[y][x].axis('off')
            ax[y][x].set_title(TitleArray[i])
            if i+1 in ColorBarArray:
                divider4=make_axes_locatable(ax[y][x])
                ax[y][x]=divider4.append_axes("right",size="5%",pad=0.05)
                cbar4=plt.colorbar(in4,cax=ax[y][x])
            ax[y][x].xaxis.set_visible(False)
            i +=1
    plt.tight_layout()

def Save(Image,pathname):
    '''Save image'''
    io.imsave(pathname,Image)

def SavePng(filename):
    '''Save image to png'''
    plt.savefig(filename + ".png")

def Show():
    '''Show image at runtime'''
    plt.show()

def FolderCompare(FolderName):
    '''Compare all images in two different folders'''
    filenames=os.listdir(FolderName)
    Images=[]
    for file in filenames:
        Images.append(Open(os.path.join(FolderName,file)))
    Compare(Images)
    Show()

def centering(thresh_im):
    '''Reduce connected regions to area that is certain to be part of that region
    Input: thresh_im: Thresholded image
    Output: thresh_im: new thresholded image with clearly segmented regions only'''
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
    '''Find the edge of an object
    Input:  img                 thresholded image
            numberoferosions    number of times to erode the image
    Output: edges               outline of the thresholded image which is
    numberoferosions in width
    Takes the input image and erodes it numberoferosions before
    subtracting the eroded image from the original and returning it'''
    print('Finding outlines...')
    eroded=Erode(img,numberoferosions)
    edges=img ^ eroded
    return(edges)

def skeleton_endpoints(skel):
    '''Find the endpoints of your skeleton
    takes as input a numpy array of the skeleton (thresholded)
    Marks the endpoints of the skeleton with 11 rather than 1
    Returns the array with only the filtered endpoints '''
    # make input nice, possibly necessary
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

def selectRegion(image):
    ''' Selects region of image with mouse clicks
    Dario function to select a single region of the image and return the coordinates
    based upon the selection'''
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
    fig = figure()
    ax = subplot(111)
    ax.imshow(image)
    toggle_selector.RS = RectangleSelector(ax, onselect, drawtype='box')
    connect('key_press_event', toggle_selector)
    plt.show()
    #print(mouseCoords)
    return(mouseCoords)

def crop(image,rectangle):
    '''Crops image using bounding box '''
    rectangle = np.asarray(rectangle,int)
    rectangle=order(rectangle)
    croppedImage = image[rectangle[2]:rectangle[3],rectangle[0]:rectangle[1]]
    return(croppedImage)

def order(coord):
    '''Orders coordinates correctly for cropping etc'''
    if coord[2]>coord[3]:
            temp=coord[2]
            coord[2]=coord[3]
            coord[3]=temp
    if coord[0]>coord[1]:
            temp=coord[0]
            coord[0]=coord[1]
            coord[1]=temp
    return(coord)

def square_distance(x,y):
    '''Get the squared distance between two points'''
    return sum([(xi, yi)**2 for xi, yi in zip(x,y)])

def max_square_dist(img):
    '''Calculate maximum distance between points in array'''
    max_square_distance=0
    for pair in combinations(img,2):
        if square_distance(*pair)>max_square_distance:
            max_square_distance=square_distance(*pair)
            max_pair=pair
    return max_pair

def max_dist(img):
    '''Find coordinates of points in array that are furthest apart'''
    D=spatial.distance.pdist(img)
    D=spatial.distance.squareform(D);
    temp=np.where(D==D.max())
    #print('temp',temp)
    temp_b=zip(temp[0],temp[1])
    #print('temp_b',temp_b)
    #N, [I_row,I_col]=np.nanmax(D),np.unravel_index(np.argmax(D),D.shape)
    #print('Dshape', D.shape)
    #print('argmax', np.argmax(D))
    #print(np.unravel_index(np.argmax(D),D.shape))
    return(temp_b) #N,[I_row,I_col])

def orientation(p,q,r):
    '''Return positive if p-q-r are clockwise, neg if ccw, zero if colinear.
    Required for rotatingCalipers'''
    return (q[1]-p[1])*(r[0]-p[0]) - (q[0]-p[0])*(r[1]-p[1])

def hulls(Points):
    '''Graham scan to find upper and lower convex hulls of a set of 2d points.
    Required for rotatingCalipers'''
    U = []
    L = []
    for p in Points:
        while len(U) > 1 and orientation(U[-2],U[-1],p) <= 0: U.pop()
        while len(L) > 1 and orientation(L[-2],L[-1],p) >= 0: L.pop()
        U.append(p)
        L.append(p)
    return U,L

def rotatingCalipers(Points):
    '''Given a list of 2d points, finds all way of sandwiching the points between
    two parallel lines that touch one point each, and yields the sequence of
    pairs of points touched by each pair of lines.
    Taken from code.activestate.com/recipes/117225-convex-hull-and-diameter-of-2d-point-sets'''
    U,L = hulls(Points)
    i = 0
    j = len(L) - 1
    while i < len(U) - 1 or j > 0:
        yield U[i],L[j]

        # if all the way through one side of hull, advance the other side
        if i == len(U) - 1: j -= 1
        elif j == 0: i += 1

        # still points left on both lists, compare slopes of next hull edges
        # being careful to avoid divide-by-zero in slope calculation
        elif (U[i+1][1]-U[i][1])*(L[j][0]-L[j-1][0]) > \
                (L[j][1]-L[j-1][1])*(U[i+1][0]-U[i][0]):
            i += 1
        else: j -= 1


def diameter(Points):
    '''Given a list of 2d points, returns the pair that's farthest apart.
    Taken from code.activestate.com/recipes/117225-convex-hull-and-diameter-of-2d-point-sets
    Calls rotatingCalipers to calculate the maximum distance between two points in an array
    Input:  Points  Numpy array consisting of the edge information of a region.
                    (Should also work for a region but will take far longer to compute as
                    iterates across all points within the array)
    Output: diameter The maximum length between two points within the array
            pair     Coordinates of the points that the maximum length is measured from'''
    diam,pair = max([((p[0]-q[0])**2 + (p[1]-q[1])**2, (p,q))
                     for p,q in rotatingCalipers(Points)])
    return np.sqrt(diam),pair

def getcoords(img,value):
    '''Returns the coordinates of the points within a numpy array that are non-zero'''
    return np.argwhere(img==value)

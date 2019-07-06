"""Mix of functions used for image processing and data extraction."""
import numpy as np
from scipy import ndimage as ndi
from scipy import misc
import scipy.fftpack
import os
import platform
import pickle
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import matplotlib.cm as cm

from skimage.filters import *
from skimage.morphology import *
from skimage import measure
from skimage import segmentation
from skimage import io
from skimage import exposure
from skimage.measure import *
from skimage.feature import peak_local_max

from matplotlib.widgets import  RectangleSelector
from pylab import *

import tifffile as tif

#Custom classes
from AnnotateImage import Annotate

def Setup(inputFolder, outputFolder):
    '''
    Creates output folder if it doesnt exist and get input file names from input
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

def StripImageFiles(fileNames, OKExtensions =["tiff","tif"]):
    '''
    Removes files with the wrong extension from a list of files
    '''
    acceptedFiles = []
    for f in fileNames:
        if f.split(".")[-1] in OKExtensions:
            acceptedFiles.append(f)
    return acceptedFiles

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
    plt.imshow(image, cmap=cmap,interpolation='none')
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

def CompareAnnotate(images,bgColor='blue',ColorBarArray=None, TitleArray=None,
        commonScaleBar=True, axX=1,axY=1,mode='Recs',data=None):
    '''Allows annotation of multiple compared images'''
    ax = Compare(images,ColorBarArray=ColorBarArray,TitleArray=TitleArray,
            commonScaleBar=commonScaleBar, show=False)
    a = Annotate(bgColor,ax=ax[axX][axY],mode='Rmv',data=data)
    plt.title(title)
    plt.show()
    finalData = a.getData()
    del a
    return finalData

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

def GlobalThreshold(Image,threshold=None):
    '''
    Simple threshold of image using single threshold value over whole image.
    '''
    if threshold == None:
        threshold = threshold_otsu(Image)
    Image = Image > threshold
    return(Image*1)

def Erode(Image,numberoftimes):
    '''
    Performs binary erosion of image a required number of times.
    '''
    for i in range(numberoftimes):
        Image = scipy.ndimage.morphology.binary_erosion(Image,border_value=0)
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

def ClearBorders(Image):
    '''
    Removes objects connected to image edges
    '''
    labeled = Image - np.amin(Image)#measure.label(Image,connectivity =2)
    return segmentation.clear_border(labeled)+np.amin(Image)

def BboxImages(Image,mask):
    '''
    Returns a vounding box image for each label in an image
    '''
    bBoxedImages = []
    bBoxedMasks = []
    l = Label(mask)
    props = measure.regionprops(l)
    print("props",len(props))
    for prop in props:
        x1,y1,x2,y2 = prop.bbox
        bBoxedImages.append(Image[x1-10:x2+10,y1-10:y2+10])
        bBoxedMasks.append(mask[x1-10:x2+2,y1-10:y2+10])
    return bBoxedImages, bBoxedMasks

def GetLengthAndWidth(Image,rawImage,debug=False):
    #Tidy up iamge for analysis. Clear it flip it if necessary
    cleared = ClearBorders(Image)
    cleared = cleared-np.amin(cleared)
    props = measure.regionprops(cleared)
    x1,y1,x2,y2 = props[0].bbox
    box = cleared[x1:x2,y1:y2]
    width , height = box.shape
    if width > height:
        box = np.swapaxes(box,0,1)
        width , height = box.shape
        rawImage = np.swapaxes(rawImage,0,1)
        Image = np.swapaxes(Image,0,1)


    #Compute all points on top and bottom of cell
    topPoints= []
    bottomPoints = []
    for y in range(height):
        topIntersection = None
        bottomIntersection = None
        for x in range(width):
            if box[x][y] > 0 and topIntersection == None:
                topIntersection = x
            if box[width-1-x][y] > 0 and bottomIntersection == None:
                bottomIntersection = width-1-x
        topPoints.append([topIntersection,y])
        bottomPoints.append([bottomIntersection,y])

    #Get edge points in order
    normBox = ((box-np.amin(box))/np.amax(box))
    edgeImage =  np.pad(normBox - Erode(normBox,1),0,'constant')
    xedgePixels, yedgePixels = np.where(edgeImage>0)

    #Order Edge Points
    orderedEdgePoints = []
    ys = list(np.arange(height)) + list(np.arange(height,0,-1))
    for delta in [1,-1]:
        for y in range(height):
            tempPoints = []
            on = False
            off = False
            if np.sum(edgeImage[:,y]) >2:
                for x in range(width):
                    if edgeImage[delta*x][y] > 0:
                        on = True
                    if edgeImage[delta*x][y] == 0 and on:
                        off = True
                    if on and not off and edgeImage[delta*x][y]>0:
                        if delta == 1:
                            tempPoints.append([x,y])
                        if delta!=1 and y!=0:
                            tempPoints.append([width-x,y])
                if y>0:
                    if tempPoints[0][1] < orderedEdgePoints[-1][0]:
                        tempPoints = tempPoints[::-1]
                else:
                    tempPoints = tempPoints[::-1]
                orderedEdgePoints = orderedEdgePoints+ tempPoints
            else:
                for x in range(width):
                    if edgeImage[delta*x][y] > 0:
                        if delta == 1:
                            orderedEdgePoints = orderedEdgePoints + [[x,y]]
                            break
                        else:
                            if x != 0:
                                orderedEdgePoints = orderedEdgePoints + [[width-x,y]]
                                break
        orderedEdgePoints= orderedEdgePoints[::-1]

    #Remove duplicates
    newOEPoints = []
    for i in range(len(orderedEdgePoints)):
        if not (orderedEdgePoints[i] in orderedEdgePoints[0:i]):
            newOEPoints.append(orderedEdgePoints[i])
    orderedEdgePoints = newOEPoints
    orderedXs = [orderedEdgePoints[i][0] for i in range(len(orderedEdgePoints))]
    orderedYs = [orderedEdgePoints[i][1] for i in range(len(orderedEdgePoints))]

    #Find mid index
    midPoint = np.argmin(np.abs(np.asarray(orderedYs)-height/2.0))

    #Run around cell from mid index calcing curvature
    curveLength = 10
    curvatures = []
    for i in range(midPoint,len(orderedEdgePoints)+midPoint):
        x1 = orderedXs[(i-curveLength)%len(orderedEdgePoints)]
        x2 = orderedXs[(i+curveLength)%len(orderedEdgePoints)]
        y1 = orderedYs[(i-curveLength)%len(orderedEdgePoints)   ]
        y2 = orderedYs[(i+curveLength)%len(orderedEdgePoints)]
        dist = ( (x2-x1)**2 + (y2-y1)**2 )**0.5
        curvatures.append(dist)

    #Mark polls
    leftData = curvatures[0:len(curvatures)//2]
    leftPole = np.argmin(leftData)
    rightData = curvatures[len(curvatures)//2:]
    rightPole = np.argmin(rightData)

    #Smooth with fft
    w = scipy.fftpack.rfft(curvatures)
    spectrum = w**2
    cutoff_idx = spectrum < (spectrum.max()/50000)
    w2 = w.copy()
    w2[cutoff_idx] = 0
    y2 = scipy.fftpack.irfft(w2)
    leftPole2 = np.argmin(y2[0:len(curvatures)//2])
    rightPole2 = np.argmin(y2[len(curvatures)//2:])

    #Create skeleton with poles added
    BackBone = Skeletonize(normBox)
    BackBone[orderedXs[leftPole2+midPoint] , orderedYs[leftPole2+midPoint]] = 1
    BackBone[orderedXs[(rightPole2+midPoint+len(curvatures)//2)%len(orderedXs)],
            orderedYs[(rightPole2+midPoint+len(curvatures)//2)%len(orderedYs)]]=1
    newImage,ydata,xdata,zs= FitSkeleton(np.transpose(BackBone),2)


    #Find points of intersection between line and segmented reigon
    P1 = -1
    P2 = -1
    for x in range(height):
        y = int(round(ComputePoly(x,zs)))
        if y >= width:
            y = width -1
        value = normBox[y,x]
        if P1 == -1 and value == 1:
            P1 = [y,x]
        if value ==1:
            P2 = [y,x]
    length = 0
    for x in range(P1[1],P2[1]):
        length += ( (x+1 - x)**2 + ( ComputePoly(x+1,zs)-
                ComputePoly(x,zs) )**2)**0.5

    #compute mid points to get width
    midx = (P1[1]+P2[1])/2
    midXs = [i for i in range(midx-10,midx+10)]
    midYs = [ComputePoly(i,zs) for i in midXs]
    #plt.plot(midXs,midYs,'*')
    P1s = []
    P2s = []
    for x,y in zip(midXs,midYs):
        #Calculate 2 points on tangent line
        m1 = 2*zs[0]*x + zs[1]
        c1 =y - x*(2*zs[0]*x + zs[1])
        Point1 = [x,y]

        #Transfrom P2 coords to tangent Point
        theta = np.pi/2
        Point2 = [0-x,c1-y]
        #Rotate
        newX = Point2[0]*np.cos(theta) - Point2[1]*np.sin(theta)
        newY = Point2[0]*np.sin(theta) + Point2[1]*np.cos(theta)
        newPoint2 = [newX+x,newY+y]
        m2 = (newPoint2[1] - Point1[1])/(newPoint2[0]-Point1[0])
        c2 = newPoint2[1] - newPoint2[0]*(m2)
        P1 = -1
        P2 = -1
        widths = []

        for runy in range(width):
            runx = (runy-c2)/m2
            if np.isnan(runx):
                runx = midYs[runy]
            if round(runx) >= height or runx < 0:
                runx = height-1
                value = -1
            else:
                value = normBox[runy,int(round(runx))]
            if P1 ==-1 and value>0:
                P1 = [runx,runy]
            if value>0:
                P2 = [runx,runy]
        width1 = ( (P2[0]-P1[0])**2 + (P2[1]-P1[1])**2 )**0.5
        widths.append(width1)
        P1s.append(P1)
        P2s.append(P2)

    meanWidth = np.mean(widths)
    if debug:
        #Display whole process for thesis
        fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(18,6))
        #Work out how to pad for plotting length and width lines and poles
        padRow = 0
        for row in range((np.shape(Image)[0])):
            if np.sum(Image,axis=0)[row] > np.shape(Image)[0]:
                padRow = row
                break
            else:
                padRow = row
        padCol = 0
        for col in range((np.shape(Image)[1])):
            if np.sum(Image,axis=1)[col] > np.shape(Image)[1]:
                padCol = col
                break
            else:
                padCol = col
        ax[0].set_title("Raw / segmented ",fontsize=30,fontweight="bold")
        ax[0].imshow(rawImage,cmap="gray")
        ax[0].imshow(Image,cmap="jet",alpha=0.25)
        ax[0].xaxis.set_ticklabels([])
        ax[0].yaxis.set_ticklabels([])
        ax[1].set_title("Length: {0:.1f} px".format(length),fontsize=30,fontweight="bold")
        ax[1].plot(ydata+padRow,xdata+padCol)
        ax[1].plot(orderedYs[midPoint+leftPole2]+padRow,orderedXs[midPoint+leftPole2]+padCol,'*',markersize=10)
        ax[1].plot(orderedYs[(rightPole2+midPoint+len(curvatures)//2)%len(orderedXs)]+padRow,orderedXs[(rightPole2+midPoint+len(curvatures)//2)%len(orderedYs)]+padCol,'*',markersize=10)
        ax[1].imshow(Image,interpolation='None',cmap="gray")
        ax[1].xaxis.set_ticklabels([])
        ax[1].yaxis.set_ticklabels([])


        colors = ["green","navy"]
        for i in range(len(P1s)):
            xs = [   P1s[i][0],P2s[i][0]   ]
            ys = [   P1s[i][1],P2s[i][1]   ]
            if i%2 == 0:
                color = colors[0]
            else:
                color = colors[1]
            ax[2].plot(np.asarray(xs)+padRow,np.asarray(ys)+padCol,color=color    )
        ax[2].imshow(Image,interpolation='None',cmap="gray")
        ax[2].set_title("Mean width: {0:.1f} px".format(meanWidth),fontsize=30,fontweight="bold")
        ax[2].xaxis.set_ticklabels([])
        ax[2].yaxis.set_ticklabels([])
        fig.tight_layout()
        try:
            GetLengthAndWidth.count = GetLengthAndWidth.count +1
        except:
            GetLengthAndWidth.count = 0

        #plt.savefig("./ThesisGraphs/ExampleCells/Example{0}.png".format(GetLengthAndWidth.count))
        #plt.show()
        plt.clf()
        plt.close()


        #**************** ALL steps PLOT  *****************************************#

        #Example showing all stages plot
        fig, ax = plt.subplots(nrows=3,ncols=3,figsize=(12,12))
        #Set titles
        fontSize = 30
        ax[0][0].set_title("(A) Raw",fontsize=fontSize,fontweight="bold")
        ax[0][1].set_title("(B) Segmented",fontsize=fontSize,fontweight="bold")
        ax[0][2].set_title("(C) Edge points",fontsize=fontSize,fontweight="bold")
        ax[1][0].set_title("(D) Curvature",fontsize=fontSize,fontweight="bold")
        ax[1][1].set_title("(E) Curvature trace",fontsize=fontSize,fontweight="bold")
        ax[1][2].set_title("(F) Poles",fontsize=fontSize,fontweight="bold")
        ax[2][0].set_title("(G) Backbone",fontsize=fontSize,fontweight="bold")
        ax[2][1].set_title("(H) Length",fontsize=fontSize,fontweight="bold")
        ax[2][2].set_title("(I) Width",fontsize=fontSize,fontweight="bold")



        #Raw iamge
        newRaw = rawImage[0:np.shape(Image)[0],0:np.shape(Image)[1]]
        ax[0][0].imshow(newRaw,cmap="gray",interpolation="None")
        #Segmented
        ax[0][1].imshow(Image,cmap="gray",interpolation="None")
        #Ordered x and y points
        ax[0][2].imshow(Image,cmap="gray",interpolation="None")
        edgePixels =  np.zeros_like(Image)
        for i in range(len(orderedXs)):
            edgePixels[orderedXs[i]+padRow][orderedYs[i]+padCol] = (i +20)

        ax[0][2].imshow(edgePixels,cmap="inferno",interpolation="None")

        #Curvature measuring stick
        for i in (20,50,80,0):
            xs = orderedXs[i:i+10]
            ys = orderedYs[i:i+10]
            ax[1][0].plot(np.asarray(ys)+padCol,np.asarray(xs)+padRow,'-',markersize=10,linewidth=5)
        ax[1][0].imshow(edgePixels,cmap="inferno",interpolation="None")
        #curvature plot
        colors = cm.inferno(np.linspace(0.2, 1, len(curvatures)))
        ax[1][1].scatter([i for i in range(len(curvatures))],curvatures,label="Raw",color=colors)
        ax[1][1].plot(y2,linewidth=3,label="Smoothed")
        ax[1][1].set_xlabel("Edge index",fontsize=18,fontweight="bold")
        ax[1][1].set_ylabel("Segment length(px)",fontsize=18,fontweight="bold")
        ax[1][1].legend(loc="upper center",ncol=2)
        ax[1][1].legend(bbox_to_anchor=(0., -0.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)
        ax[1][1].patch.set_facecolor('black')
        asp = (np.diff(ax[1][1].get_xlim())[0]) / (np.diff(ax[1][1].get_ylim())[0])
        asp /= np.abs(np.diff(ax[0][0].get_xlim())[0] / np.diff(ax[0][0].get_ylim())[0])
        ax[1][1].set_aspect(6.5)
        ax[1][1].set_xlim(0,len(curvatures))
        #Mark poles
        ax[1][2].plot(orderedYs[midPoint+leftPole2]+padRow,orderedXs[midPoint+leftPole2]+padCol,'o--',markersize=10)
        ax[1][2].plot(orderedYs[(rightPole2+midPoint+len(curvatures)//2)%len(orderedXs)]+padRow,orderedXs[(rightPole2+midPoint+len(curvatures)//2)%len(orderedXs)]+padCol,'o--',markersize=10)
        ax[1][2].imshow(edgePixels,cmap="inferno",interpolation="None")
        #Show backbone
        newBack = np.zeros_like(Image)
        for row in range(np.shape(BackBone)[0]):
            for col in range(np.shape(BackBone)[1]):
                if BackBone[row][col] > 0:
                    newBack[row+padRow][col+padCol] = 1

        #ax[2][0].plot(orderedYs[midPoint+leftPole2]+padRow,orderedXs[midPoint+leftPole2]+padCol,'o--',markersize=10)
        #ax[2][0].plot(orderedYs[(rightPole2+midPoint+len(curvatures)//2)%len(orderedXs)]+padRow,orderedXs[(rightPole2+midPoint+len(curvatures)//2)%len(orderedXs)]+padCol,'o--',markersize=10)
        ax[2][0].imshow(newBack,cmap="gray",interpolation="None",alpha= 0.9)
        ax[2][0].imshow(edgePixels,cmap="inferno",interpolation="None",alpha= 0.1)

        #Show length
        ax[2][1].plot(orderedYs[midPoint+leftPole2]+padRow,orderedXs[midPoint+leftPole2]+padCol,'o--',markersize=10)
        ax[2][1].plot(orderedYs[(rightPole2+midPoint+len(curvatures)//2)%len(orderedXs)]+padRow,orderedXs[(rightPole2+midPoint+len(curvatures)//2)%len(orderedXs)]+padCol,'o--',markersize=10)
        ax[2][1].plot(ydata+padRow,xdata+padCol)
        ax[2][1].imshow(Image,cmap="gray",interpolation="None",alpha= 0.5)
        ax[2][1].imshow(edgePixels,cmap="inferno",interpolation="None",alpha= 0.5)
        #Show width
        ax[2][2].plot(orderedYs[midPoint+leftPole2]+padRow,orderedXs[midPoint+leftPole2]+padCol,'o--',markersize=10)
        ax[2][2].plot(orderedYs[(rightPole2+midPoint+len(curvatures)//2)%len(orderedXs)]+padRow,orderedXs[(rightPole2+midPoint+len(curvatures)//2)%len(orderedXs)]+padCol,'o--',markersize=10)
        ax[2][2].plot(ydata+padRow,xdata+padCol)

        colors = ["green","navy"]
        for i in range(len(P1s)):
            xs = [   P1s[i][0],P2s[i][0]   ]
            ys = [   P1s[i][1],P2s[i][1]   ]
            if i%2 == 0:
                color = colors[0]
            else:
                color = colors[1]
            ax[2][2].plot(np.asarray(xs)+padRow,np.asarray(ys)+padCol,color=color )
        ax[2][2].imshow(Image,cmap="gray",interpolation="None",alpha= 0.5)
        ax[2][2].imshow(edgePixels,cmap="inferno",interpolation="None",alpha= 0.5)
        plt.show()
        fig.tight_layout()
        plt.clf()
        plt.close()
        #exit()


    return length,meanWidth

    #plt.show()
def getMeanIntensity(mask,rawImage):
    mask = ClearBorders(mask)
    mask = mask-np.amin(mask)
    intensity = []
    #plt.imshow(rawImage,cmap="inferno",alpha =0.5)
    #plt.imshow(mask,cmap="gray",alpha=0.3)
    #plt.show()
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if (mask[row][col]) != 0:
                intensity.append(rawImage[row][col])
    return np.mean(intensity)
    return meanIntensity

def GetArea(Image):
    '''
    Get area from binary mask
    '''
    props = measure.regionprops(Image)
    area = None
    for prop in props:
        if prop.label !=1:
            area = prop.area
    return area

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
    Image = (Image - np.amin(Image))/np.amax(Image)
    return(skeletonize(Image)*1)

def ComputePoly(x,zs):
    '''
    Computes y value from given x value and polynomial coefficients
    '''
    deg = len(zs)-1
    y=0
    for n in range(deg+1):
        y += zs[deg-n]*x**n
    return y

def FitSkeleton(Image,degree=2):
    ''' Returns image with fitted polynomial to given skeleton
    '''
    coords = (np.where(Image>0))
    xs = coords[0]
    ys = coords[1]
    zs = np.polyfit(xs,ys,degree)
    width , height = Image.shape
    newImage = np.zeros((width,height))
    for x in range(width):
        y = ComputePoly(x,zs)
        y=int(y)
        if y>=height: y = height -1
        if y< 0: y = 0
        newImage[x,y] =1

    xdata = np.arange(0,width,0.1)
    ydata = [ComputePoly(x,zs) for x in xdata]
    ydata = np.clip(ydata,0,height)
    return newImage,xdata,ydata,zs

def getCellLength(Image,zs):
    if len(zs) == 2:
        started = False
        width,height = Image.shape
        startPoint = -1
        endPoint =-1
        for x in range(width):
            y = ComputePoly(x,zs)
            if y>=height: y = height -1
            if y< 0: y = 0
            if Image[x,y] > 0 and not started:
                startPoint = [x,y]
                started = True
            if Image[x,y] > 0 and started:
                endPoint = [x,y]
        if startPoint != -1 and endPoint != -1:
            cellLength = ((1+zs[0]**2)**0.5)*(endPoint[0]-startPoint[0])
        else:
            cellLength = -1
    else:
        print("Not polynomial fit")
        exit()
    return cellLength

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


def Compare(ImageArray,ColorBarArray=None, TitleArray=None,
        commonScaleBar=True,show=True):
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
                im4 = ax[y][x].imshow(ImageArray[i], cmap= "jet",interpolation='None', vmin=np.min(ImageArray),vmax=np.max(ImageArray))
            else:
                im4 = ax[y][x].imshow(ImageArray[i], cmap= "jet",interpolation='None')
            #ax[y][x].axis('off')
            ax[y][x].set_title(TitleArray[i])
            if i+1 in ColorBarArray:
                divider4 = make_axes_locatable(ax[y][x])
                ax[y][x] = divider4.append_axes("right", size="5%", pad=0.05)
                cbar4 = plt.colorbar(im4, cax=ax[y][x])
            #ax[y][x].xaxis.set_visible(False)
            i += 1
    plt.tight_layout()
    if show:
        plt.show()
    return ax

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

def SaveCellToFile(cellImage,binaryImage,parentFolder,fileName,path):
    '''
    Saves cell image to pickle file
    '''
    dictionary = {"ParentFolder":parentFolder,"FileName":fileName,
            "RawImage":cellImage, "BinaryMask":binaryImage}
    f = open(path,'a')
    pickle.dump(dictionary, f)
    f.close()

def LoadCellFromFile(path):
    f = open(path,'r')
    dictionary = pickle.load(f)
    return dictionary

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

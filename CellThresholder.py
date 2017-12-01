from pyzota_image_toolbox import imageTools as IT
import numpy as np
import sys
import matplotlib.pyplot as plt

#Machine learning stuff
sys.path.insert(0, '/home/s1033855/OperationLeadPelican/MachineLearningTest/KerasTests/ImageSegmentationWithKeras/MachineSegmenter')
from MachineSegmenter import MachineSegmenter
inputFolder = "./Analysis/Slide1_Vetted"
thresholdFolder = "./Analysis/Slide1_Threshold"
IT.CreateFolder(thresholdFolder)
M1 = MachineSegmenter()
M1.loadModel("/home/s1033855/OperationLeadPelican/MachineLearningTest/KerasTests/ImageSegmentationWithKeras/MachineSegmenter/Files/Models/TempModels/TempModel1.h5")
M1.compileModel()

#File IO
fileNamesPlusFolder = IT.GetFileNamesFromFolder(inputFolder,fileOnly=False)
fileNamesPlusFolder.sort()
fileNamesOnly = IT.GetFileNamesFromFolder(inputFolder,fileOnly=True)
fileNamesOnly.sort()

#Analysis
plt.ion()
for i,im in enumerate(fileNamesPlusFolder):
    im = IT.Open(im)
    machinePrediction = M1.predict([im])
    thresholdPrediciton = machinePrediction[0] > 0.5
    distanceTransform = IT.DistanceTransform(thresholdPrediciton)
    thresholdDT = distanceTransform > 0.5*np.amax(distanceTransform)
    skeleton = IT.Skeletonize(thresholdPrediciton)
    skeletonLabels = IT.Label(skeleton)
    fits = []
    xdatas = []
    ydatas   =[]
    for skel in range(-1,np.amax(skeletonLabels)+1):
        singleSkel = 1*(skeletonLabels == skel)
        if skel!= -1:
            fit,xdata,ydata = IT.FitSkeleton(singleSkel,degree=1)
            fits.append(fit)
            xdatas.append(xdata)
            ydatas.append(ydata)

    #Plotting
    segmentMask = np.ma.masked_where(IT.Label(thresholdPrediciton) < 0, IT.Label(thresholdPrediciton))
    plt.imshow(segmentMask,cmap='jet',interpolation='none')
    for i,fit in enumerate(fits):
        fit = fit*(i+1)
        fitmask = np.ma.masked_where(IT.Label(skeleton) < 0, skeleton)
        plt.imshow(fitmask,cmap='brg',alpha=1,interpolation='none')
        plt.plot(ydatas[i],xdatas[i],linewidth=2)
    #plt.imshow(im,cmap='gray',alpha=0.5,interpolation='none')
    #plt.colorbar()
    plt.show()
    plt.pause(2)
    plt.clf()

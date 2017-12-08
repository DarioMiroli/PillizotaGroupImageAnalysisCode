from pyzota_image_toolbox import imageTools as IT
import numpy as np
import sys
import matplotlib.pyplot as plt

#Machine learning stuff
sys.path.insert(0, '/home/s1033855/OperationLeadPelican/MachineLearningTest/KerasTests/ImageSegmentationWithKeras/MachineSegmenter')
from MachineSegmenter import MachineSegmenter
inputFolder1 = "./Analysis/400mM_Slide1_CroppedCells"
inputFolder2 = "./Analysis/Slide1_CroppedCells"
inputFolders = [inputFolder1,inputFolder2]
movieFolder = "./Analysis/MovieOutput/"

#Analysis
cellLengths = [[],[]]
cellAreas = [[],[]]

#thresholdFolder = "./Analysis/Slide1_Vetted"
#IT.CreateFolder(thresholdFolder)
M1 = MachineSegmenter()
M1.loadModel("/home/s1033855/OperationLeadPelican/MachineLearningTest/KerasTests/ImageSegmentationWithKeras/MachineSegmenter/Files/Models/TempModels/TempModel1.h5")
M1.compileModel()

jim = 0
for k, inputFolder in enumerate(inputFolders):
    #File IO
    fileNamesPlusFolder = IT.GetFileNamesFromFolder(inputFolder,fileOnly=False)
    fileNamesPlusFolder.sort()
    fileNamesOnly = IT.GetFileNamesFromFolder(inputFolder,fileOnly=True)
    fileNamesOnly.sort()

    plt.ion()
    for i,im in enumerate(fileNamesPlusFolder):
        im = IT.Open(im)
        machinePrediction = M1.predict([im])
        thresholdPrediciton = machinePrediction[0] > 0.5
        thresholdLabels = IT.Label(thresholdPrediciton)
        distanceTransform = IT.DistanceTransform(thresholdPrediciton)
        thresholdDT = distanceTransform > 0.5*np.amax(distanceTransform)
        skeleton = IT.Skeletonize(thresholdPrediciton)
        skeletonLabels = IT.Label(skeleton)
        noOfSkeletons = np.amax(skeletonLabels) + 1
        noOfCells =  np.amax(thresholdLabels) + 1
        fits = []
        xdatas = []
        ydatas   =[]

        if noOfCells == noOfSkeletons:
            for skel in range(-1,noOfSkeletons):
                singleSkel = 1*(skeletonLabels == skel)
                singleCell = 1*(thresholdLabels == skel)
                if skel!= -1:
                    fit,xdata,ydata,zs = IT.FitSkeleton(singleSkel,degree=1)
                    cellLength = IT.getCellLength(singleCell,zs)
                    fits.append(fit)
                    xdatas.append(xdata)
                    ydatas.append(ydata)
                    cellLengths[k].append(cellLength)
                    print("xaviere is a prick", len(singleCell.nonzero()[0]))
                    cellAreas[k].append(len(singleCell.nonzero()[0]))

            #Plotting
            segmentMask = np.ma.masked_where(IT.Label(thresholdPrediciton) < 0, IT.Label(thresholdPrediciton))
            #plt.imshow(segmentMask,cmap='jet',interpolation='none')
            for i,fit in enumerate(fits):
                fit = fit*(i+1)
                fitmask = np.ma.masked_where(IT.Label(skeleton) < 0, skeleton)
                #plt.imshow(fitmask,cmap='brg',alpha=1,interpolation='none')
                #plt.plot(ydatas[i],xdatas[i],linewidth=2)
            #plt.imshow(im,cmap='gray',alpha=0.5,interpolation='none')
            #plt.show()
            #plt.pause(1)
            plt.clf()
            plt.hist(cellLengths[0],bins=[i*5 for i in range(50)],normed=True,alpha=0.5)
            plt.hist(cellLengths[1],bins=[i*5 for i in range(50)],normed=True,alpha=0.5)
            #plt.show()
            plt.pause(0.1)

plt.pause(100)
plt.savefig("./100mLengthsTestHist1.png")

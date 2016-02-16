from Image_Analysis_Tools import *

def Run(Input, Output,Functions,ParameterArray):
    filenames = Setup(Input, Output)
    for file in filenames:
        Images = []
        Images.append(Open(Input+"/"+file))
        for i in range(len(Functions)):
            if Functions[i] == Compare:
                Parameters = [Images]
                Functions[i](*Parameters)
                SavePlot(Output+"/"+file+"Compilation")
                #Show()
            else:
                Parameters = [Images[-1]] + ParameterArray[i]
                Image = Functions[i](*Parameters)
                Images.append(Image)

if __name__ == "__main__":
    #Set up Steps to Image Analysis ************************************************
    InputFolder = "Images"
    OutputFolder = "Output"
    sigma = 0   #Sigma of gaussian Blur
    blocks = 20 #Area used by adaptive threshold
    erosions  = 1 #Number of binary erosions
    dilations = 1 #Number of binary dilations #3
    smallest, largest = 100,2400 #Removes objects smaller/greater than this size
    Functions  = [Blurr,Threshold,Erode, Dilate,Label,SiveArea,Compare]
    Parameters = [[sigma],[blocks],[erosions],[dilations],[],[smallest,largest],[]]
    FolderCompare(InputFolder)
    Run(InputFolder,OutputFolder,Functions,Parameters)
    #********************************************************************************

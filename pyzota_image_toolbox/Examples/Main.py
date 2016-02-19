def Run(Input, Output,Functions,ParameterArray):
    filenames = Setup(Input, Output)
    for file in filenames:
        Images = []
        Images.append(Open(os.path.join(Input,file)))
        for i in range(len(Functions)):
            if Functions[i] == Compare:
                Parameters = [Images]
                Functions[i](*Parameters)
                SavePlot(os.path.join(Output,file+"Compilation"))
                Show()
            else:
                Parameters = [Images[-1]] + ParameterArray[i]
                Image = Functions[i](*Parameters)
                Images.append(Image)
def addToolsToPath():
    currentPath=sys.path[0]
    back1 = os.path.dirname(currentPath)
    toolsPath = os.path.join(back1, 'Tools')
    sys.path.insert(0, toolsPath)

if __name__ == "__main__":
    import sys, os
    addToolsToPath()
    from ImageAnalysisTools import *
    #Set up Steps to Image Analysis ************************************************
    InputFolder = os.path.join(os.path.dirname(sys.path[0]),"ExampleImages")
    OutputFolder = os.path.join(os.path.dirname(sys.path[0]),"ExampleOutput")
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

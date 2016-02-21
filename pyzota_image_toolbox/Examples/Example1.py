def Run(Input, Output,Functions,ParameterArray):
    '''
    Takes a list of functions and a list of parameter lists. Runs through each
    one in turn for each file in Input and saves results to output.
    '''
    filenames = IT.Setup(Input, Output)
    for file in filenames:
        Images = []
        Images.append(IT.Open(os.path.join(Input,file)))
        for i in range(len(Functions)):
            if Functions[i] == IT.Compare:
                Parameters = [Images]
                Functions[i](*Parameters)
                IT.SavePlot(os.path.join(Output,file+"Compilation"))
                IT.Show()
            else:
                Parameters = [Images[-1]] + ParameterArray[i]
                Image = Functions[i](*Parameters)
                Images.append(Image)

def addToolsToPath():
    '''Adds the pyzota_image_toolbox to path so it can be imported as normal '''
    currentPath=sys.path[0]
    back1 = os.path.dirname(currentPath)
    packagePath =  os.path.dirname(back1)
    sys.path.insert(0, packagePath)

if __name__ == "__main__":
    """Example which loads images from a file and runs through a few standard
    routines saving results to ExampleOutput."""
    import sys, os
    addToolsToPath()
    from pyzota_image_toolbox import imageTools as IT
    #Set up Steps to Image Analysis ************************************************
    InputFolder = "ExampleImages"
    OutputFolder = "ExampleOutput"
    sigma = 0   #Sigma of gaussian Blur
    blocks = 20 #Area used by adaptive threshold
    erosions  = 1 #Number of binary erosions
    dilations = 1 #Number of binary dilations #3
    smallest, largest = 100,2400 #Removes objects smaller/greater than this size
    Functions  = [IT.Blurr,IT.Threshold,IT.Erode, IT.Dilate,IT.Label,IT.SiveArea,IT.Compare]
    Parameters = [[sigma],[blocks],[erosions],[dilations],[],[smallest,largest],[]]
    IT.FolderCompare(InputFolder)
    Run(InputFolder,OutputFolder,Functions,Parameters)
    #********************************************************************************

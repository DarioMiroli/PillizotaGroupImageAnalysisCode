from pyzota_image_toolbox import imageTools as IT
import sys
import matplotlib.pyplot as plt
#Import Machine learning stuff
pathToMachineLearning = ("/home/s1033855/OperationLeadPelican/"
        "MachineLearningTest/KerasTests/ImageSegmentationWithKeras/"
        "MachineSegmenter")
sys.path.insert(0, pathToMachineLearning)
from MachineSegmenter import MachineSegmenter

#**************************** Enter input file here ***************************#
dataStore = ("/run/user/1001/gvfs/smb-share:server=csce.datastore.ed.ac.uk,shar"
        "e=csce/biology/groups/pilizota/")
pathToData = ("Dario/Data/GrowthAtHigherOsmolarities/30_11_17/40mM_NaCl_M63"
        "+Glu+CAA_OD_0-32/Slide2/")
outputFolder =  pathToData.split("/")[-3]
outputFile = "CellData.pickle"
#**************************** Enter input file here ***************************#

#Check input output folders with user
pathToData = dataStore+pathToData
outputFolderPath = "Analysis/" + outputFolder
print("Reading from:{}\nWriting to: {}\n".format(pathToData,outputFolderPath))
ok = raw_input("Is this ok (y/n)")
if ok != 'y':
    exit()

#Get filenames and paths
fileNames = IT.StripImageFiles(IT.Setup(pathToData,outputFolderPath))

#Set up machine segmenter
pathToModel = ("/home/s1033855/OperationLeadPelican/MachineLearningTest/"
        "KerasTests/ImageSegmentationWithKeras/MachineSegmenter/Files/Models"
        "/TempModels/TempModel1.h5")
M1 = MachineSegmenter()
M1.loadModel(pathToModel)
M1.compileModel()


for i,f in enumerate(fileNames):
    print("Processed up to image {}".format(i))
    image = IT.Open(pathToData+f)
    rects = IT.SelectReigon(image,
            title="Processing Image {} of {} .".format(i,len(fileNames)))
    for rec in rects:
        ROI = IT.Crop(image,rec)
        segmented = M1.predict([ROI])[0]
        labelled = IT.Label(IT.GlobalThreshold(segmented,0.5))
        vetted = IT.CompareAnnotate([ROI,labelled],commonScaleBar=False,
                TitleArray=["ROI","Segmented"],data=labelled)
        bBoxImages,bBoxMasks = IT.BboxImages(ROI,vetted+2)
        for b,box in enumerate(bBoxMasks):
            IT.SaveCellToFile(bBoxImages[b],box,outputFolder,f,
            outputFolderPath+"/" + outputFile)

from pyzota_image_toolbox import imageTools as IT
import matplotlib.pyplot as plt

inputFolder = "./Analysis/Slide1_CroppedCells"
vettedFolder = "./Analysis/Slide1_Vetted"
IT.CreateFolder(vettedFolder)


fileNamesPlusFolder = IT.GetFileNamesFromFolder(inputFolder,fileOnly=False)

fileNamesPlusFolder.sort()

fileNamesOnly = IT.GetFileNamesFromFolder(inputFolder,fileOnly=True)
fileNamesOnly.sort()

plt.ion()
for i,im in enumerate(fileNamesPlusFolder):
    im = IT.Open(im)
    plt.clf()
    plt.imshow(im,interpolation='none',cmap='gray')
    plt.colorbar()
    keep = "0"
    while keep != "k" and keep!="d":
        keep = raw_input("Keep or delete image?(k/d)")
        if keep == "k":
            IT.Save(im,vettedFolder+"/"+fileNamesOnly[i])

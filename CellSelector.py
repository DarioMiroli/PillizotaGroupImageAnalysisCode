from pyzota_image_toolbox import imageTools as IT

folder = "/run/user/1001/gvfs/smb-share:server=csce.datastore.ed.ac.uk,share=csce/biology/groups/pilizota/Dario/Data/GrowthAtHigherOsmolarities/30_11_17/40mM_NaCl_M63+Glu+CAA_OD_0-32/Slide2/"
croppedFolder = "./Analysis/400mM_Slide1_CroppedCells"

IT.CreateFolder(croppedFolder)
fileNamesPlusFolder = IT.GetFileNamesFromFolder(folder,fileOnly=False)
fileNamesPlusFolder.sort()
fileNamesOnly = IT.GetFileNamesFromFolder(folder,fileOnly=True)
fileNamesOnly.sort()

#Remove non tifs
for i in range(len(fileNamesOnly)):
    print(len(fileNamesOnly))
    print(fileNamesOnly[i])
    if not fileNamesOnly[i].endswith(".tiff"):
        del fileNamesOnly[i]
        del fileNamesPlusFolder[i]
    if i == len(fileNamesOnly)-1:
        break


images = []
for f in fileNamesPlusFolder:
        images.append(IT.Open(f))

for j,image in enumerate(images):
    croppedImages = []
    rects = IT.SelectReigon(image,title="On Image {}/{}".format(j,len(images)))
    for rec in rects:
        croppedImages.append(IT.Crop(image,rec))
    IT.AniShow(croppedImages)
    shouldSave = raw_input("SaveImages?(y/n)")
    if shouldSave == "y":
        for i,crop in enumerate(croppedImages):
            path = croppedFolder+"/"+str(i)+"_"+fileNamesOnly[j]
            IT.Save(crop,path)

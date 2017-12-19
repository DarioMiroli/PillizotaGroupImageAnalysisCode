from pyzota_image_toolbox import imageTools as IT
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np

dataFolder = "./Analysis/40mM_NaCl_M63+Glu+CAA_OD_0-32/400mM_NaCl_Slide1.pickle"
areaCutOff = 10
f = open(dataFolder,'r')
endOfFile = False
dataDic = {"Areas":[],"Widths":[],"Lengths":[]}

#plt.ion()
while not endOfFile:
    try:
        d = pickle.load(f)
        mask = d["BinaryMask"]
        if mask.size > 10:
            mask = IT.ClearBorders(mask)
            area = IT.GetArea(mask)
            if area > areaCutOff:
                dataDic["Areas"].append(area)
                Length = IT.GetSebLength(mask)
                #plt.imshow(mask,interpolation = "None")
                #plt.show()
                #plt.title(area)
                #plt.pause(5)
                #plt.clf()


    except EOFError:
        print("End of File")
        endOfFile = True
    #exit()
#Plot Histograms
#plt.ioff()
#plt.hist(dataDic["Areas"],bins=[100*i for i in range(30)])
#plt.show()

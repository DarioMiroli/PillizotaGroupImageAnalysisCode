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
jj=0
while not endOfFile:
    try:
        d = pickle.load(f)
        mask = d["BinaryMask"]
        if mask.size > 10:
            mask = IT.ClearBorders(mask)
            area = IT.GetArea(mask)
            jj+=1
            if area > areaCutOff:
                dataDic["Areas"].append(area)
                length,width = IT.GetLengthAndWidth(mask,jj)
                dataDic["Lengths"].append(length)
                dataDic["Widths"].append(width)

    except EOFError:
        print("End of File")
        endOfFile = True

#Plot Histograms
plt.ioff()
plt.clf()

plt.hist(dataDic["Areas"],bins=np.linspace(0,2000,50))
plt.show()

plt.hist(dataDic["Lengths"],bins=np.linspace(0,300,50))
plt.show()

plt.hist(dataDic["Widths"],bins=np.linspace(0,30,50))
plt.show()

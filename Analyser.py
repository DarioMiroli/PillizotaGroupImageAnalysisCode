from pyzota_image_toolbox import imageTools as IT
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
dataFolder = "./Analysis/40mM_NaCl_M63+Glu+CAA_OD_0-32/CellData.pickle"

f = open(dataFolder,'r')
endOfFile = False
areas = []
plt.ion()
i=0
while not endOfFile:
    i+=1
    try:
        d = pickle.load(f)
        mask = d["BinaryMask"]
        if len(np.unique(mask)) != 2:
            if len(np.unique(mask)) > 2:
                cleared = IT.ClearBorders(mask)
                areas.append(IT.GetArea(cleared))
        else:
            areas.append(IT.GetArea(d["BinaryMask"]))
        plt.clf()
        #plt.imshow(d["RawImage"],cmap='gray',alpha =1)
        if areas[-1] < 5:
            del areas[-1]
        if areas[-1] < 350:
            plt.imshow(mask,alpha=1,interpolation='None')
            plt.colorbar()
            plt.pause(0.5)
            plt.title(str(i))
            print(i,areas[-1])
    except EOFError:
        print("End of File")
        endOfFile = True
plt.clf()
plt.ioff()
plt.hist(areas,bins=[i*50 for i in range(50)])
plt.show()
print("Done")

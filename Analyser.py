from pyzota_image_toolbox import imageTools as IT
import sys
import pickle


dataFolder = "./Analysis/40mM_NaCl_M63+Glu+CAA_OD_0-32/CellData.pickle"

f = open(dataFolder,'r')
endOfFile = False
areas = []
while not endOfFile:
    try:
        d = pickle.load(f)
        plt.imshow(d["BinaryMask"],alpha=0.3)
        plt.imshow(d["RawImage"],cmap='jet')
        plt.imshow(d["BinaryMask"],alpha=0.3)
        plt.show()
        IT.ShowMe(d["RawImage"],cmap='jet')
        IT.ShowMe(d["BinaryMask"])
        IT.GetArea(d["BinaryMask"])
    except EOFError:
        print("End of File")
        endOfFile = True

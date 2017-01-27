"""Short sript to test that all modules can be imported correctly. Test.py sits
at the correct place in the file hierachy such that pyzota_image_toolbox can be
imported as a standard package without needing to make changes to the path."""
import pyzota_image_toolbox
from pyzota_image_toolbox import imageTools as IT
import numpy as np
import matplotlib.pyplot as plt
print("Start")
from skimage import data
path = "pyzota_image_toolbox/Examples/ExampleImages"

fileNames = IT.getFileNamesFromFolder(path)

RGB = IT.open(path+"/" + fileNames[3])
grayScale = IT.open(path+"/" + fileNames[2])

ColorBlurr = IT.blurr(RGB,10)
grayBlurr = IT.blurr(grayScale,10,imageType="gray")
IT.showMe(RGB)
IT.showMe(ColorBlurr)
IT.showMe(grayScale)
IT.showMe(grayBlurr)

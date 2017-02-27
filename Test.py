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
images = []

for i in range(len(fileNames)):
    images.append(IT.open(path+"/" + fileNames[i]))

IT.Compare(images, TitleArray=fileNames, commonScaleBar=False)
rectangle = IT.selectReigon(images[0])
cropped = IT.crop(images[0],rectangle)
IT.showMe(cropped)

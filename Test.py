"""Short sript to test that all modules can be imported correctly. Test.py sits
at the correct place in the file hierachy such that pyzota_image_toolbox can be
imported as a standard package without needing to make changes to the path."""
import pyzota_image_toolbox
from pyzota_image_toolbox import imageTools as IT
import numpy as np

print("Success")

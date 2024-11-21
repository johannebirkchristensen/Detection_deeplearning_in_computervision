
import cv2
import sys
import numpy as np

print(sys.argv[1])
image = cv2.imread(sys.argv[1])
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
#ss.switchToSelectiveSearchFast()
ss.switchToSelectiveSearchQuality() #(base_k=150, inc_k=150, sigma=0.8)
rects = ss.process()

print(rects)
print(len(rects))
np.save(f"{sys.argv[1]}.npz", rects)




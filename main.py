import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def resize_pic(frame, scale=0.7):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimension = (width,height)
    return cv.resize(frame, dimension, interpolation=cv.INTER_AREA)

img = cv.imread('face.webp')
resize = resize_pic(img)
gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
rgb = cv.cvtColor(gray, cv.COLOR_BGR2RGB)




cv.imshow('img', rgb)


lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('lap', lap)
threshold, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)
cv.imshow('threshold', thresh)

plt.figure()
plt.title('histogram')
plt.xlabel('bins')
plt.ylabel('# of pixels')
color = ('b','g','r')

for i,col in enumerate(color):
    hist = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])

plt.show()


resized = resize_pic(img)
rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB)

plt.imshow(rgb)
plt.show()

cv.waitKey(0)


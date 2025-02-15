import cv2
from matplotlib import pyplot as plt

img = cv2.imread('0000018.jpg')  # 读取图像
cv2.imshow('original', img)  # 显示原图像
# img2 = cv2.Sobel(img, cv2.CV_8U, 0, 1)  # 边缘检测
img2 = cv2.Laplacian(img, cv2.CV_8U) 	    # 边缘检测
img3 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
plt.imshow(img3, cmap=plt.cm.tab20c)
plt.axis('off')
plt.savefig("laplacian_.jpg", bbox_inches='tight', pad_inches = -0.1)
plt.close()
# cv2.imwrite("laplacian.jpg", img2)
cv2.imshow('Sobel', img2)  # 显示结果
cv2.waitKey(0)

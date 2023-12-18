'''
Author: WHURS-THC
Date: 2023-04-27 23:28:18
LastEditTime: 2023-04-27 23:28:36
Description: 
'''
# 用opencv读入一张图片 
import cv2
img = cv2.imread('test.jpg')
# 创建窗口并显示图像
cv2.namedWindow('Image')
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


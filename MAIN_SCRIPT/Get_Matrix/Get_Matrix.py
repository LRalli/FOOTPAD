import cv2
import numpy as np

#Immagine del campo da gioco come visto nel video
CampoVideo = cv2.imread('src.jpg') 
#Immagine del campo da gioco in bird view scelta
CampoPlane = cv2.imread('dst1.png') 

CampoVideoShape = CampoVideo.shape
#print("Dimensioni campo video: ", CampoVideoShape)
CampoPlaneShape = CampoPlane.shape
#print("Dimensioni campo bird view: ", CampoPlaneShape)

#Test per trovare punti su mappa bird view
#cv2.circle(CampoPlane, (548,45), 5, (0,0,255), -1)
#cv2.circle(CampoPlane, (460,368), 5, (0,0,255), -1)
#cv2.circle(CampoPlane, (633,368), 5, (0,0,255), -1)
#cv2.circle(CampoPlane, (548,693), 5, (0,0,255), -1)

#imgS = cv2.resize(CampoPlane, None, fx=0.9, fy=0.9)
#cv2.imshow('Test', imgS)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#Test per trovare punti su frame video
#cv2.circle(CampoVideo, (940,96), 10, (0,0,255), -1)
#cv2.circle(CampoVideo, (1427,395), 10, (0,0,255), -1)
#cv2.circle(CampoVideo, (455,395), 10, (0,0,255), -1)
#cv2.circle(CampoVideo, (943,1022), 10, (0,0,255), -1)

#imgS = cv2.resize(CampoVideo, None, fx=0.8, fy=0.8)
#cv2.imshow('Test',imgS)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#(top, right, left, bottom)
pts1 = np.float32([[940,96],[1427,395],[455,395],[943,1022]])
#print(pts1)
pts2 = np.float32([[548,45],[633,368],[460,368],[548,693]])   
#print(pts2)
pts3 = np.float32([[943,395]]) #Per test su punto centrale
#print(pts3)

M = cv2.getPerspectiveTransform(pts1,pts2)
print(M)

#Test Funzionamento Matrice
"""
pts3o = cv2.perspectiveTransform(pts3[None, :, :],M)
print(pts3o)

x = int(pts3o[0][0][0]) # = ascissa punto centrale
y = int(pts3o[0][0][1]) # = ordinata punto centrale
p = (x,y)

cv2.circle(CampoPlane, p, 5, (0,0,255),-1)

cv2.imshow('Test',CampoPlane)

cv2.waitKey(0)

cv2.destroyAllWindows()
"""


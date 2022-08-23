import cv2
import numpy as np
import tensorflow as tf
from numpy import argmax
from tensorflow import keras
from tensorflow.keras.models import load_model

############################# Set-up ########################################

#Carica model Keras allenato
KerasModel = load_model(r'files/model.h5')

#Leggi il file video di input
cap = cv2.VideoCapture(r'data/video.avi')

#controlla se il file video di input è stato aperto correttamente
if (cap.isOpened()== False): 
    print("Errore nell'apertura del file video")

temp = cv2.imread(r'data/temp.jpg',0)
bird = cv2.imread(r'data/dst1.jpg')

#Salva dimensioni palla
wt, ht = temp.shape[::-1]

#Prendi dimensioni video input che vanno prima convertite da float a int
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

#Salva il video in game con le stesse dimensioni del video in input e il birdview in bird con le stesse
#dimensioni dell'immagine su cui si vuole mappare il movimento dei giocatori
game = cv2.VideoWriter('game.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (1920,1080))
birdview = cv2.VideoWriter('bird.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (1100, 746))

##############################################################################


############################# Load Yolo ########################################

#Carica Network con pre-trained weights e configuration file di Yolo
net = cv2.dnn.readNet("files\yolo-obj_best.weights", "files\yolo-obj.cfg")

#Carica le classi dal file classes
classes = []
with open("files\obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

#Set-up layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#Colora layers in modo diverso per ogni classe
colors = np.random.uniform(0, 255, size=(len(classes), 3)) 

################################################################################


############################# Get Detected #####################################

def get_detected(outs,height, width):

    class_ids = []
    confidences = []
    boxes = []
    detected = []
    info = []

    for out in outs:
        for detection in out:
            #Prendiamo la confidence
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)   # Coordinata x centro oggetto 
                center_y = int(detection[1] * height)  # Coordinata y centro oggetto
                w = int(detection[2] * width)          # Larghezza oggetto trovato
                h = int(detection[3] * height)         # Altezza oggetto trovato
                # Coordinate angolo alto sinistro Rettangolo
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                info.append([x, y, w, h, class_id])  
                boxes.append([x, y, w, h])            # coordinate oggetto trovato
                confidences.append(float(confidence)) # confidence oggetto trovato
                class_ids.append(class_id)            # nome oggetto trovato
                
    # Funzione NoMaxSuppression per ridurre rumore
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    #Ciclo su tutte le detection per aggiungere a i-esimo detected le sue informazioni contenute in boxes
    for i in range(len(boxes)):
        #tengo conto della NoMaxSuppression per evitare che un oggetto compaia 2 volte
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label!='Palla':
                detected.append(info[i])
            
    return detected

################################################################################

############################# Bird View ########################################

def plane(players,ball):
    planetemp = bird.copy()
    #Inserisco la matrice calcolata con Get_Matrix.py
    matrix=np.array([[3.16690465e-01, 7.60754322e-01, 2.79973193e+02],
                     [2.11272233e-02, 1.79420066e+00, -1.38669899e+02],
                     [5.74109329e-05, 1.38992407e-03, 1.00000000e+00]])

    for p in players:
        x = p[0] + int(p[2]/2) 
        y = p[1] + p[3]
        ptsGiocatore = np.float32([[x,y]])
        ptsGiocatoreo = cv2.perspectiveTransform(ptsGiocatore[None, :, :],matrix)
        x1=int(ptsGiocatoreo[0][0][0]) # = ascissa punto centrale
        y1=int(ptsGiocatoreo[0][0][1]) # = ordinata punto centrale
        pp = (x1,y1)
        if(p[5]==0):
            cv2.circle(planetemp, pp, 15, (255,0,0),-1)
        elif p[5]==1:
            cv2.circle(planetemp, pp, 15, (255,255,255),-1)
        elif p[5]==2:
            cv2.circle(planetemp, pp, 15, (0,0,255),-1)
    if len(ball) !=0:
        
        xb = ball[0] + int(ball[2]/2)
        yb = ball[1] + int(ball[3]/2)
        ptsball = np.float32([[xb,yb]])
        ptsballo = cv2.perspectiveTransform(ptsball[None, :, :],matrix)
        x2 = int(ptsballo[0][0][0]) # = ascissa punto centrale
        y2 = int(ptsballo[0][0][1]) # = ordinata punto centrale
        pb = (x2,y2)
        cv2.circle(planetemp, pb, 15, (0,0,0),-1)
    return planetemp

################################################################################


############################# Main Phase ######################################

opr=0
frameN = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    
    players=[]
    ball=[]
    if opr<310:
        opr=opr+1
        continue
    
    if ret == True :
        
        frameN += 1
        copy = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #Calcolo dimensioni frame
        height, width, channels = frame.shape
        
        #Estrazione features tramite blob
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        #Test funzionamento blob con display dei 3 canali
        #for b in blob:
        #for n, img_blob in enumerate(b):
        #cv2.imshow(str(b), img_blob)

        #Blob processato dall'algoritmo YOLO
        net.setInput(blob)
        outs = net.forward(output_layers)
        outs = get_detected(outs, height, width)
        for i in range(len(outs)):
            x, y, w, h, class_id = outs[i]
            Clabel = str(classes[class_id])
            roi = frame[y:y+h,x:x+w]
            
            #Alcuni frame potrebbero non andare bene per la classification quindi la funzione può lanciare errori
            try:
                roi = cv2.resize(roi, (96,96))
            except:
                continue
            ym = KerasModel.predict(np.reshape(roi,(1,96,96,3)))
            ym = argmax(ym)
            
            players.append([x,y,w,h,Clabel,ym])
            
            if ym==0:
                cv2.rectangle(copy, (x, y), (x + w, y + h), (0,0,255), 2)  #Disegno rettangolo con coordinate angolo alto sx e basso dx
            elif ym==1:
                cv2.rectangle(copy, (x, y), (x + w, y + h), (0,255,0), 2)
            elif ym==2:
              if Clabel=="Arbitro":
                cv2.rectangle(copy, (x, y), (x + w, y + h), (255,0,0), 2)
              elif Clabel=="Guardalinee":
                cv2.rectangle(copy, (x, y), (x + w, y + h), (255,255,0), 2)
            
        
        res = cv2.matchTemplate(gray,temp,cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if min_val < 0.05:
            top_left = min_loc
            bottom_right = (top_left[0] + wt, top_left[1] + ht)
            ball.append(top_left[0])
            ball.append(top_left[1])
            ball.append(wt)
            ball.append(ht)
            cv2.rectangle(copy,top_left, bottom_right, (0,255,100), 2)
            
        p = plane(players, ball)
            
        game.write(copy)
        birdview.write(p)
        
        print("Frame", frameN, ": Detection Completata!")
        
    if cv2.waitKey(1)==27:
        break

#Release della clip di input e output
cap.release()
game.release()
birdview.release()

################################################################################

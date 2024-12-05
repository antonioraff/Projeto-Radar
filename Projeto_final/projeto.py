import numpy as np
import cv2
import sys
import time

video = "videocorreto.mp4" 

cap = cv2.VideoCapture(video) 

mog2 = cv2.createBackgroundSubtractorMOG2(history = 500, detectShadows=True, varThreshold=100)  

w_min = 70 
h_min = 70  
offset = 4 
linha_ROI = 300 
carros = 0

def centroide(x, y, w, h): 

    x1 = w // 2 
    y1 = h // 2
    cx = x + x1
    cy = y + y1
    return cx, cy

detec = []

def set_info(detec): 
    global carros
    for (x, y) in detec:
        if (linha_ROI + offset) > y > (linha_ROI - offset): 
            carros += 1
            cv2.line(frame, (0, linha_ROI), (1400, linha_ROI), (0, 127, 255), 3) 
            detec.remove((x, y))
            print("Carros detectados: " + str(carros))


def show_info(frame, mask):
    texto = f'Carros: {carros}' 
    cv2.putText(frame, texto, (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6) 
    cv2.putText(frame, texto, (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3) 
    cv2.imshow("Video Original", frame) 
    cv2.imshow("Detectar", mask)
    

while True:

    true, frame = cap.read() 

    if not true: 
        break
    
    mask = mog2.apply(frame)

    kernel_circular = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) 
    kernel_retangular = np.ones((3,3), np.uint8) 
    
    fechamento = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_retangular, iterations=2)
    abertura = cv2.morphologyEx(fechamento, cv2.MORPH_OPEN, kernel_retangular, iterations=2)
    dilatacao = cv2.dilate(abertura, kernel_circular, iterations=2)
    
    contorno, img = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    cv2.line(frame, (0, linha_ROI), (1400, linha_ROI), (0, 0, 0), 2) 

    
    for (i, j) in enumerate(contorno): 
        (x, y, w, h) = cv2.boundingRect(j)
        validar_contorno = (w >= w_min) and (h >= h_min) 
        if not validar_contorno:
            continue
 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 
        centro = centroide(x, y, w, h) 
        detec.append(centro)
        cv2.circle(frame, centro, 5, (0, 0, 255), -1) 


    set_info(detec) 
    show_info(frame, dilatacao)

    time.sleep(0.08) 
    
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
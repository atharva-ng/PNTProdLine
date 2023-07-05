from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# Install requirements.txt and torch vision properly

model = YOLO("yoloWeights/yolov8x.pt")
cap = cv2.VideoCapture("videos/vid.mp4")

className=["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter"
            ,"bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra","giraffe", "backpace", "umbrella", "handbag", "tie"
            ,"suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle"
            ,"wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair"
            ,"couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote","keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator"
            ,"book", "clock","vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

mask=cv2.imread("images/mask.png")          #Masking the unnecessary area

tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3)           #Tracking the objects through multiple frames

#Limits for the line
limits1=[210,535,610,535 ]
limits2=[200,600,610,600]

counts=[]           #List of object IDS

while True:
    success,img =cap.read()
    imgRegion=cv2.bitwise_and(img,mask)
    result=model(imgRegion,stream=True,device="mps")

    detections = np.empty((0, 5))

    line1=cv2.line(img, (limits1[0],limits1[1]), (limits1[2],limits1[3]), (0,0,255), 5)
    line2 = cv2.line(img, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 0, 255), 5)

    for r in result:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]

            x1, y1, x2, y2=int(x1),int(y1),int(x2),int(y2)

            w,h=x2-x1,y2-y1

            confidence = math.ceil(box.conf[0]*100)/100

            cls=int(box.cls[0])

            if className[cls]=="car" or className[cls]=="motorcycle" or className[cls] =="bus" or className[cls]=="truck":

                currentArray=np.array([x1,y1,x2,y2,confidence])
                detections=np.vstack((detections,currentArray))
                cvzone.putTextRect(img, f"{className[cls]}", (max(0, x1), max(35, y1)), thickness=1, scale=1.5,
                                                       offset=1)


    resultsTracker=tracker.update(detections)

    for dets in resultsTracker:
        x1,y1,x2,y2,id=dets

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        cvzone.putTextRect(img, f"{int(id)}", (max(0, x1), max(35, y1)), thickness=1, scale=1.5,
                           offset=1)

        cvzone.cornerRect(img, (x1, y1, w, h), l=5, rt=1,colorR=(255,0,0))

        cx,cy=x1+w//2,y1+h//2
        cv2.circle(img,(cx,cy),2,(255,0,255),cv2.FILLED)

        if limits1[0]<cx<limits1[2] and limits1[1]-10<cy<limits1[1]+10:
            if counts.count(id)==0:
                counts.append(id)
                line1 = cv2.line(img, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (0, 255, 0), 5)

        if limits2[0]<cx<limits2[2] and limits2[1]-20<cy<limits2[1]+20:
            if counts.count(id)==0:
                counts.append(id)
                line2 = cv2.line(img, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 255, 0), 5)

        cvzone.putTextRect(img, f"Count : {len(counts)}", (50,50))


    cv2.imshow("Image",img)

    key=cv2.waitKey(1)
    if key==27:
        break

cap.release()

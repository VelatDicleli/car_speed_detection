from ultralytics import YOLO
import cv2
from sort import*
import cvzone
import math
import time

mymodel = YOLO('yolov8m.pt')

coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]




cap = cv2.VideoCapture("road2.mp4")

cap.set(3, 1280)  # Frame genişliğini ayarla
fps = int(cap.get(cv2.CAP_PROP_FPS))
cap.set(4, 720)  # Frame yüksekliğini ayarla



# output_path = "myvid09.mp4"
# fourcc = cv2.VideoWriter_fourcc(*'H264')
# out = cv2.VideoWriter(output_path, fourcc,30, (1280, 720))



mask = cv2.imread("mask.png")
mask = cv2.resize(mask, (1280, 720))  

tracker = Sort(max_age=18,min_hits=1, iou_threshold=0.3)


    
lim =[270, 220,496, 220]

lim2 = [499, 215,675, 203]
lim3 = [1029, 295,1257, 416]

lim4 = [150, 296,504, 300]

lim5 = [510, 300,731, 283]

lim6 = [857, 356,1133, 540]

mesafe = 9



cars=[]
cars1=[]
cars2=[]

cars_speed =[]
cars_speed1 = []
cars_speed2 = []


car_start_times = {}
car_start_times1 = {}
car_start_times2 = {}






while True:
    ret, frame = cap.read()
    
    

    imgRegion = cv2.bitwise_and(frame, mask)
    results = mymodel(imgRegion, stream=True)

    detects=np.empty((0,5))
   
    for r in results:
        boxes =r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            currentClass = coco_classes[cls]
            
            
            if currentClass == "car" or  currentClass == "truck" or\
            currentClass =="bus" or currentClass == "motorcycle"and conf > 0.5:
                
                cArray = np.array([x1,y1,x2,y2,conf])
                detects = np.vstack((detects,cArray))

      # Non-Maximum Suppression (NMS) 
    confThreshold = 0.3
    nmsThreshold = 0.5
    indices = cv2.dnn.NMSBoxes(detects[:, :4], detects[:, 4], confThreshold, nmsThreshold)
    filtered_detects = detects[indices]       
   
    resultsTracker = tracker.update(filtered_detects)
    cv2.line(frame,(lim[0],lim[1]),(lim[2],lim[3]),(200,0,100),2)
    cv2.line(frame,(lim2[0],lim2[1]),(lim2[2],lim2[3]),(200,0,100),2)
    cv2.line(frame,(lim3[0],lim3[1]),(lim3[2],lim3[3]),(200,0,100),2)

    cv2.line(frame,(lim4[0],lim4[1]),(lim4[2],lim4[3]),(200,0,100),2)
    cv2.line(frame,(lim5[0],lim5[1]),(lim5[2],lim5[3]),(200,0,100),2)
    cv2.line(frame,(lim6[0],lim6[1]),(lim6[2],lim6[3]),(200,0,100),2)
    
    
    
    
    
    
    
    for result in resultsTracker:
        x1,y1,x2,y2,id =result
        x1,y1,x2,y2  = int(x1),int(y1),int(x2),int(y2)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
        w,h = x2-x1,y2-y1
        bx,by =  (x1 + x2) // 2, (y1 + y2) // 2  # Dikdörtgenin ortası
        cx,cy = x1 + w // 2, y1 
        
        

        cv2.circle(frame, (bx,by),2, (0, 200, 200), -1)  
          
          
        cv2.putText(frame, f'fps : {fps}', (50,50), 3, 1, (255, 0, 0))




        if lim[0] < bx < lim[2] and lim[1]- 10< by < lim[3] +10:
            if cars.count(id)==0 or len(cars)==0:
                cars.append(id)
                if id not in car_start_times:
                    start_time = time.time()
                    car_start_times[id] = start_time
              
                
        if lim4[0]< bx < lim4[2] and lim4[1] - 10< by < lim4[3] + 10:
            if id in car_start_times:
                current_time = time.time()
                time_diff = current_time - car_start_times[id]
                if time_diff > 0:
                    speed = (mesafe * 3.6) / (time_diff)
                    
                    cars_speed.append((id, str(speed)))
                    
                
                    

        
        
                    
        
        if lim2[0]< bx <lim2[2] and lim2[1]- 10 <by< lim2[3]+10:
            if cars1.count(id)==0 or len(cars1)==0:
                cars1.append(id)
                if id not in car_start_times1:
                    start_time1 = time.time()
                    car_start_times1[id] = start_time1
            
        if lim5[0]< bx< lim5[2] and lim5[1] - 10< by < lim5[3]+10:
            if id in car_start_times1:
                current_time1 = time.time()
                time_diff1 = current_time1 - car_start_times1[id]
                if time_diff1 > 0:
                    speed1 = (mesafe * 3.6) / (time_diff1)
                    
                    cars_speed1.append((id, str(speed1)))
                    
                  
                    

    
        

        if lim3[0]< bx <lim3[2] and lim3[1]  <by< lim3[3] :
            if cars2.count(id)==0 or len(cars2)==0:
                cars2.append(id)
                if id not in car_start_times2:
                    start_time2 = time.time()
                    car_start_times2[id] = start_time2
            
        if lim6[0]< bx < lim6[2] and lim6[1]  < by < lim6[3]:
            if id in car_start_times2:
                current_time2 = time.time()
                time_diff2 = current_time2 - car_start_times2[id]
                if time_diff2 > 0:
                    speed2 = (mesafe * 3.6) / (time_diff2)
                    
                    cars_speed2.append((id, str(speed2)))
                    
                   


        for car_id, car_speed in cars_speed :
            if car_id == id and car_speed == str(speed):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'hiz: {speed:.2f} km/saat', (cx, cy), 1, 2, (255, 255, 255), (255, 100, 0))

        
        for car_id1, car_speed1 in cars_speed1:
            if car_id1 == id and car_speed1 == str(speed1):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'hiz: {speed1:.2f} km/saat', (cx, cy), 1, 2, (255, 255, 255), (255, 100, 0))

                    

        for car_id2, car_speed2 in cars_speed2:
            if car_id2 == id and car_speed2 == str(speed2):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'hiz: {speed2:.2f} km/saat', (cx, cy), 1, 2, (255, 255, 255), (255, 100, 0))


   
    # out.write(frame)
    

   
    cv2.imshow("screen",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# out.release()
cv2.destroyAllWindows()
cap.release()

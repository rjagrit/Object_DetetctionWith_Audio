# we are using mobilenet here because it provides us the good balance between accuracy and speed
# the detection will be real time one
# we have coco.names dataset here

import cv2
import pyttsx3


Assistant= pyttsx3.init('sapi5')
voices= Assistant.getProperty('voices')
Assistant.setProperty('voices',voices[0].id)
Assistant.setProperty('rate',150)

def say(text):
    Assistant.say(text)
    Assistant.runAndWait()


cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)


#import the image using imread()
#img = cv2.imread('img.png')

#importing coco.name dataset automatically not manually
# classNames = ['person', 'car'] #manual methid it is

classNames = []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# creating our model
net = cv2.dnn.DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# objdet= False

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    print(classIds)

    #for classID in classIds here we use only one for loop but if we have more variables
    if len(classIds!=0):
    # creating a rectangular box around the images that we want to train with name from dataset using classId
        for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color= (0,255,0),thickness=2)
            cv2.putText(img,classNames[classId-1].upper(),(box[0]-5,box[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),thickness=3)
            # objdet= True

    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
        detectedObj= classNames[classId-1]
        print(detectedObj)
        say('Object detected was')
        say(detectedObj)



    # for displaying
    cv2.imshow("Output",img)
    cv2.waitKey(1)


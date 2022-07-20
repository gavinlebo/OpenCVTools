import cv2
import pandas
from sklearn import linear_model
import numpy as np
import copy

#Class for Tracker Pro
class TrackerPro:
    #Initialize all data for the tracker
    def __init__(self, tracker, video, waitTime):
        self.tracker = tracker
        self.previousPredictions = []
        self.waitTime = waitTime
        self.video = video
        self.framenumber = 0
    
    def init(self, frame, box):
        self.tracker.init(frame, box)
        self.framenumber = 0


    def update(self, frame):
        #Calculate information about given time
        fps = self.video.get(cv2.CAP_PROP_FPS)
        timeelapsed = self.framenumber / fps
        projectedFinish = (self.waitTime/fps) + timeelapsed
        
        #Attempt to use default tracker
        ret, bbox = self.tracker.update(frame)
        
        #If default tracker succeeds, store the prediction and return it
        if ret:
            if len(self.previousPredictions) >= 10:
                del self.previousPredictions[0]
            self.previousPredictions.append((bbox[0], bbox[1], bbox[2], bbox[3], timeelapsed))
            self.framenumber += 1
            return self.framenumber - 1, bbox
        else:
            #Use linear regression to predict trajectory
            #proccess data of results of all past frames into dataframe
            df = pandas.DataFrame(self.previousPredictions, dtype=float)
            Time = df[[4]]
            X = df[0]
            Y = df[1]
            
            #Make a linearregression model for both coordinates in relation to time, and fit corresponding data
            Xpredictor = linear_model.LinearRegression()
            Ypredictor = linear_model.LinearRegression()
            Xpredictor.fit(Time, X)
            Ypredictor.fit(Time, Y)

            #predict the coordinates at the projected time to finish, and create a new box with the results
            predictedX = np.round(Xpredictor.predict([[projectedFinish]]))
            predictedY = np.round(Ypredictor.predict([[projectedFinish]]))
            newbox = list((int(predictedX), int(predictedY), int(self.previousPredictions[-1][2] * 1.5), int(self.previousPredictions[-1][3] * 1.5)))

            for i in range(self.waitTime):
                ret, frame = self.video.read()
                
            
            #reinitialize tracker
            try:
                self.tracker = cv2.TrackerKCF_create()
                self.tracker.init(frame, newbox)
            except:
                print("ERROR: Backup Tracker Failed or video ended")
                return False, False
            #return coordinates of the new box
            self.framenumber += self.waitTime
            return self.framenumber, newbox

#Function to create KCFPro object
def TrackerPro_create(tracker, video, waitTime):
    return TrackerPro(tracker, video, waitTime)

def Tracker_create(tracker_type):
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    if tracker_type in tracker_types:
        if tracker_type == 'BOOSTING':
            return cv2.legacy.TrackerBoosting_create()
        elif tracker_type == 'MIL':
            return cv2.TrackerMIL_create()
        elif tracker_type == 'KCF':
            return cv2.TrackerKCF_create()
        elif tracker_type == 'TLD':
            return cv2.legacy.TrackerTLD_create()
        elif tracker_type == 'MEDIANFLOW':
            return cv2.legacy.TrackerMedianFlow_create()
        elif tracker_type == 'GOTURN':
            return cv2.TrackerGOTURN_create()
        elif tracker_type == 'MOSSE':
            return cv2.legacy.TrackerMOSSE_create()
        elif tracker_type == "CSRT":
            return cv2.TrackerCSRT_create()

#Class for tracking multiple objects
class MultiTracker:
    def __init__(self):
        self.trackers = []

    def add(self, tracker, frame, object):
        tracker.init(frame, object.box)
        self.trackers.append(tracker)

    def addAll(self, tracker_type, frame, listofObjects):
        tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
        if tracker_type not in tracker_types:
            print(f"Tracker {tracker_type} does not exist")
            return False
        for object in listofObjects:
            tracker = Tracker_create(tracker_type)
            tracker.init(frame, object.box)
            self.trackers.append(tracker)
        return True
        
    def update(self, frame):
        boxes = []
        success = True
        for tracker in self.trackers:
            ok, box = tracker.update(frame)
            if ok:
                boxes.append(box)
            else:
                success = False
        return success, boxes

#Function to create MultiTracker object
def MultiTracker_create():
    return MultiTracker()



#Class for objects
class Object:
    def __init__(self, class_id, box, ID):
        self.class_id = class_id
        self.box = box
        self.ID = ID

# Function to return list of objects classification and boxes from a image or frame with an object detection model.
def getObjects(DetectionModel, image, allowedClassifications = None, confidenceThreshold = 0.5):
    objects = []
    object_id = 1

    #generate info on all predictions 
    class_ids, confidences, boxes = DetectionModel.detect(image, nmsThreshold=0.4)

    #Iterate through objects by their corresponding data
    for class_id, confidence, box in zip(class_ids, confidences, boxes):

        #If The confidence for a classification meets the minimum threshold to determine its classifciation, and it is part of the specified classifications, add it to list of objects.
        if confidence > confidenceThreshold:
            if allowedClassifications:
                if class_id in allowedClassifications:
                    objects.append(Object(class_id, box, object_id))
                    object_id += 1
            else:
                objects.append(Object(class_id, box, object_id))
                object_id += 1
    return objects

#Function to draw bounding box around an object
#note object param can either be an object, or box coordinates
def drawBox(frame, object, color = (255,0,0)):

    # Generate values for rectangle
    if type(object) == Object:
            p1 = (int(object.box[0]), int(object.box[1]))
            p2 = (int(object.box[0] + object.box[2]), int(object.box[1] + object.box[3]))
    else:
        p1 = (int(object[0]), int(object[1]))
        p2 = (int(object[0] + object[2]), int(object[1] + object[3]))

    #Try to draw rectangle to frame
    try:
        cv2.rectangle(frame, p1, p2, color, 2, 1)
    except:
        print("Error Drawing box at frame")
        return False
    finally:
        return True

#note object param can either be an object, or box coordinates
def drawAllBoxes(frame, listofObjects, color = (255,0,0)):
    for object in listofObjects:
        if type(object) == Object:
            p1 = (int(object.box[0]), int(object.box[1]))
            p2 = (int(object.box[0] + object.box[2]), int(object.box[1] + object.box[3]))
        else:
            p1 = (int(object[0]), int(object[1]))
            p2 = (int(object[0] + object[2]), int(object[1] + object[3]))
        try:
            cv2.rectangle(frame, p1, p2, color, 2, 1)
        except:
            print("Error Drawing box at frame")
            return False
    return True


#function to quickly create a object detection/classification model
def ObjectDetection_create(weightsPath, cfgPath):
    #Read weights and cfg file and create detection model
    dnn = cv2.dnn.readNet(weightsPath, cfgPath)
    model = cv2.dnn_DetectionModel(dnn)
    model.setInputParams(size=(832, 832), scale=1 / 255)
    return model

#Determine how much time has elapsed in a video
def timeElapsed(video, framenumber):
    fps = video.get(cv2.CAP_PROP_FPS)
    timeelapsed = framenumber / fps
    return timeelapsed
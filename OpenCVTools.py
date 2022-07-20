import cv2
import pandas
from sklearn import linear_model
import numpy as np

#Class for Tracker Pro
class TrackerData:
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
        fps = self.video.get(cv2.CAP_PROP_FPS)
        timeelapsed = self.framenumber / fps
        projectedFinish = (self.waitTime/fps) + timeelapsed
        
        ret, bbox = self.tracker.update(frame)
        
        if ret:
            if len(self.previousPredictions) >= 10:
                del self.previousPredictions[0]
            self.previousPredictions.append((bbox[0], bbox[1], bbox[2], bbox[3], timeelapsed))
            self.framenumber += 1
            return self.framenumber - 1, bbox
        else:
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
    return TrackerData(tracker, video, waitTime)

#Class for tracking multiple objects
class MultiTracker:
    def __init__(self):
        self.trackers = []
    def add(self, tracker, frame, box):
        tracker.init(frame, box)
        self.trackers.append(tracker)
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

# Function to return list of objects classification and boxes from a image or frame with an object detection model.
def getObjects(DetectionModel, image, allowedClassifications = None, confidenceThreshold = 0.5):
    objects = []
    class_ids, confidences, boxes = DetectionModel.detect(image, nmsThreshold=0.4)
    for class_id, confidence, box in zip(class_ids, confidences, boxes):
        if confidence > confidenceThreshold:
            if allowedClassifications:
                if class_id in allowedClassifications:
                    objects.append([class_id, box])
            else:
                objects.append([class_id, box])
    return objects

#Function to draw bounding box around an object
def drawBox(frame, box, color = (255,0,0)):
    p1 = (int(box[0]), int(box[1]))
    p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
    try:
        cv2.rectangle(frame, p1, p2, color, 2, 1)
    except:
        print("Error Drawing box at frame")
        return False
    finally:
        return True

#function to quickly create a object detection/classification model
def ObjectDetection_create(weightsPath, cfgPath):
    dnn = cv2.dnn.readNet(weightsPath, cfgPath)
    model = cv2.dnn_DetectionModel(dnn)
    model.setInputParams(size=(832, 832), scale=1 / 255)
    return model
import cv2
from handdetector import hand_detector_utils
import fistclassifier as fclasifier
#import winsound  use on Windows to make a beep sound after detection


def Detecthand(image_bgr,tfgraph,tfsession,threshold):
    "Detect the hand, based on the highest score of SSD anchor boxes which is above the threshold"
    #Args:
    #-image_bgr: captured image (OpenCV BGR image)
    #-tfgraph: TensorFlow model (graph) of the trained SSD network hand detector
    #-tfsession: TensorFlow session for running inference
    #-threshold: the threshold value
    #Returns:
    #the tuple (True,Box), where Box is (top, left, bottom, right) of the box with highest score which is above the threshold (Dec_thresh_high)
    #,and returns (False,None) if none of the boxes has a score above the threshold


    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    boxes, scores = hand_detector_utils.detect_objects(image_rgb, tfgraph, tfsession)

    if max(scores)>threshold:

        maxsc=0

        for i in range(0, len(scores)):
            if scores[i] > maxsc:
                #(top, left, bottom, right)
                best_box=(boxes[i][0], boxes[i][1],
                                              boxes[i][2], boxes[i][3])
                maxsc = scores[i]

        return best_box,boxes,scores

    else:
        return None,boxes,scores



class SignDetector:
    "Class for detecting signals made by hand. Here a signal means closing and opening the hand several times.\
    Objects of this class have memory."

    count=0 #the number of open/close transitions so far
    time0=0 #the first time at which the classifier has detected a closed fist
    time1=0 #the last time at which the classifier has detected a open/close transition
    state_close=False #whether the current state is 'closed' (true) or 'open' (false)
    enoughcount=4 #the rquired number of transitions to count as a complete signal
    maxdelay=1000 #the maximum allowed time between trasnistions (ms)
    maxspeed=0.002 #the maxmimum valid rate of transistions per ms

    def Reset(self,time):
        "Resets the counters and state. 'time' must be the current time"
        self.count=0
        self.time0 = time
        self.time1 = time
        self.state_close=False

    def Detect(self, image, time:float):
        #Check if a signal has happend, according to the current frame and history from the past ones
        #Args:
        #-image: the image of the hand at the current frame (OpenCV bgr image)
        #-time: current time (milliseconds)
        #Returns:
        # True if a signal has been detected, and False otherwise


        pclass=fclasifier.Query(image) #run the fist classifier

        if self.count == self.enoughcount : #if number of transitions has reached the threshold
            #winsound.Beep(1500,300) use on Windows to make a beep sound after detection
            self.Reset(time)
            print("Signal detected!")
            return True

        if self.count==0 and pclass[0]>pclass[1]: #if this is the first detected closed fist
            self.count=1
            self.time0=time
            self.time1=time
            self.state_close=True
            return False
        elif time-self.time1>self.maxdelay:
            self.Reset(time)
            return False

        if self.state_close==True and pclass[1]>0.6: #if the current state is 'closed' and the classifier has returned a higer prob for 'open'
            if (time-self.time1)>300:
                self.count+=1
                self.state_close = False
                print("signal counter: " + str(self.count))
            self.time1 = time

        elif self.state_close==False and pclass[0]>0.6:
            if time - self.time1 > 300:
                self.state_close = True
                self.count+=1
                print("signal counter: " + str(self.count))
            self.time1 = time

        return False #a signal has not been detected
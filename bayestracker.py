import numpy as np

class Tracker:
    "Tracking a single object, based on the outputs of the object detector network (boxes-scores) and Bayesian inference"

    Cur_state=[] #current state (bounding box)
    Motion_var=[2.66e-4,2.66e-4,2.66e-4,2.66e-4] #the prior variance (uncertainty) that is assumed for modeling the motion
    Nboxes=100 #number of anchor boxes of the detector network

    def __init__(self,box0:np.ndarray):
        self.Cur_state=box0
        self.Cur_var=np.array(self.Motion_var) #the variance represnting the uncertainty regrading the current state

    def find_most_probable(self,curstate,curvar,boxes:list,scores:list)->int:
        "Find the most probable bounding box, based on combining observation and state (Bayesian)"
        #Args:
        #curstate, curvar: current state and variance
        #boxes, scores: the observation (current boxes and their scores, as produced by the object detector network)
        #Returns:
        #The index of the bounding box which is the most likely to contain the target object

        maxi=-1
        maxprob=0

        for i in range(0,self.Nboxes):
                m=((boxes[i]-curstate)**2)/(2*curvar)
                prob=np.log(scores[i])-m.sum()
                if prob>maxprob or maxi==-1:
                    maxprob=prob
                    maxi=i

        if maxprob>-10:
            return maxi
        else:
            return -1

    def Track(self,boxes:list,scores:list)->list:
        "Track the object. This function takes the outputs of an object detector model, updates the internal state of the tracker object, and returns the new coordinates of the bounding box"
        #Args:
        #-boxes: the list of the coordiantes of boxes obtained from the object detector network (each box is in the format [left,top,width,high])
        #-scores: the scores associated with the boxes
        #Returns:
        #the bounding box of the target object ([left,top,width,high])

        sm=sum(scores)
        scores=[s/sm for s in scores]
        imax=self.find_most_probable(self.Cur_state,self.Cur_var,boxes,scores)
        if imax>-1:
            self.Cur_state=boxes[imax]
            self.Cur_var=(1-scores[imax])*self.Cur_var+self.Motion_var
        else:
            self.Cur_var =  self.Cur_var + self.Motion_var

        return self.Cur_state


def prepareboxes(boxes:list)->list:
    "Convert bounding boxes from [top, left, bottom, right] format to [left,top,width,high]"

    nb=len(boxes)
    cboxes=[[]]*nb
    for i in range(0,nb):
        l=boxes[i][1]
        t=boxes[i][0]
        w=boxes[i][3]-l
        h=boxes[i][2]-t
        cboxes[i]=np.array([l,t,w,h])

    if nb==1:
        return cboxes[0]
    else:
        return cboxes
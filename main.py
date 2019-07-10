# AiryDraw!
# A simple augmented reality program implemented with Python, TensorFlow, and OpenCV
# Draw using your hand on the air, and have fun!
# https://github.com/amirhossein-ahmadian/airydraw
# MIT License
#
#Hand detection SSD model from
#Victor Dibia, HandTrack
#https://github.com/victordibia/handtracking



import cv2
import detector
from handdetector import hand_detector_utils
import datetime
import bayestracker as btrack
import fistclassifier as fclasifier



CAM_DEVICE_INDEX=0 #index of camera device
PRE_RESOLUTION=(180,320) #desired resolution for capturing video (may differ from real resolution of camera)
DETECT_THRESH=0.5 #minimum score for detecting hand directly (high confidence detection)
DRAWING_COLOR=(0,0,255) #color of drawing


def Run():
    "The main procedure"

    # Load the trained model of the fist classifier
    fclasifier.LoadModel("fist_network")
    print("Fist Classifier loaded.")
    #Load the TensorFlow model of hand detector
    hand_detect_graph, hand_detect_sess = hand_detector_utils.load_inference_graph()


    #Initialize the signal detector object
    sgd=detector.SignDetector()
    sgd.Reset(0)
    sgd_stime = datetime.datetime.now()

    canvas = [] #the canvas (storing a set of trajectories)

    seen=False #hand detected at least once
    dsign=False #a signal has been detected

    # Prepare for capturing video from webcam
    cap = cv2.VideoCapture(CAM_DEVICE_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PRE_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PRE_RESOLUTION[0])
    imagesize = (cap.get(4), cap.get(3))
    cv2.namedWindow('Write on Air!', cv2.WINDOW_NORMAL)
    print("Video capturing initialized.")

    state = 0
    #states:
    # 0: not drawing + direct detection with high score (not tracking)
    # 1: not drawing + tracking
    # 2: drawing + tracking

    #The main loop
    while True:

        #Capture image and flip it
        _, image_bgr = cap.read()
        image_bgr = cv2.flip(image_bgr, 1)

        #Query the signal detector to see if a signal (several hand closing/opening) has happened if hand is in the image
        if seen:
            now = (datetime.datetime.now() - sgd_stime).total_seconds() * 1000
            dsign = sgd.Detect(image_bgr[curbox.top:curbox.bottom, curbox.left:curbox.right, :],
                           now)

        # Run hand detector
        det_nbox,boxes,scores = detector.Detecthand(image_bgr, hand_detect_graph, hand_detect_sess, DETECT_THRESH)
        deth=(det_nbox is not None)

        if state==0:

            if dsign: #if a signal is detected, start drawing and tracking
                state=2
                canvas.append(([],[]))
                tracker = btrack.Tracker(btrack.prepareboxes([last_det_nbox]))  # initialize the tracker
                print("Drawing!...")
            elif deth: #if there is a direct (high-confidence) detection, update the bounding box
                curbox=intbox(det_nbox[0]*imagesize[0],det_nbox[1]*imagesize[1],det_nbox[2]*imagesize[0],det_nbox[3]*imagesize[1])
                seen=True
                last_det_nbox=det_nbox
            elif seen: #if the hand may be visible, but not directly detectable, try tracking
                tracker = btrack.Tracker(btrack.prepareboxes([last_det_nbox])) #initialize the tracker
                state=1
            bcolor = (0, 255, 0) #color of the bounding box in this state

        elif state == 1 or state==2:

            #track the hand, using the outputs of hand detector network and current detector object
            #boxes, scores = hand_detector_utils.detect_objects(imagergb, hand_detect_graph, hand_detect_sess)
            rect=tracker.Track(btrack.prepareboxes(boxes),scores)
            curbox=intbox(rect[1]*imagesize[0],rect[0]*imagesize[1],(rect[1]+rect[3])*imagesize[0],(rect[0]+rect[2])*imagesize[1])

            # if state is 2 (tracking with drawing), append the coordinates of the center of the bounding box to the current trajectory
            if state == 2:
                AppendToTraj(canvas[-1][0], canvas[-1][1], (curbox.left + curbox.right) / 2.0,
                             (curbox.top + curbox.bottom) / 2.0)

            if dsign and state==2: #change the state in case of detecting a signal (several hand opening/closing)
                state = 0
                print("Not drawing")
            elif dsign and state==1:
                canvas.append(([], []))
                state=2
                print("Drawing!...")
            elif state==1 and deth: #stop tracking if the hand is directly detectable
                curbox=intbox(det_nbox[0]*imagesize[0],det_nbox[1]*imagesize[1],det_nbox[2]*imagesize[0],det_nbox[3]*imagesize[1])
                state=0

            #set the color of the bounding box
            if state==1:
                bcolor=(0, 200, 0)
            if state==2:
                bcolor=(255, 0, 0)

        #render the trajectories (canvas) and the bounding box on the frame
        DrawCanvas(image_bgr, canvas,DRAWING_COLOR)
        if seen:
            cv2.rectangle(image_bgr, (curbox.left, curbox.top), (curbox.right, curbox.bottom), bcolor, 2, 1)

        #show the current frame
        cv2.imshow('Write on Air!', image_bgr)

        #scan for key press
        k = cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            print("Terminated.")
            cv2.destroyAllWindows()
            exit(0)
        if k & 0xFF == ord('c'):
            canvas = [([], [])]
            seen = False
            state = 0
            print("Cleared.")
        if k & 0xFF == ord('h'):
            seen=False
            state=0
            print("Tracking halted.")



def AppendToTraj(traj_x,traj_y,newpoint_x,newpoint_y):
    "Append a new point to a trajectory, with smoothing"
    #Args:
    #-traj_x,traj_y: x/y coordinates of current points in the trajectory
    #-newpoint_x, newpoint_y: x/y coordinates of the point being added
    #Updates:
    #-traj_x,traj_y

    wsize=2
    tl=len(traj_x)
    if tl>wsize:
        traj_x.append((sum(traj_x[tl-wsize-1:tl-1])+newpoint_x)/(wsize+1))
        traj_y.append((sum(traj_y[tl - wsize - 1:tl-1]) + newpoint_y) / (wsize + 1))
    elif tl>0:
        traj_x.append((sum(traj_x)+newpoint_x)/(tl+1))
        traj_y.append((sum(traj_y)+newpoint_y)/(tl + 1))
    else:
        traj_x.append(newpoint_x)
        traj_y.append(newpoint_y)

def DrawCanvas(image,canvas,color):
    "Draws all of the trajectories existing in a canvas on an image"
    #Args:
    #-image: the image (OpenCV bgr)
    #-canavs: a list of trajectories. each trajectory is a list of points, which are connected to each other
    #-color: color of the drawn lines
    #Updates:
    #-image

    for traj in canvas:
        px=traj[0]
        py=traj[1]
        for t in range(1,len(traj[0])):
            cv2.line(image,(int(px[t-1]),int(py[t-1])),(int(px[t]),int(py[t])),color,2)

class intbox:
    "Bounding box with integer coordinates"
    def __init__(self,_top,_left,_bottom,_right):
        self.left=int(_left)
        self.right=int(_right)
        self.bottom=int(_bottom)
        self.top=int(_top)


Run() #Run the main loop





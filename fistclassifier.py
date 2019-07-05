import tensorflow as tf
import cv2 as cv
import numpy as np
import os


Res=55 #Dimensions of images (Res*Res) which are fed into the neural network
Nsyn=5 #Number of generated images per each real training image, for data augmentation

#Currently loaded Tensorflow session and graph (input/output variables)
Loadedses=None
Loadedout=None
Loadedinp=None


def CreateNet(indata,name,istrainig):
    "Constructing TensorFlow computational graph of the feedforward neural network"
    #Args:
    #-indata: input data vectors, which are fed into the network
    #-istraining: must be True if the model is being trained (when it is False, the values of weights are loaded instaed of being intitialized)


    with tf.variable_scope(name,not istrainig):

        c1=tf.layers.conv2d(indata,20,7,(3,3))
        p1=tf.layers.max_pooling2d(c1,2,2)

        c2 = tf.layers.conv2d(p1, 20, 4, (3,3))
        p2 = tf.layers.max_pooling2d(c2, 2, 2)

        f1=tf.layers.dense(tf.layers.flatten(p2),70,tf.nn.sigmoid)
        f1d=tf.layers.dropout(f1,0.15,training=istrainig)

        f2 = tf.layers.dense(f1d, 70, tf.nn.sigmoid)
        f2d = tf.layers.dropout(f2, 0.15, training=istrainig)

        y=tf.layers.dense(f2d,2,tf.nn.softmax)

        return y

def Resample(image,nnew):
    "Resampling the data points (generating new images) by flipping and adding noise to image, to augment the dataset"
    #Args:
    #-image: the original image
    #-nnew: desired number of new images
    #Returns:
    # a list of generated images

    sams=[]

    sams.append(image)
    sams.append(np.fliplr(image))
    for i in range(0,nnew):
        gn = np.random.normal(0, 0.1)
        sams.append(cv.add(image,gn))

    return sams


def Train(pospath,numpos,negpath,numneg,savepath):
    "Training the neural network"
    #Args:
    #-pospath: the path of positive (fist) example image files
    #-negpath: the path of negative example image files
    #-numpos: number of positive examples
    #-numneg: number of negative examples
    #-savepath: the path where the files of the trained model will be saved


    ntot=(numpos+numneg)*(Nsyn+2)
    data=np.zeros((ntot,Res,Res,1))
    q=0

    hotlabels=[]

    for f in os.listdir(pospath):
        im=cv.imread(os.path.join(pospath,f))
        im=cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        for t in Resample(im,Nsyn):
            data[q,:,:,0]=cv.resize(t,(Res,Res))
            q+=1
            hotlabels.append([1,0])
    for f in os.listdir(negpath):
        im=cv.imread(os.path.join(negpath,f))
        im=cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        for t in Resample(im,Nsyn):
            data[q,:,:,0]=cv.resize(t,(Res,Res))
            q+=1
            hotlabels.append([0,1])

    data=tf.convert_to_tensor(data)

    netout=CreateNet(data,'net1',True)
    objfun=tf.losses.softmax_cross_entropy(hotlabels,netout)
    #opt=tf.train.GradientDescentOptimizer(0.1).minimize(objfun)
    opt=tf.train.AdamOptimizer().minimize(objfun)

    with tf.Session() as s:
        s.run(tf.initialize_all_variables())
        for i in range(0,800):
            _,er=s.run([opt,objfun])
            print("iter " + str(i))
            print(er)

        sav=tf.train.Saver()
        sav.save(sess=s,save_path=savepath)


def Query(image_bgr,curses:tf.Session=None,netin=None,netout=None):
    "Feeding an input image to the neural network (TensorFlow model), and getting the output (scores of classes)"
    #Args:
    #-image_bgr: the input image (in BGR colorspace)
    #-curses: the TensorFlow session
    #-netin/netout: the variables which have the role of the input/output of the network in the TensorFlow computational graph


    if curses is None:
        curses=Loadedses
        netin=Loadedinp
        netout=Loadedout
    x=np.zeros((1,Res,Res,1))
    imageg=cv.cvtColor(image_bgr,cv.COLOR_BGR2GRAY)
    x[0,:,:,0]=cv.resize(imageg,(Res,Res))
    y=curses.run(netout,{netin:x})

    return (y[0,0],y[0,1])


def LoadModel(path):
    "Load the trained model from file"

    x=tf.placeholder(tf.double,(1,Res,Res,1))
    netout=CreateNet(x,'net1',False)
    global Loadedses, Loadedinp, Loadedout

    s=tf.Session()
    sav=tf.train.Saver()
    sav.restore(s,path)

    Loadedses=s
    Loadedinp=x
    Loadedout=netout
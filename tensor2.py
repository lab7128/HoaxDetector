import tflearn
from tqdm import tqdm
import cv2
import os
import os.path
import numpy as np
from random import shuffle
import tensorflow as tf

tf.reset_default_graph()

TRAIN_DIR='C:\\Users\\Lenovo\\Desktop\\BharathiProjects\\Testdata_SigComp2011 - Copy\\SigComp11-Offlinetestset\\Dutch\\TrainSig'
#TRAIN_DIR consists of 1st 400 forged and 1st 400 real images
TEST_DIR='C:\\Users\\Lenovo\\Desktop\\BharathiProjects\\Testdata_SigComp2011 - Copy\\SigComp11-Offlinetestset\\Dutch\\TestSig'
#TEST_DIR consists of the rest of the images
#we change it later :|
IMG_SIZE=200
#LR=le-3 #0.001
LR=0.001

MODEL_NAME='RealvsForge2.model'.format(LR, '6conv-basic-video') #2 convusional layers


#get features

#convert label to 1hot  array i.e. real or forge

def label_img(img):
    #this function is run only once. creates numpy arrays of all the train images
    word_label = img[0]
    if word_label=='G': 
        return [1,0]
    elif word_label=='F':
        return [0,1]
def create_train_data():
    training_data=[]
    for img in tqdm(os.listdir(TRAIN_DIR)):
        
        label=label_img(img)
        path=os.path.join(TRAIN_DIR,img)
        
        
        img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE), (IMG_SIZE,IMG_SIZE))
       
        
        #print(img)
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy',training_data)
    return training_data
    


def process_test_data():
    testing_data=[]
    for img in tqdm(os.listdir(TEST_DIR)):
        path=os.path.join(TEST_DIR,img)
        img_num=label_img(img)
        img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE), (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img),img_num])
    np.save('test_data.npy',testing_data)
    return testing_data
train_data=create_train_data()
#if we already have train data: train_data=np.load('train_data.npy')






from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression



convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu') #for linear problems
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu') #non linear problems
convnet = max_pool_2d(convnet, 2)


convnet = conv_2d(convnet, 32, 2, activation='relu') #for linear problems
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu') #non linear problems
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='1ogdir')

'''if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded')'''

#training:
train=train_data[0:825] #1st 825 - forged and 1st 825 of genuine \
train += train_data[1650:2475]
test=train_data[825:1650]
test+=train_data[2475:]
#feature set x
X=np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y=[i[1] for i in train] #THESE 2 LINES USED FOR FITTING

test_x=np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE,1)
test_y=[i[1] for i in test] #these are for testing accuracy. not the actual testing data. 

model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}),snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

#tensorboard --logdir=foo:C:\Users\Lenovo\Desktop\BharathiProjects\log --host=127.0.0.1
logdir='C:\\Users\\Lenovo\\Desktop\\BharathiProjects\\log'

'''with tf.Session() as sess:
    writer=tf.summary.FileWriter(logdir,sess.graph)
    result=sess.run
    print('outcomes',result)
writer.close()'''

model.save(MODEL_NAME)

import matplotlib.pyplot as plt
#if we dont have it
test_data=process_test_data()
#shuffle(test_data)
#if we have it
#test_data=np.load('test_data.npy')

fig=plt.figure()

for num,data in enumerate(test_data[-20:]):
    #g is 1,0 and F is 0,1
    
    img_num=data[1]
    img_data=data[0]
    
    y=fig.add_subplot(5,5,num+1) #subplot grid of 3 by 4
    orig=img_data
    data=img_data
    data=img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    
    model_out= model.predict([data])[0] #output model predicts. 0th is what we want
    
    if np.argmax(model_out) == 1: str_label='Genuine'+str(img_num)+" "
    else: str_label='Fake'+str(img_num)
    
    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()

#with open('submissiom-file.csv','w') as f:
        #f.write('id, label\n')
#with open('submission-file.csv','s') as f:
        #for data in tqdm(test_data):
'''img_num=data[1][0]
img_data=data[0]


orig=img_data
data=img_data
data=img_data.reshape(IMG_SIZE,IMG_SIZE,1)

model_out= model.predict([data])[0]'''
        #f.write('{},{}\n',format(img_num,model_out[1]))
                
                
        
    
    
    
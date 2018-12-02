import os
import math
import cv2
import numpy as np

def pad(im,padx1,padx2,pady1,pady2):
    row, col= im.shape[:2]
    border=cv2.copyMakeBorder(im, top=pady1, bottom=pady2, left=padx1, right=padx2, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])
    return border


Ffiles=os.listdir('C:\\Users\\Lenovo\\Desktop\\BharathiProjects\\Testdata_SigComp2011 - Copy\\SigComp11-Offlinetestset\\Dutch\\Reference(646)')
for i in Ffiles:
    files=os.listdir('C:\\Users\\Lenovo\\Desktop\\BharathiProjects\\Testdata_SigComp2011 - Copy\\SigComp11-Offlinetestset\\Dutch\\Reference(646)\\'+i)
    j=1
    for f in files:
        demo=str(i+'\\')
        img=cv2.imread('C:\\Users\\Lenovo\\Desktop\\BharathiProjects\\Testdata_SigComp2011 - Copy\\SigComp11-Offlinetestset\\Dutch\\Reference(646)\\'+demo+f,1)
        fname='G'+i+'0'+str(j)+'.png'
        cv2.imwrite('C:\\Users\\Lenovo\\Desktop\\BharathiProjects\\Testdata_SigComp2011 - Copy\\SigComp11-Offlinetestset\\Dutch\\All\\'+fname,img)
        j+=1
    
    
maxheight=0
maxwidth=0
minheight=999
minwidth=999
    
Ffiles=os.listdir('C:\\Users\\Lenovo\\Desktop\\BharathiProjects\\Testdata_SigComp2011 - Copy\\SigComp11-Offlinetestset\\Dutch\\All')
for sign in Ffiles:
    img=cv2.imread('C:\\Users\\Lenovo\\Desktop\\BharathiProjects\\Testdata_SigComp2011 - Copy\\SigComp11-Offlinetestset\\Dutch\\All\\'+sign,1)
    height,width=img.shape[:2]
    if height>maxheight:
        maxheight=height
        s1=sign
    if width>maxwidth:
        maxwidth=width
        s2=sign
    if height<minheight:
        minheight=height
        s3=sign
    if width<minwidth:
        minwidth=width
        s4=sign

print("MaX")
print(maxheight)
print(maxwidth)    
print(s1+"  "+s2)

print("Min")
print(minheight)
print(minwidth)    
print(s1+"  "+s2)


Ffiles=os.listdir('C:\\Users\\Lenovo\\Desktop\\BharathiProjects\\Testdata_SigComp2011 - Copy\\SigComp11-Offlinetestset\\Dutch\\All')
for sign in Ffiles:
    img=cv2.imread('C:\\Users\\Lenovo\\Desktop\\BharathiProjects\\Testdata_SigComp2011 - Copy\\SigComp11-Offlinetestset\\Dutch\\All\\'+sign,1)
    height,width=img.shape[:2]
    if height>width:
        x=(200/height)
        nw=int(math.ceil(width*x))
        img=cv2.resize(img,(nw,200))
    else:
        x=(200/width)
        nh=int(math.ceil(height*x))
        img=cv2.resize(img,(200,nh))
    cv2.imwrite('C:\\Users\\Lenovo\\Desktop\\BharathiProjects\\Testdata_SigComp2011 - Copy\\SigComp11-Offlinetestset\\Dutch\\All - Copy\\'+'Resized'+sign,img)

#Run twice for proper results       
padx1=0
padx2=0
pady1=0
pady2=0

maxwidth=maxheight=200
Ffiles=os.listdir('C:\\Users\\Lenovo\\Desktop\\BharathiProjects\\Testdata_SigComp2011 - Copy\\SigComp11-Offlinetestset\\Dutch\\All - Copy')
for sign in Ffiles:
    img=cv2.imread('C:\\Users\\Lenovo\\Desktop\\BharathiProjects\\Testdata_SigComp2011 - Copy\\SigComp11-Offlinetestset\\Dutch\\All - Copy\\'+sign,1)
    height,width=img.shape[:2]
    
    if width<200:
        if((maxwidth-width)%2==0):
            padx1=int((maxwidth-width)/2)
            padx2=padx1
        elif maxwidth==width:
            padx1=padx2=0
        else:
            padx2=int(math.ceil(maxwidth-width)/2)-1
            padx1=int(math.ceil(maxwidth-width)/2)
    else:
        padx1 = 0
        padx2 = 0
    
    if height<200:
        if((maxheight-height)%2==0):
            pady1=int((maxheight-height)/2)
            pady2=pady1
        elif maxheight==height:
            pady1=pady2=0
        else:
            pady2=int(math.ceil(maxheight-height)/2)-1
            pady1=int(math.ceil(maxheight-height)/2)
    else:
        pady1 = 0
        pady2 = 0
    
    padded=pad(img,padx1,padx2,pady1,pady2)        
    cv2.imwrite('C:\\Users\\Lenovo\\Desktop\\BharathiProjects\\Testdata_SigComp2011 - Copy\\SigComp11-Offlinetestset\\Dutch\\All-copy3\\Padded'+sign,padded)
c=0
Ffiles=os.listdir('C:\\Users\\Lenovo\\Desktop\\BharathiProjects\\Testdata_SigComp2011 - Copy\\SigComp11-Offlinetestset\\Dutch\\All - Copy')
for sign in Ffiles:
    img=cv2.imread('C:\\Users\\Lenovo\\Desktop\\BharathiProjects\\Testdata_SigComp2011 - Copy\\SigComp11-Offlinetestset\\Dutch\\All - Copy\\'+sign,1)
    height,width=img.shape[:2]
    if height != 200 or width != 200:
        #print(str(height)+" "+str(width)+" "+str(sign))
        c+=1
    '''if height<200:
        img=pad(img,padx1,padx2,1,1)'''
        
Ffiles=os.listdir('C:\\Users\\Lenovo\\Desktop\\BharathiProjects\\Testdata_SigComp2011 - Copy\\SigComp11-Offlinetestset\\Dutch\\All-copy3')
for sign in Ffiles:
    img=cv2.imread('C:\\Users\\Lenovo\\Desktop\\BharathiProjects\\Testdata_SigComp2011 - Copy\\SigComp11-Offlinetestset\\Dutch\\All-copy3\\'+sign,1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    cv2.imwrite('C:\\Users\\Lenovo\\Desktop\\BharathiProjects\\Testdata_SigComp2011 - Copy\\SigComp11-Offlinetestset\\Dutch\\All - Copy\\Bin'+sign,threshed)





Ffiles=os.listdir('C:\\Users\\Lenovo\\Desktop\\BharathiProjects\\Testdata_SigComp2011 - Copy\\SigComp11-Offlinetestset\\Dutch\\All - Copy')
for sign in Ffiles:
    img=cv2.imread('C:\\Users\\Lenovo\\Desktop\\BharathiProjects\\Testdata_SigComp2011 - Copy\\SigComp11-Offlinetestset\\Dutch\\All - Copy\\'+sign,1)
    
    for i in range(0,len(sign)):
        if sign[i]=='F' or sign[i]=='G':
            s=sign[i:]
    cv2.imwrite('C:\\Users\\Lenovo\\Desktop\\BharathiProjects\\Testdata_SigComp2011 - Copy\\SigComp11-Offlinetestset\\Dutch\\All - Copy\\'+s,img)
 #All - Copy contains final pre processed images after rezing, padding, resizing again and convertion to binary   
  
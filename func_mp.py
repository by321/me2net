import os, sys, threading
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageOps

def GetFaceMask(theCtx:dict,img:Image) -> Image:
    maskImg=Image.new('L',size=img.size,color=0)

    haar=theCtx['haar_cascade']
    #haar=GetHaarCascade()
    cv_img = np.array(img.convert("L"))
    with theCtx['cascade_classifier_lock']:
        faces_rect = haar.detectMultiScale(cv_img, scaleFactor=1.1, minNeighbors=3,minSize=(64,64))

    landmarker=theCtx['mp_face_landmarker']
    vidx = theCtx['mp_face_oval']
    nFound:int=0
    face_scale=theCtx['face_scale']
    for x, y, w, h in faces_rect: #open cv Rect coordinates are [inclusive,exclusive)
        x0,x1=int(x-w/3), int(x+w+w/3)
        y0,y1=int(y-h/2), int(y+h+h/3)
        #print(f"haar cascade found face: {x0},{y0} {x1},{y1}")
        imgcrop=img.crop((x0,y0,x1,y1))
        #imgcrop.save(f"c:\\temp\\found{nFound}.png")
        image=mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(imgcrop.convert('RGB')))
        detection_result = landmarker.detect(image)
        if 0==len(detection_result.face_landmarks): continue
        
        #print("mediapipe found face")
        nFound=nFound+1
        facelandmark=detection_result.face_landmarks[0]
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in facelandmark
        ])
        vertices=[]
        sumx:float=0
        sumy:float=0
        for i in vidx:
            if (0<=i<len(face_landmarks_proto.landmark)):
                x,y= face_landmarks_proto.landmark[i].x,face_landmarks_proto.landmark[i].y
                sumx=sumx+x
                sumy=sumy+y
                vertices.append([x,y])
        ctrx=sumx/len(vertices)
        ctry=sumy/len(vertices)
        #print("center:",ctrx,ctry)
        #print(vertices)
        v2=[]
        for xy in vertices:
            x2=ctrx+(xy[0]-ctrx)*face_scale
            y2=ctry+(xy[1]-ctry)*face_scale
            v2.append( ( round(imgcrop.size[0]*x2)+x0, round(imgcrop.size[1]*y2)+y0 ) )
        ImageDraw.Draw(maskImg).polygon(v2,fill=255,outline=255)
        #ImageDraw.Draw(img).polygon(v2,fill=None,outline=255) #testing cx

    if 0==nFound:
        print("warning: no face detected",file=sys.stderr)
        fImg=maskImg
    else:
        fImg=maskImg.filter(ImageFilter.BoxBlur(1))
    if theCtx['invert_mask']: fImg=ImageOps.invert(fImg)
    return fImg
    
def GetFaceMask2(theCtx:dict,img:Image) -> Image:
    maskImg=Image.new('L',size=img.size,color=0)
    # makes no assumption about outline
    vidx = frozenset( [i for v in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for i in v] )
    image=mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(img.convert('RGB')))
    detector=theCtx['mp_face_landmarker']
    detection_result = detector.detect(image)
    #print(detection_result)
    if 0==len(detection_result.face_landmarks):
        print("warning: no face detected",file=sys.stderr)
    for facelandmark in detection_result.face_landmarks:
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in facelandmark
        ])
        vertices=[]
        sumx:float=0
        sumy:float=0
        for i in vidx:
            if (0<=i<len(face_landmarks_proto.landmark)):
                x,y= face_landmarks_proto.landmark[i].x,face_landmarks_proto.landmark[i].y
                sumx=sumx+x
                sumy=sumy+y
                vertices.append([x,y])
        ctrx=sumx/len(vertices)        
        ctry=sumy/len(vertices)
        v2=[]
        for xy in vertices:
            x2=ctrx+(xy[0]-ctrx)*1.05
            y2=ctry+(xy[1]-ctry)*1.05
            v2.append([round(img.size[0]*x2),round(img.size[1]*y2)])
        #print(ctrx,ctry)
        #print(v2)
        vertices=cv2.convexHull(np.asarray(v2)) # why does it return a [[[]...]] ?
        v2 = [xy for sublist in vertices for item in sublist for xy in item]
        #print(vertices3)
        ImageDraw.Draw(maskImg).polygon(v2,fill=255,outline=255)

    fImg=maskImg.filter(ImageFilter.BoxBlur(3))
    if theCtx['invert_mask']: fImg=ImageOps.invert(fImg)
    return fImg

def GetMediaPipeLandmarker():
    current_dir = os.path.dirname(__file__)
    full_model_path=os.path.join(current_dir,"pretrained_models","face_landmarker_v2_with_blendshapes.task")
    if not os.path.isfile(full_model_path):
        print(f"model file doesn't exist: {full_model_path}",file=sys.stderr)
        return None

    base_options = python.BaseOptions(model_asset_path=full_model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                    output_face_blendshapes=True, output_facial_transformation_matrixes=True,
                    num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    return detector

def GetMediaPipeFaceOval():
    d={}
    for xy in mp.solutions.face_mesh.FACEMESH_FACE_OVAL:
        d[xy[0]]=xy[1]
    v=[]
    for i0 in d.keys(): break
    i=i0
    while i in d:
        v.append(i)
        i=d[i]
        if i==i0: break
    if len(v)<3: return None
    #print(v)
    return v
        
def GetHaarCascade():
    current_dir = os.path.dirname(__file__)
    full_model_path=os.path.join(current_dir,"pretrained_models","haarcascade_frontalface_alt2.xml")
    if not os.path.isfile(full_model_path):
        print(f"model file doesn't exist: {full_model_path}",file=sys.stderr)
        return None

    haar_cascade = cv2.CascadeClassifier(full_model_path)
    return haar_cascade

def InitMediaPipe(theCtx:dict)->int:
    print(f"openCV thread count:",cv2.getNumThreads())

    fd=GetHaarCascade()
    if fd is None: return -1
    theCtx['haar_cascade']=fd

    ### OpenCV's cascade classifier is not multithread-safe, but may be internally multi-threaded.
    ### By default, openCV uses one thread per core, so using its classifier with a mutex is an
    ### OK approach. To get max performance, we should create a classifier for each
    ### thread, and tune the number of threads we use and openCV uses.
    theCtx['cascade_classifier_lock']=threading.Lock()

    fd=GetMediaPipeFaceOval()
    if fd is None: return -2
    theCtx['mp_face_oval']=fd
    
    fd=GetMediaPipeLandmarker()
    if fd is None: return -3
    theCtx['mp_face_landmarker']=fd
    return 0

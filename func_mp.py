import os, sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageOps


def GetFaceMask(theCtx:dict,img:Image) -> Image:
    maskImg=Image.new('L',size=img.size,color=0)
    # makes no assumption about outline
    vidx = frozenset( [i for v in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for i in v] )
    image=mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(img.convert('RGB')))
    detector=theCtx['face_detector']
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

    fImg=maskImg.filter(ImageFilter.BoxBlur(5))
    if theCtx['invert_mask']: fImg=ImageOps.invert(fImg)
    return fImg

def GetFaceDetector():
    current_dir = os.path.dirname(__file__)
    full_model_path=os.path.join(current_dir,"pretrained_models","face_landmarker_v2_with_blendshapes.task")
    if not os.path.isfile(full_model_path):
        print(f"model file doesn't exist: {full_model_path}",file=sys.stderr)
        return None

    base_options = python.BaseOptions(model_asset_path=full_model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                    output_face_blendshapes=True, output_facial_transformation_matrixes=True,
                    num_faces=10)
    detector = vision.FaceLandmarker.create_from_options(options)
    return detector


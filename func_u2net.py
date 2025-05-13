import os, sys, time, threading, queue
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# u2net is  173.6 MB full size version, u2netp is smaller version 4.7 MB
from u2net_engine import U2NET, U2NETP 

def GetU2NetModel(model_name:str) -> nn.Module:
    #The net object should be thread-safe and can be shared among threads.
    #If it's not, we can just create a separate instance in each thread.

    if(model_name=='u2net'):
        model_filename="u2net.pth"
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        model_filename="u2netp.pth"
        net = U2NETP(3,1)
    elif(model_name=='u2neths'):
        model_filename="u2net_human_seg.pth"
        net = U2NET(3,1)
    else:
        print(f"unexpected model name: {model_name}")
        return None

    current_dir = os.path.dirname(__file__)
    full_model_path=os.path.join(current_dir,"pretrained_models",model_filename)
    if not os.path.isfile(full_model_path):
        print(f"model file doesn't exist: {full_model_path}",file=sys.stderr)
        return None

    mb= int(os.stat(full_model_path).st_size/1048576+0.5)

    if torch.cuda.is_available():
        print(f"loading {full_model_path}, {mb} MB, CUDA mode ...")
        net.load_state_dict(torch.load(full_model_path))
        net.cuda()
    else:
        print(f"loading {full_model_path}, {mb} MB, CPU mode ...")
        net.load_state_dict(torch.load(full_model_path, map_location='cpu'))
    net.eval()
    return net

def GetForegroundMask(theCtx:dict,i1:Image) -> Image:
    net:nn.Module=theCtx['u2net']
    
    #image=transforms.PILToTensor()(i1.resize((320,320),Image.LANCZOS)) #use PIL to resize
    image=transforms.Resize((320,320),antialias=True)(transforms.PILToTensor()(i1)) #use torch to resize

    npmax=torch.max(image)
    if (npmax>1e-6): image = image/npmax
    else: image=image.type(torch.FloatTensor) #need to convert to float type

    if(1==image.shape[0]):
        tnorm = transforms.Normalize(mean=0.485, std=0.229)
        image=tnorm(image)
        image = image.tile((3,1,1))
    else:
        tnorm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        image=tnorm(image)

    inputs_test = image.type(torch.FloatTensor)
    if torch.cuda.is_available():
        inputs_test = inputs_test.cuda()
    else:
        pass

    inputs_test=torch.unsqueeze(inputs_test, dim=0)
    d1 = net(inputs_test)

    ma=torch.max(d1)
    mi=torch.min(d1)
    #print(f"ma {ma} mi {mi}")
    if (ma!=mi):
        d1=(d1-mi)/(ma-mi)
    if theCtx['invert_mask']: d1=1-d1
    im= transforms.ToPILImage("L")(d1)
    del d1

    return im.resize(i1.size,resample=Image.LANCZOS)

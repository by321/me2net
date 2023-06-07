import os, sys, time, threading, queue
from PIL import Image

def CommonInit(theCtx:dict):
    _PrepareBackgroundImage(theCtx)
    if theCtx['model'] in ['u2net','u2netp']:
        global func_u2net
        import func_u2net
        net=func_u2net.GetU2NetModel(theCtx['model'])
        if net is None: sys.exit(-1)
        theCtx['u2net']=net
        # function pointer: Image* (*GetForegroundMask)(Dict&,Image&)
        theCtx['GetForeGroundMask']=func_u2net.GetForegroundMask
    else:
        global func_mp
        import func_mp
        i=func_mp.InitMediaPipe(theCtx)
        if i!=0: sys.exit(i)
        
        theCtx['GetForeGroundMask']=func_mp.GetFaceMask
 
def _AdjustBackgroundImage(bgOriginal:Image,bgCached:Image,toSize,toMode)->Image:
    if bgOriginal is None: return None

    if bgCached.size != toSize: # need to resize
        bgCached=bgOriginal # resize from original image
    elif bgCached.mode != toMode: # need to change mode
        if bgOriginal.mode!=bgCached.mode: # original image in different mode
            bgCached=bgOriginal # change mode from original image
    if bgCached.size != toSize:
        #print("scaling background image...")
        bgCached=bgCached.resize(toSize,Image.LANCZOS)
    if bgCached.mode != toMode: # different modes, convert to RGB
        #print("changing mode of background image...")
        # mode will be grayscale only if both input image and background image are grayscale
        bgCached=bgCached.convert("RGB")
    return bgCached


def _PrepareBackgroundImage(theCtx:dict):
    if theCtx['background_image'] is None: 
        theCtx['bgimg_loaded']=None
        return
    if theCtx['mask_usage']!='0':
        print("warning: mask usage option is not 0, the -bi option is ignored")
        theCtx['bgimg_loaded']=None
        return

    i1 = Image.open(theCtx['background_image'])
    i1.load()
    if i1.mode != "L" and i1.mode != "RGB":
        print(f"converting {theCtx['background_image']} from mode {i1.mode} to RGB")
        i1 = i1.convert("RGB")
    theCtx['bgimg_loaded']=i1


def _SaveOutputFile(theCtx:dict,inputImg:Image,maskImg:Image,imgBG:Image,output_file:str) -> int :
    
    mu=theCtx['mask_usage']
    if (mu=='2'): #mask only
        maskImg.save(output_file)
    elif (mu=='1'): #input image + mask
        imgC=inputImg.copy()
        imgC.putalpha(maskImg)
        imgC.save(output_file)
        #print(f"{output_file} {inputImg.mode} {imgC.mode}")
    elif (mu=='0'): #alpha blend input image with a solid color or background image
        if imgBG is None:
            imgBG = Image.new('RGB', inputImg.size, theCtx['background_color'])
        #print(f"{inputImg.size} {imgBG.size}")    
        imgC=Image.composite(inputImg,imgBG,maskImg)
        imgC.save(output_file,format="PNG")
    else:
        print(f"unexpected output mode: {mu}")
        return -1
    return 0

def _GetForegroundMask(theCtx:dict,i1:Image) -> Image:
    return theCtx['GetForeGroundMask'](theCtx,i1)
    #if theCtx['model'] in ['u2net','u2netp']:
    #    return u2net_func.GetForegroundMask(theCtx,i1)
    #else:
    #    return mp_func.GetFaceMask(theCtx,i1)
    
def _LoadInputImage(input_file:str) -> Image:
    try:
        i1 = Image.open(input_file)
        if i1.mode != "L" and i1.mode != "RGB":
            print(f"    converting {input_file} from mode {i1.mode} to RGB")
            i1 = i1.convert("RGB")
    except Exception as inst:
        print(type(inst),':',inst)
        print(f"{input_file} : could not load successfully")
        return None
    return i1


def ProcessOneFile(theCtx:dict,input_file:str,output_file:str) -> int:
    i1:Image=_LoadInputImage(input_file)
    if i1 is None: return -1
    maskImg:Image=_GetForegroundMask(theCtx,i1)
    bgImg=_AdjustBackgroundImage(theCtx['bgimg_loaded'],theCtx['bgimg_loaded'],i1.size,i1.mode)
    if 0==_SaveOutputFile(theCtx,i1,maskImg,bgImg,output_file):
        print("output file saved successfully")


def _dir_worker_thread(theCtx:dict, lck:threading.Lock, input_dir:str, output_dir:str, 
                       items:list[str], idx0:int, idx1:int):
    with lck: print(f"thread {threading.get_native_id()} running ...")
    bgImg=theCtx['bgimg_loaded']
    nOK:int=int(0)
    while idx0<idx1:
        fn=items[idx0]
        idx0=idx0+1

        input_file=os.path.join(input_dir,fn)
        if not os.path.isfile(input_file): continue
        i1=_LoadInputImage(input_file)
        if i1 is None: continue

        fnout = os.path.splitext(fn)[0]+".png"
        output_file=os.path.join(output_dir,fnout)
        with lck:
            print(f"thread {threading.get_native_id()}: {input_file} => {output_file} ...")
        maskImg:Image=_GetForegroundMask(theCtx,i1)
        bgImg=_AdjustBackgroundImage(theCtx['bgimg_loaded'],bgImg,i1.size,i1.mode)
        if 0==_SaveOutputFile(theCtx,i1,maskImg,bgImg,output_file): nOK=nOK+1

    with lck: 
        print(f"thread {threading.get_native_id()} files successfully processed: {nOK}")
        theCtx['nOK']=theCtx['nOK']+nOK

def ProcessOneDirectory(theCtx:dict, input_dir:str, output_dir:str):
    if not os.path.isdir(output_dir): os.makedirs(output_dir, exist_ok=True)
    items=[x for x in os.listdir(input_dir)]
    nItems=len(items)

    nThreads=theCtx['threads']
    if (nThreads>nItems): nThreads=nItems
    theCtx['nOK']:int=int(0)
    lck=threading.Lock()
    workers = []
    idx0:int=0
    for i in range(nThreads):
        idx1:int=(nItems*(i+1))//nThreads
        wt = threading.Thread(target=_dir_worker_thread,args=[theCtx,lck,input_dir,output_dir,items,idx0,idx1])
        wt.start()
        workers.append(wt)
        idx0=idx1

    for wt in workers: wt.join()
    print(f"\ntotal files successfully processed: {theCtx['nOK']}")

def _stdin_worker_thread(q:queue.Queue, theCtx:dict, lck:threading.Lock, output_specifier:str):
    with lck: print(f"thread {threading.get_native_id()} running ...")
    while True:
        work_item:tuple(int,Image)=q.get()
        if work_item[0]==-1: # image index, -1 is used as exit signal
            break
        try:
            outfn=output_specifier % work_item[0]
            with lck: print(f"thread {threading.get_native_id()} processing img#{work_item[0]} to {outfn} ...")
            i1:Image=work_item[1]
            maskImg:Image=_GetForegroundMask(theCtx,i1)
            _SaveOutputFile(theCtx,i1,maskImg,theCtx['bgimg_loaded'],outfn)
        except Exception as e:
            with lck: print(f"thread {threading.get_native_id()} exception {type(e)}: {e}")
    with lck: print(f"thread {threading.get_native_id()} exiting ...")

def _read_piped_input(file_number:int, outBuf:bytearray, bufLen:int) -> int :
#Most likely, OS's pipe buffer is smaller than a full image, and there's no guarantee how
#many bytes are available to read at any given time, thus this complicated read procedure.
#Basically, this is like reading a TCP/IP socket.
#On Windows, it seems I could read at most 32K bytes at a time.
    bytesAlreadyRead:int=0 #how many bytes were already read
    consecutive_errors:int=0
    while True:
        byteBuf=os.read(file_number,bufLen-bytesAlreadyRead)
        if (len(byteBuf)>0): #we read some bytes
            #print(f"read {len(byteBuf)} bytes\n")
            j:int=bytesAlreadyRead+len(byteBuf) #copy what we just read into outBuf[]
            outBuf[bytesAlreadyRead:j]=byteBuf
            bytesAlreadyRead=j
            if (bytesAlreadyRead==bufLen): #yes, we got all the bytes we need
                break
            consecutive_errors=0 #reset error counter
        else: #read failed
            consecutive_errors+=1
            if consecutive_errors==6:
                break
            time.sleep(0.5) #wait for more data to get into pipe
    return bytesAlreadyRead

def ReadStdin(theCtx:dict,image_width:int,image_height:int,output_specifier:str)->int:
    """Process a sequence of RGB24 images from stdin. This is intended to be used with another program, such
    as FFMPEG, that outputs RGB24 pixel data to stdout, which is piped into the stdin of this program.

      image_width, image_height : dimension of image(s)

      output_specifier: printf-style specifier for output filenames, for example if abc%03u.png, then
        output files will be named abc000.png, abc001.png, abc002.png, etc.
        Output files will be saved in PNG format regardless of the extension specified.

    Example usage with FFMPEG:

    \b
      ffmpeg -i input.mp4 -ss 10 -an -f rawvideo -pix_fmt rgb24 pipe:1 | python rembg.py v 1280 720 out%03u.png

    The width and height values must match the dimension of output images from FFMPEG.
    Note for FFMPEG, the "-an -f rawvideo -pix_fmt rgb24 pipe:1" part is required.
    """

    # since image size and pixel format are fixed, we can adjust the
    # background image now, instead of having each thread do it
    theCtx['bgimg_loaded']=_AdjustBackgroundImage(theCtx['bgimg_loaded'],theCtx['bgimg_loaded'],
                                                  (image_width,image_height),"RGB")

    output_dir=os.path.dirname(os.path.abspath(output_specifier))
    if not os.path.isdir(output_dir): os.makedirs(output_dir, exist_ok=True)

    nThreads=theCtx['threads']
    q=queue.Queue(nThreads+1)
    lck=threading.Lock() # use this lock when printing to console
    workers = []
    for i in range(nThreads):
        wt = threading.Thread(target=_stdin_worker_thread,args=[q,theCtx,lck,output_specifier])
        wt.start()
        workers.append(wt)

    bytesPerImage:int=image_width*image_height*int(3)
    fullBuf=bytearray(bytesPerImage)
    img_index:int=0
    while True:
        bytesRead:int=_read_piped_input(sys.stdin.fileno(),fullBuf,bytesPerImage)
        if (bytesRead!=bytesPerImage):
            with lck: print(f"read stopped at image index {img_index}")
            break
        img=Image.frombytes("RGB",(image_width,image_height),bytes(fullBuf),"raw")
        q.put([img_index,img])
        img_index=img_index+1
    
    for i in range(len(workers)): q.put([-1,''])
    for wt in workers: wt.join()

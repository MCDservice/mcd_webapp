from urllib.request import urlopen
import cv2
import numpy as np


def url_to_image(url, readFlag=cv2.IMREAD_COLOR):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, readFlag)

    # return the image
    return image

def analyse_photo(Image_Path, Image_Name):
    #=============================================================================
    #                PROGRAM: USE CNN MODEL (SEMANTIC SEGMENTATION)
    #=============================================================================

    # Description:

    #=============================================================================
    #                               OPTIONS
    #=============================================================================
    # Path options
    #-----------------------------------------------------------------------------

    # lets you choose the file (if False, uses Image_Path from below)
    Image_Browser = False

    #
    Save_Output = True

    #-----------------------------------------------------------------------------

    # Image path now given as a function argument
    # Image_Path = "Input//Database//Images//M1-019.jpg"

    # Model_Path = "C://+ BACKUP//PROGRAM DATA//PYTHON//CNN - Semantic Segmentation//Output//+ Tests//+ Best//"
    # Model_Path += "Blocks2_198_epoch_97_val_F1_score_0.960.h5"
    #
    # JModel_Path = "C://+ BACKUP//PROGRAM DATA//PYTHON//CNN - Semantic Segmentation//Output//+ Tests//+ Best//"
    # JModel_Path += "Blocks2_198_(Model).json"

    # new model:
    Model_Path = "C://Users//Dom//Documents//3-Year//2021-Internship//" \
                  "pretrained-model//model for crack detection//" \
                  "142_epoch_45_f1_m_dil_0.796.h5"
    #
    # JModel_Path5 = "C://Users//Dom//Documents//3-Year//2021-Internship//" \
    #                "crack_detection_CNN_masonry//output//model_json//" \
    #                "crack_detection_44072.json"

    JModel_Path = "C://Users//Dom//Documents//3-Year//2021-Internship//" \
                   "pretrained-model//model for crack detection//" \
                   "my_dataset_VGG16_FCN_142.json"

    # Output_Path = "Output//Images"
    Output_Path = "media"

    #=============================================================================
    # Image-size options (Input and output of model)
    #-----------------------------------------------------------------------------

    # Input image-resolution of model (initial training size of model)
    # Note1: Input Format: (224,224,3), (256,256,3), (512,512,3), etc.

    # image dimensions that are passing through the network
    # (CNN, if no fully connected layer, it causes accuracy loss for high resolutions ...
    #  ... so this image_dims is the slice sizes)
    Image_Dims = (224,224,3)

    #-----------------------------------------------------------------------------

    # Adjust resolution for very large/small image sizes
    # ... if the image is adjusted in size (too large/small)
    Adjust_Size = True

    # The maximum resolution-ratio, based on the output-resolution.
    # ImDim = average(xIm,yIm); OutDim = average(xOut,yOut);
    # Limit = OutDim*ResRatio; Scale = Limit/ImDim
    MaxResRatio = 4

    # The efficiency-ratio of the adjusted-scale when downscaling.
    # Note1: A value of 0 retains the original-size
    # Note2: A value of 1 resizes the image to the limit-size.
    # AdjScale = 1-(1-Scale)*ScaleEff

    # allows some deviation (for very large images)
    MaxScaleEff = 0.75

    # The minimum resolution-ratio, based on the output-resolution.
    # ImDim = average(xImg,yImg); OutDim = average(xOut,yOut);
    # Limit = OutDim*ResRatio; Scale = Limit/ImDim
    MinResRatio = 2

    # The efficiency-ratio of the adjusted-scale when upscaling.
    # Note1: A value of 0 retains the original-size
    # Note2: A value of 1 resizes the image to the limit-size.
    # AdjScale = 1-(1-Scale)*ScaleEff
    MinScaleEff = 0.75

    #-----------------------------------------------------------------------------

    # Overlap value of image slices
    # how many pixels overlap (to retain only the predictions from the middle layer ...
    # ... not from the overlap - increases accuracy)
    # (if overlap is not set, then might not output anything)

    Overlap = 50

    # Padding value
    # image padded before passed through the network - 1000/244 needs padding for example)
    Padding_Value = 255 # (padding colour - 255 = white, 0 = black)

    # threshold - over 0.5 - colour white (1), less than 0.5 - black (0)
    Bin_Threshold = 0.5

    #=============================================================================
    # Post-Processing options
    #-----------------------------------------------------------------------------

    # Post-Processing on resized image
    # if true, applies postprocessing on the resized image (not original size)

    # if True,  it will apply post-processing on resized image that is used to ...
    # ... go through the network
    Resized_Adjustments = True

    #-----------------------------------------------------------------------------

    # Remove small elements by the application of watershed segmentation
    # and removal of small segmentations.

    # if false, does not apply the adjustments:
    #  dilation (remove small artifacts), erosion (if using groups, remove small artifacts)
    Mask_Adjustments1 = False

    # Padding before application of erosion/dilation
    A1_Padding = True
    # Padding size
    A1_Pad_Size = 1
    # Padding value
    A1_Pad_Value = 0

    # Mask dilation/erosion/dilation (in specified order) - to clean noise from images
    # NOTE: can change order!
    A1_Dilation1 = 5 # it increases white, reduces either black or white pixels the are less than 5 pixels
    A1_Erosion2 = 5 # it reduces white increases the size (bring back to original)
    A1_Dilation3 = 1


    #-----------------------------------------------------------------------------

    # Remove small elements by the application of watershed segmentation
    # and removal of small segmentations.
    Mask_Adjustments2 = False

    # Threshold limit of segmentation in pixels
    A2_Clean_Thld = 5 # if too big - might remove a crack!

    # Remove small-areas of zero values
    # if false, does not do segmentation on the black pixels (not count pixels)
    A2_Clean_Background = True

    # Remove small-areas of positive values (same as before, but on white pixels)
    A2_Clean_Foreground = True

    #=============================================================================
    # Overlay options (Dimension calculation)
    #-----------------------------------------------------------------------------

    # Create overlay of the image source and CNN output
    Create_Overlay = True

    # Overlay colour
    Overlay_Colour = [255,55,150]

    # Alpha: Transparency of image source
    alpha = 1.0
    # Beta: Transparency of binarised output
    beta = 0.5
    # Gamma: General transparency (does nothing as of now, but don't remove)
    gamma = 1.0

    #=============================================================================
    # PLT Plotting options (DimensionS)
    #-----------------------------------------------------------------------------

    # Horizontal resolution
    X_Res = 1920;

    # Vertical resolution
    Y_Res = 1080;

    # Monitor DPI
    my_dpi = 96;

    # Default font size of figure
    Font_Size = 10

    #=============================================================================
    # Metric options (Dimension calculation)
    #-----------------------------------------------------------------------------

    # Thinning/Skeletonise mask/segmentation to measure length of elements
    # if false - does not do the skeleton
    Evaluate_Metrics = True

    #=============================================================================
    #                          IMPORTING PACKAGES
    #=============================================================================

    print("")
    print("[INFO] Package Importing Started")

    #-----------------------------------------------------------------------------

    # import the necessary packages
    import time
    # Start timer
    start0 = time.time()
    start = start0


    # import the necessary packages
    import os
    # (Optional) Enable CUDA growth if it gives issues.
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    # (Optional) Disable tensorflow warnings.
    os.environ['AUTOGRAPH_VERBOSITY'] = '0'
    # SM Compatibility option with TF 2.0+
    os.environ["SM_FRAMEWORK"] = "tf.keras"


    import tensorflow as tf
    cfg = tf.compat.v1.ConfigProto()
    cfg.gpu_options.per_process_gpu_memory_fraction = 0.9


    # import the necessary packages
    #import os
    import sys
    import cv2
    import math
    import shutil
    import pathlib
    import numpy as np
    import pandas as pd
    import tkinter as tk
    import matplotlib.pyplot as plt
    from tkinter import filedialog

    # import the necessary packages
    import skimage.morphology
    import skimage.feature
    import skimage.segmentation
    import scipy.ndimage


    # import the necessary packages
    #import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.models import model_from_json

    #-----------------------------------------------------------------------------

    # (Optional) Enable growth CUDA if it gives issues.
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    # (Optional) Disable tensorflow warnings.
    os.environ['AUTOGRAPH_VERBOSITY'] = '0'
    # Destroy plotted windows
    cv2.destroyAllWindows()
    plt.close('all')

    #-----------------------------------------------------------------------------

    end = time.time()
    print("[INFO] Package Importing Finished - T: " \
          + str(round(end-start,3)) + "s -> " + str(round(end-start0,3)) + "s")

    #=============================================================================
    #                            MINI FUNCTIONS
    #=============================================================================

    print("")
    print("[INFO] Mini Functions Started")
    start = time.time()

    #-----------------------------------------------------------------------------

    #import os
    #import pathlib
    #import shutil

    # Create complete path of folders
    def CreatePath(FullPath,File=False):
        Parts = pathlib.Path(FullPath).parts
        for [n1,Folder] in enumerate(Parts):
            if File==True and n1==len(Parts)-1 and "." in Parts[n1]:
                continue
            elif n1==0:
                FolderPath = Parts[n1]
            else:
                FolderPath = os.path.join(FolderPath,Folder)
            if os.path.exists(FolderPath)==False:
                os.mkdir(FolderPath)


    # Delete folder
    def DeleteFolder(FullPath):
        FullPath = pathlib.Path(FullPath)
        try:
            shutil.rmtree(FullPath)
        except:
            pass

    #-----------------------------------------------------------------------------

    #import numpy as np

    # Create circular kernel with custom values and modes
    def CircularKernel(Ksize,Mode=1,Value1=0,Value2=1):
        kernel = np.zeros([Ksize,Ksize])
        kernel[:,:] = Value1
        center = (len(kernel)-1)/2;
        if Mode==1:
            radius = (len(kernel)-0.5)/2
        elif Mode==2:
            radius = (len(kernel)-1)/2
        elif Ksize/2==round(Ksize/2):
            radius = (len(kernel)-0.5)/2
        else:
            radius = (len(kernel)-1)/2
        #radius = (len(kernel)-1)/2
        for i in range(len(kernel)):
            for j in range(len(kernel)):
                x1 = center; y1 = center;
                x2 = j; y2 = i;
                distance = ((x2-x1)**2+(y2-y1)**2)**(1/2)
                if distance<=radius:
                    kernel[i,j] = Value2
        return kernel

    #-----------------------------------------------------------------------------

    #import cv2
    #import numpy as np

    # Pad image by specified values
    def EqualPadding(Image,Pad=1,PadValue=255):
        Padded = np.copy(Image);
        top = Pad; bottom = Pad; left = Pad; right = Pad;
        borderType = cv2.BORDER_REPLICATE
        Padded = cv2.copyMakeBorder(Padded, top, bottom, left, right, borderType)
        Padded[0:Pad, 0:Padded.shape[1]] = PadValue
        Padded[Padded.shape[0]-Pad:Padded.shape[0], 0:Padded.shape[1]] = PadValue
        Padded[0 : Padded.shape[0] , 0 : Pad] = PadValue
        Padded[0 : Padded.shape[0]  , Padded.shape[1]-Pad : Padded.shape[1]] = PadValue
        return Padded

    #-----------------------------------------------------------------------------

    end = time.time()
    print("[INFO] Mini Functions Finished - T: " \
          + str(round(end-start,3)) + "s -> " + str(round(end-start0,3)) + "s")

    #=============================================================================
    #                           DETECTING PATHS
    #=============================================================================

    print("")
    print("[INFO] Reading Paths Started")
    start = time.time()

    #-----------------------------------------------------------------------------

    # Do not change this section
    if Image_Browser==True:
        print("[INFO] Select Image File")
        FDtitle = "Select Image File"
        FDtypes = [("All Files", "*.*")]
        # Launch selection window
        root = tk.Tk(); root.withdraw(); root.update();
        root.call('wm', 'attributes', '.', '-topmost', True)
        FullPath = filedialog.askopenfilename(title=FDtitle, filetypes=FDtypes)
        # Check if the input is appropriate
        if len(FullPath)>=1:
            Image_Path = FullPath
            print("[INFO] Selected File:",FullPath)
        else:
            print("[INFO] No File Selected")
            print("[INFO] Program Finished")
            sys.exit()

    #-----------------------------------------------------------------------------

    Image_Path = "http://127.0.0.1:8000" + Image_Path
    print("Image Path: ", Image_Path)
    # Image_Path = str(pathlib.Path(Image_Path))
    Model_Path = str(pathlib.Path(Model_Path))
    JModel_Path = str(pathlib.Path(JModel_Path))
    Output_Path = str(pathlib.Path(Output_Path))

    Image_File = os.path.basename(Image_Path)
    # The Image_Name is now given as a function argument.
    # Image_Name = os.path.splitext(Image_File)[0]
    Model_Name = os.path.splitext(os.path.basename(Model_Path))[0]

    Predictions_Path = os.path.join(Output_Path, Model_Name)
    AllPredictions_Path = os.path.join(Output_Path, Model_Name, Image_Name)
    AllPredictions_Path1 = os.path.join(AllPredictions_Path, "CV2")
    AllPredictions_Path2 = os.path.join(AllPredictions_Path, "PLT")

    DeleteFolder(AllPredictions_Path)
    CreatePath(AllPredictions_Path1)
    CreatePath(AllPredictions_Path2)

    #-----------------------------------------------------------------------------

    end = time.time()
    print("[INFO] Reading Paths Finished - T: " \
          + str(round(end-start,3)) + "s -> " + str(round(end-start0,3)) + "s")

    #=============================================================================
    #                    MODEL DEFINITION/PREDICTIONS
    #=============================================================================

    print("")
    print("[INFO] Model Definition/Predictions Started")
    start = time.time()

    #-----------------------------------------------------------------------------

    # Load CNN model
    try:
        model = load_model(Model_Path, compile=False, custom_objects={'tf': tf})
        print("[INFO] Loaded Full Model")
    except:
        json_file = open(JModel_Path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json, custom_objects={'tf': tf})
        model.load_weights(Model_Path)
        print("[INFO] Loaded Json Model")


    # Load image
    # Source = cv2.imread(Image_Path)
    print(">>> Image_Path:", Image_Path)
    Source = url_to_image(Image_Path)
    List_Images = [["Original Image",np.copy(Source)]]


    Image = np.copy(Source)
    # Adjust size of images with very small/high resolutions
    [x0,y0] = [Image.shape[1], Image.shape[0]];
    [xi,yi] = [x0,y0]; [xr,yr] = [xi,yi];
    [xo,yo] = [Image_Dims[0],Image_Dims[1]];
    if Adjust_Size==True:
        ImDim = np.average([xi,yi])
        OutDim = np.average([xo,yo])
        MaxLimit = MaxResRatio*OutDim
        MinLimit = MinResRatio*OutDim
        if ImDim>MaxLimit or ImDim<MinLimit:
            if ImDim>MaxLimit:
                Limit = MaxLimit
                Eff = MaxScaleEff
            elif ImDim<MaxLimit:
                Limit = MinLimit
                Eff = MinScaleEff
            Scale = Limit/ImDim
            AdjScale = 1-(1-Scale)*Eff
            # Resize values
            xr = int(np.round(xi*AdjScale));
            yr = int(np.round(yi*AdjScale));
            Image = cv2.resize(Image, (xr,yr))
            # Save image to list
            Saved_Title = "Resized Image"
            Saved_Image = Image
            List_Images.append([Saved_Title, np.copy(Saved_Image)])
        else:
            Adjust_Size = False
    print("[INFO] Adjust Size:", Adjust_Size)


    # Padding image
    PadValue = Padding_Value
    [xi,yi] = [Image.shape[1], Image.shape[0]];
    if len(Image.shape)==3:
        PadValue = [PadValue]*Image.shape[2]
    # Inner window size
    [xw,yw] = [xo-Overlap*2, yo-Overlap*2]
    # Padding size
    xp = int(np.ceil(xi/xw) * int(xw)) + Overlap*2
    yp = int(np.ceil(yi/yw) * int(yw)) + Overlap*2
    # Calculate padding per dimension
    xp1 = int(np.round((xp - xi) / 2)); xp2 = int((xp - xi) - xp1)
    yp1 = int(np.round((yp - yi) / 2)); yp2 = int((yp - yi) - yp1)
    [top,bottom,left,right] = [yp1,yp2,xp1,xp2]
    # Pad image to the prefered size
    Image = cv2.copyMakeBorder(Image, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=PadValue)
    # Create empty mask
    ImgOutput = np.zeros((yp,xp))*0


    # Image sections with overlap
    [xi,yi] = [Image.shape[1], Image.shape[0]];
    ny = int((yi-Overlap*2)/yw); nx = int((xi-Overlap*2)/xw);
    for n1 in range(ny):
        y1 = n1 * yw; y1a = y1 + Overlap
        y2 = y1 + yo; y2a = y2 - Overlap
        for n2 in range(nx):
            x1 = n2 * xw; x1a = x1 + Overlap
            x2 = x1 + xo; x2a = x2 - Overlap
            # Extract part from image
            Part = Image[y1:y2,x1:x2]
            AdjPart = np.array([Part/255])
            PartPred = model.predict(AdjPart)
            PartPred =  PartPred[0,:,:,0]
            PartPred = PartPred[Overlap:yo-Overlap,Overlap:xo-Overlap]
            ImgOutput[y1a:y2a,x1a:x2a] = PartPred


    # Adjust image size
    [xi,yi] = [ImgOutput.shape[1], ImgOutput.shape[0]];
    ImgOutput = ImgOutput[yp1:yp-yp2,xp1:xp-xp2]*255
    if Adjust_Size==True and Resized_Adjustments==False:
        # Save image to list
        Saved_Image = ImgOutput
        Saved_Title = "CNN Output (R)"
        List_Images.append([Saved_Title, np.copy(Saved_Image)])
        # Adjusting mask size
        ImgOutput = cv2.resize(ImgOutput, (x0,y0))
        # Save image to list
        Saved_Image = ImgOutput
        Saved_Title = "CNN Output (O)"
        List_Images.append([Saved_Title, np.copy(Saved_Image)])
    elif Adjust_Size==True and Resized_Adjustments==True:
        # Save image to list
        Saved_Image = cv2.resize(ImgOutput, (x0,y0))
        Saved_Title = "CNN Output (O)"
        List_Images.append([Saved_Title, np.copy(Saved_Image)])
        # Save image to list
        Saved_Image = ImgOutput
        Saved_Title = "CNN Output (R)"
        List_Images.append([Saved_Title, np.copy(Saved_Image)])
    else:
        # Save image to list
        Saved_Image = ImgOutput
        Saved_Title = "CNN Output"
        List_Images.append([Saved_Title, np.copy(Saved_Image)])


    # Adjust image
    BinMask = np.copy(ImgOutput)
    BinMask[BinMask>=Bin_Threshold*255] = 255
    BinMask[BinMask<Bin_Threshold*255] = 0
    BinMask = np.uint8(BinMask)
    FinalMask = np.copy(BinMask)
    # Save image to list
    Saved_Image = BinMask
    Saved_Title = "Binarised (t=" + str(Bin_Threshold) + ")"
    List_Images.append([Saved_Title, np.copy(Saved_Image)])

    #-----------------------------------------------------------------------------

    end = time.time()
    print("[INFO] Model Definition/Predictions Finished - T: " \
          + str(round(end-start,3)) + "s -> " + str(round(end-start0,3)) + "s")

    #=============================================================================
    #                             POST-PROCESSING
    #=============================================================================

    print("")
    print("[INFO] Post-Processing Started")
    start = time.time()

    #-----------------------------------------------------------------------------
    # Apply dilation/erotion/dilation to the mask to remove small artifacts

    # Adjust dilation/erotion values for compatibility purposes
    Adj_List = [A1_Dilation1,A1_Erosion2,A1_Dilation3]
    Adj_Array = np.around(np.array(Adj_List))
    Adj_Array = [i if i>=1 else 1 for i in Adj_Array]
    [A1_Dilation1,A1_Erosion2,A1_Dilation3] = Adj_Array


    # Test if adjustments are required
    if Mask_Adjustments1==True and (np.max(Adj_Array)>=2):
        Mask_Adjustments1 = True
    else:
        Mask_Adjustments1 = False


    # Apply dilation/erotion/dilation
    if Mask_Adjustments1==True:
        AdjMask = np.copy(FinalMask)
        # Mask padding (to adjust the edges of the mask)
        if A1_Padding==True:
            Pad = A1_Pad_Size; PadValue = A1_Pad_Value;
            AdjMask = EqualPadding(AdjMask,Pad=Pad,PadValue=PadValue)
        # Convert mask to uint8 type
        AdjMask = np.array(AdjMask,dtype=np.uint8)
        # Mask dilation
        Ksize = A1_Dilation1
        kernel = CircularKernel(Ksize,Mode=1,Value1=0,Value2=1)
        kernel = np.array(kernel,dtype=np.uint8)
        AdjMask = cv2.dilate(AdjMask,kernel,iterations=1)
        # Mask erosion
        Ksize = A1_Erosion2
        kernel = CircularKernel(Ksize,Mode=1,Value1=0,Value2=1)
        kernel = np.array(kernel,dtype=np.uint8)
        AdjMask = cv2.erode(AdjMask,kernel,iterations=1)
        # Mask dilation
        Ksize = A1_Dilation3
        kernel = CircularKernel(Ksize,Mode=1,Value1=0,Value2=1)
        kernel = np.array(kernel,dtype=np.uint8)
        AdjMask = cv2.dilate(AdjMask,kernel,iterations=1)
        # Correcting mask size
        if A1_Padding==True:
            AdjMask = AdjMask[Pad:AdjMask.shape[0]-Pad, Pad:AdjMask.shape[1]-Pad]
        # Assign adjusted mask as the final mask
        FinalMask = np.copy(AdjMask)
        # Save image to list
        Saved_Image = AdjMask
        Saved_Title = "Adjusted #1"
        List_Images.append([Saved_Title, np.copy(Saved_Image)])
        print("[INFO] Adjustments #1 Finished")

    #-----------------------------------------------------------------------------
    # Remove small areas by adjusting the segmentation of the binary image

    if Mask_Adjustments2==True and \
        (A2_Clean_Foreground==True or A2_Clean_Background==True):

        # Removal of small foreground objects
        if A2_Clean_Foreground==True:
            AdjMask = np.copy(FinalMask)
            Markers = scipy.ndimage.label(AdjMask, structure=None, output=None)[0]
            Watershed = skimage.segmentation.watershed(AdjMask, Markers, mask=Markers)
            [unique, counts] = np.unique(Watershed, return_counts=True)
            UElem = sorted(list(zip(unique, counts)))
            for [n1,[TID,NID]] in enumerate(UElem):
                if NID<A2_Clean_Thld:
                    print("[INFO] Removed item1 with ID: " + str(TID) + " (" + str(NID) + ")")
                    AdjMask[Watershed==TID] = 0
            # Save image to list
            Saved_Image = Watershed
            Saved_Title = "Watershed #1"
            List_Images.append([Saved_Title, np.copy(Saved_Image)])
            # Save image to list
            Saved_Image = AdjMask
            Saved_Title = "Adjusted #2-1"
            List_Images.append([Saved_Title, np.copy(Saved_Image)])
            print("[INFO] Adjustments #2-1 Finished")

        # Removal of small background objects
        if A2_Clean_Background==True:
            AdjMask = np.invert(AdjMask)
            Markers = scipy.ndimage.label(AdjMask, structure=None, output=None)[0]
            Watershed = skimage.segmentation.watershed(AdjMask, Markers, mask=Markers)
            [unique, counts] = np.unique(Watershed, return_counts=True)
            UElem = sorted(list(zip(unique, counts)))
            for [n1,[TID,NID]] in enumerate(UElem):
                if NID<A2_Clean_Thld:
                    print("[INFO] Removed item2 with ID: " + str(TID) + " (" + str(NID) + ")")
                    AdjMask[Watershed==TID] = 0
            # Invert mask
            AdjMask = np.invert(AdjMask)
            # Save image to list
            Saved_Image = Watershed
            Saved_Title = "Watershed #2"
            List_Images.append([Saved_Title, np.copy(Saved_Image)])
            # Save image to list
            Saved_Image = AdjMask
            Saved_Title = "Adjusted #2-2"
            List_Images.append([Saved_Title, np.copy(Saved_Image)])
            print("[INFO] Adjustments #2-2 Finished")

        # Save final mask
        FinalMask = np.copy(AdjMask)

    #-----------------------------------------------------------------------------

    # Adjust size of the final mask if needed
    [xi,yi] = [FinalMask.shape[1], FinalMask.shape[0]];
    #if Adjust_Size==True:
    if [xi,yi]!=[x0,y0]:
        FinalMask = cv2.resize(FinalMask, (x0,y0))
        FinalMask[FinalMask>=Bin_Threshold*255] = 255
        FinalMask[FinalMask<Bin_Threshold*255] = 0
        FinalMask = np.uint8(FinalMask)
        # Save image to list
        Saved_Image = FinalMask
        Saved_Title = "Resized Final Mask"
        List_Images.append([Saved_Title, np.copy(Saved_Image)])

    #-----------------------------------------------------------------------------

    end = time.time()
    print("[INFO] Post-Processing Finished - T: " \
          + str(round(end-start,3)) + "s -> " + str(round(end-start0,3)) + "s")

    #=============================================================================
    #                            IMAGE OVERLAY
    #=============================================================================

    print("")
    print("[INFO] Image Overlay Started")
    start = time.time()

    #-----------------------------------------------------------------------------

    # Create overlay image
    if Create_Overlay==True:
        Overlay = np.copy(Source)
        TrMask = np.copy(FinalMask)
        TrMask = np.dstack([TrMask, TrMask, TrMask])
        TrMask[FinalMask!=0] = Overlay_Colour
        Overlay = cv2.addWeighted(Overlay,alpha,TrMask,beta,gamma)
        # Save image to list
        Saved_Image = Overlay
        Saved_Title = "Overlay"
        List_Images.append([Saved_Title, np.copy(Saved_Image)])

    #-----------------------------------------------------------------------------

    end = time.time()
    print("[INFO] Image Overlay Finished - T: " \
          + str(round(end-start,3)) + "s -> " + str(round(end-start0,3)) + "s")

    #=============================================================================
    #                            IMAGE METRICS
    #=============================================================================

    print("")
    print("[INFO] Image Metrics Started")
    start = time.time()

    #-----------------------------------------------------------------------------

    if Evaluate_Metrics==True:
        # Skeleton of all marked locations
        Skeleton = skimage.morphology.skeletonize(FinalMask/255)*255
        # Skeleton per segmentation
        Markers = scipy.ndimage.label(FinalMask, structure=None, output=None)[0]
        Watershed = skimage.segmentation.watershed(FinalMask, Markers, mask=Markers)
        [unique, counts] = np.unique(Watershed, return_counts=True)
        UElem = sorted(list(zip(unique, counts)))
        List_Sizes = [];
        for [n1,[TID,NID]] in enumerate(UElem):
            # Skip background (skip 0 layer ID)
            if TID==0:
                continue
            Single = np.copy(FinalMask)*0
            Single[Watershed==TID] = 255
            SingleSkeleton = skimage.morphology.skeletonize(Single/255)*255
            NID2 = cv2.countNonZero(SingleSkeleton)
            Locs = np.column_stack(np.where(Single == 255))
            idx = list(Locs[:,0]).index(np.min(Locs[:,0]))
            Loc = list(Locs[idx,:]); Loc.reverse();
            List_Sizes.append([TID,Loc,NID,NID2])
        Df_Sizes = pd.DataFrame(List_Sizes,columns=[['Label','Loc (x,y)','Count (pxls)','Length (pxls)']])
        # Save image to list
        Saved_Image = Skeleton
        Saved_Title = "Skeleton"
        List_Images.append([Saved_Title, np.copy(Saved_Image)])
        # Save image to list
        Saved_Image = Watershed
        Saved_Title = "Final Watershed"
        List_Images.append([Saved_Title, np.copy(Saved_Image)])

    #-----------------------------------------------------------------------------

    end = time.time()
    print("[INFO] Image Metrics Finished - T: " \
          + str(round(end-start,3)) + "s -> " + str(round(end-start0,3)) + "s")

    #=============================================================================
    #                           IMAGE PLOTTING
    #=============================================================================

    print("")
    print("[INFO] Image Plotting Started")
    start = time.time()

    #-----------------------------------------------------------------------------

    # Evaluate number of subplots based on options
    SubPlots = len(List_Images)
    # Evaluate number of rows and collumns of plot
    pltr = math.floor(math.sqrt(SubPlots))
    pltc = math.ceil(SubPlots/pltr)


    # Plot image using plt
    #plt.rcParams.update({'font.size': 10})
    plt.rc('font', size=Font_Size)
    title1 = Image_File + " (" + Model_Name + ")"
    # divide by dpi - because resolution not in pixels, but in inches!
    fig = plt.figure(title1,figsize=(X_Res/my_dpi, Y_Res/my_dpi), dpi=my_dpi)
    fig.tight_layout()
    SubPlot = 0
    for [n1,[Saved_Title,Saved_Image]] in enumerate(List_Images):
        print("[INFO] Image Title:",Saved_Title,Saved_Image.shape)
        if len(Saved_Image.shape)==3 and Saved_Image.shape[2]==3:
            Saved_Image = cv2.cvtColor(Saved_Image, cv2.COLOR_BGR2RGB)
        SubPlot += 1
        ax1 = fig.add_subplot(pltr,pltc,SubPlot)
        ax1.imshow(Saved_Image)
        ax1.title.set_text(Saved_Title)
        ax1.axes.xaxis.set_visible(False)
        ax1.axes.yaxis.set_visible(False)
    # Show plot
    #plt.show()

    #-----------------------------------------------------------------------------

    end = time.time()
    print("[INFO] Model Image Plotting Finished - T: " \
          + str(round(end-start,3)) + "s -> " + str(round(end-start0,3)) + "s")

    #=============================================================================
    #                             DATA SAVING
    #=============================================================================

    print("")
    print("[INFO] Data Saving Started")
    start = time.time()

    #-----------------------------------------------------------------------------

    # make a dictionary to refer to image paths later:
    url_dict = {}

    # Store images
    if Save_Output==True:
        # Store final mask
        BinMask_Path = os.path.join(Predictions_Path, Image_Name + ".png")
        cv2.imwrite(BinMask_Path, FinalMask)
        print("[INFO] Saved image to path: " + BinMask_Path)
        # Store plt figure
        Figure_Path = os.path.join(AllPredictions_Path, Image_Name + " (Fig).png")
        plt.savefig(Figure_Path, dpi=my_dpi, bbox_inches = "tight")
        
        # Save all images from the list
        for [n1,[Saved_Title,Saved_Image]] in enumerate(List_Images):
            # Save using CV2
            Temp_Path = os.path.join(AllPredictions_Path1, Image_Name + \
                        " (#" + str(n1) + " - " + Saved_Title + ").png")
            cv2.imwrite(Temp_Path, Saved_Image)
            # Save using PLT
            if len(Saved_Image.shape)==3 and Saved_Image.shape[2]==3:
                Saved_Image = cv2.cvtColor(Saved_Image, cv2.COLOR_BGR2RGB)
            Temp_Path = os.path.join(AllPredictions_Path2, Image_Name + \
                        " (#" + str(n1) + " - " + Saved_Title + ").png")
            plt.imsave(Temp_Path,Saved_Image)

            url_dict[Saved_Title] = Temp_Path.split('\\', 1)[1]

        # Save segmentation properties
        if Evaluate_Metrics==True:
            crack_length_path = os.path.join(AllPredictions_Path,'Sizes.csv')
            url_dict["crack_len_csv"] = crack_length_path #.split('\\', 1)[1]
            Df_Sizes.to_csv(crack_length_path, index=False)
    else:
        print("[INFO] Output not saved")

    #-----------------------------------------------------------------------------

    end = time.time()
    print("[INFO] Data Saving Finished - T: " \
          + str(round(end-start,3)) + "s -> " + str(round(end-start0,3)) + "s")

    #=============================================================================

    # Show plot (do not show plot on the web application)
    # plt.show()

    print("overlay: ", url_dict["Overlay"])
    print("binaris: ", url_dict["Binarised (t=" + str(Bin_Threshold) + ")"])

    return url_dict["Overlay"], \
           url_dict["Binarised (t=" + str(Bin_Threshold) + ")"], \
           url_dict["crack_len_csv"]

    #=============================================================================
    #                             END PROGRAM
    #=============================================================================
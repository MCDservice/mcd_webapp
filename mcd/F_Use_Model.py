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

def analyse_photo(Image_Path5, Image_Name):
    #=============================================================================
    #                PROGRAM: USE CNN MODEL (SEMANTIC SEGMENTATION)
    #=============================================================================

    # Description:

    #=============================================================================
    #                               OPTIONS
    #=============================================================================

    Image_Browser = False
    import os

    #-----------------------------------------------------------------------------

    # Image_Path5 = "Input//Database//Images//M1-019.jpg"
    # Image_Path5 = "C://Users//Dom//Documents//3-Year//2021-Internship//crack_detection_CNN_masonry//dataset//crack_detection_224_images//guide-to-cracks-in-brick-walls.jpg"
    # Image_Path5 = "C://Users//Dom//Documents//3-Year//2021-Internship//crack_detection_CNN_masonry//dataset//crack_detection_224_images_old//a_4_1.png"
    print(">>> Received Image_Path5: ", "http://127.0.0.1:8000"+Image_Path5)
    Image_Path5 = "http://127.0.0.1:8000" + Image_Path5


    # Model_Path5 = "C://+ BACKUP//PROGRAM DATA//PYTHON//CNN - Semantic Segmentation\Output//+ Tests//Test2//DLV3PLUS//Loss5 (BCE)//Blocks2_198_epoch_97_val_F1_score_0.960.h5"
    Model_Path5 = "C://Users//Dom//Documents//3-Year//2021-Internship//crack_detection_CNN_masonry//output//weights//crack_detection_44072_epoch_10_F1_score_dil_0.747.h5"

    # JModel_Path5 = ""
    JModel_Path5 = "C://Users//Dom//Documents//3-Year//2021-Internship//crack_detection_CNN_masonry//output//model_json//crack_detection_44072.json"

    # Output_Path5 = "Output//Images"
    # Output_Path5 = "C://Users//Dom//Documents//3-Year//2021-Internship//crack_detection_CNN_masonry//output//predictions//"

    dir = os.path.dirname(__file__)
    Output_Path5 = os.path.join(dir, '..', 'media')

    #-----------------------------------------------------------------------------

    # Input image-resolution,
    # Note1: Input Format: (224,224,3), (256,256,3), (512,512,3), etc.
    Image_Dims = (224,224,3)

    #-----------------------------------------------------------------------------

    # Adjust resolution for very large/small image sizes
    Adjust_Size = True

    # The maximum resolution-ratio, based on the output-resolution.
    # ImDim = average(xIm,yIm); OutDim = average(xOut,yOut);
    # Limit = OutDim*ResRatio; Scale = Limit/ImDim
    MaxResRatio = 4

    # The efficiency-ratio of the adjusted-scale when downscaling.
    # Note1: A value of 0 retains the original-size
    # Note2: A value of 1 resizes the image to the limit-size.
    # AdjScale = 1-(1-Scale)*ScaleEff
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

    Overlap = 50

    Padding_Value = 255

    Bin_Threshold = 0.5

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
    # moved the following two imports to top of file:
    # import cv2
    # import numpy as np
    import tkinter as tk
    from tkinter import filedialog
    import matplotlib.pyplot as plt
    from pathlib import Path


    # import the necessary packages
    # import tensorflow as tf
    # from tensorflow.keras.models import load_model
    from tensorflow.keras.models import model_from_json
    # import the necessary packages
    # from Packages.misc.PathTools import CreatePath

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
            Image_Path5 = FullPath
            print("[INFO] Selected File:",FullPath)
        else:
            print("[INFO] No File Selected")
            print("[INFO] Program Finished")
            sys.exit()

    #-----------------------------------------------------------------------------

    Image_Path = str(Path(Image_Path5))
    Model_Path = str(Path(Model_Path5))
    JModel_Path = str(Path(JModel_Path5))
    Output_Path = str(Path(Output_Path5))

    Image_File = os.path.basename(Image_Path)
    # Image_Name = os.path.splitext(Image_File)[0]
    # Image_Name = "test"
    Model_Name = os.path.splitext(os.path.basename(Model_Path))[0]

    Predictions_Path = os.path.join(Output_Path, Model_Name)
    print("output predictions path: ", Predictions_Path)
    # CreatePath(Predictions_Path)

    Mask_Path = os.path.join(Predictions_Path, Image_Name + " (CNN)" + ".png")
    BinMask_Path = os.path.join(Predictions_Path, Image_Name + ".png")

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
    # try:
    #     model = load_model(Model_Path, compile=False, custom_objects={'tf': tf})
    #     print("[INFO] Loaded Full Model")
    # except:
    #     model = model_from_json(JModel_Path, custom_objects={'tf': tf})
    #     model.load_weights(Model_Path)
    #     print("[INFO] Loaded Json Model")

    #################################################################################
    # load json and create model
    json_file = open(JModel_Path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    try:
        model = model_from_json(loaded_model_json)
    except:
        from tensorflow.keras.models import model_from_json

        model = model_from_json(loaded_model_json)

    # load weights into new model
    model.load_weights(Model_Path)
    #################################################################################

    # Load image
    # Source = cv2.imread(Image_Path)
    # Image = np.copy(Source)

    Source = url_to_image(Image_Path5)
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
    Mask = np.zeros((yp,xp))*0


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
            Mask[y1a:y2a,x1a:x2a] = PartPred


    # Adjust image size
    [xi,yi] = [Mask.shape[1], Mask.shape[0]];
    Mask = Mask[yp1:yp-yp2,xp1:xp-xp2]
    Mask = cv2.resize(Mask, (x0,y0))
    # Adjust image
    BinMask = np.copy(Mask)
    BinMask[BinMask>=Bin_Threshold] = 1
    BinMask[BinMask<Bin_Threshold] = 0
    BinMask = np.uint8(BinMask*255)
    Mask = np.uint8(Mask*255)


    # Plot image using plt
    SourceRGB = cv2.cvtColor(Source, cv2.COLOR_BGR2RGB)
    TrMask = np.float64(np.copy(BinMask))
    TrMask[TrMask==0] = np.NaN;
    title1 = Image_File + " (" + Model_Name + ")"
    fig = plt.figure(title1)
    fig.tight_layout()
    # Subplot #1
    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(SourceRGB)
    ax1.title.set_text("Image")
    ax1.axis('off')
    # Subplot #2
    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(SourceRGB)
    ax2.imshow(TrMask, cmap='brg', alpha=0.35)
    ax2.title.set_text("Overlay")
    ax2.axis('off')
    # Subplot #3
    ax3 = fig.add_subplot(2,2,3)
    ax3.imshow(Mask, cmap='gray')
    ax3.title.set_text("CNN Output")
    ax3.axes.xaxis.set_visible(False)
    ax3.axes.yaxis.set_visible(False)
    # Subplot #4
    ax4 = fig.add_subplot(2,2,4)
    ax4.imshow(BinMask, cmap='gray')
    ax4.title.set_text("Binarised (t= " + str(Bin_Threshold) + ")")
    ax4.axes.xaxis.set_visible(False)
    ax4.axes.yaxis.set_visible(False)
    # Show plot
    plt.show()
    # Store plt image
    Figure_Path = os.path.join(Predictions_Path, Image_Name + "(Fig).png")
    #plt.savefig(Figure_Path, dpi=200, bbox_inches = "tight", pad_inches=0.05, constrained_layout=True)
    plt.savefig(Figure_Path, dpi=200, bbox_inches = "tight")


    # Store cv2 image
    cv2.imwrite(Mask_Path, Mask)
    cv2.imwrite(BinMask_Path, BinMask)

    #-----------------------------------------------------------------------------

    end = time.time()
    print("[INFO] Model Definition/Predictions Finished - T: " \
          + str(round(end-start,3)) + "s -> " + str(round(end-start0,3)) + "s")

    #=============================================================================
    #                             DATA SAVING
    #=============================================================================

    print("")
    print("[INFO] Data Saving Started")
    start = time.time()

    #-----------------------------------------------------------------------------


    #-----------------------------------------------------------------------------

    end = time.time()
    print("[INFO] Data Saving Finished - T: " \
          + str(round(end-start,3)) + "s -> " + str(round(end-start0,3)) + "s")

    return "/crack_detection_44072_epoch_10_F1_score_dil_0.747/"+ \
           Image_Name + ".png"


    #=============================================================================
    #                             END PROGRAM
    #=============================================================================
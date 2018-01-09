import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
import dicom  # for reading dicom files
import SimpleITK as sitk
from glob import glob
import numpy as np
from keras.models import load_model
import pandas as pd  # just to load in the labels data and quickly reference it
import re
from skimage.measure import label, regionprops
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier as RF
import xgboost as xgb

import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Conv2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

def create_lungmask(file_list):
    for img_file in file_list:
        # I ran into an error when using Kmean on np.float16, so I'm using np.float64 here
        imgs_to_process = np.load(img_file).astype(np.float64)
        print ("on image", img_file)
        for i in range(len(imgs_to_process)):
            img = imgs_to_process[i]
            #Standardize the pixel values
            mean = np.mean(img)
            std = np.std(img)
            img = img-mean
            img = img/std
            # Find the average pixel value near the lungs
            # to renormalize washed out images
            middle = img[100:400,100:400]
            mean = np.mean(middle)
            max = np.max(img)
            min = np.min(img)
            # To improve threshold finding, I'm moving the
            # underflow and overflow on the pixel spectrum
            img[img==max]=mean
            img[img==min]=mean
            #
            # Using Kmeans to separate foreground (radio-opaque tissue)
            # and background (radio transparent tissue ie lungs)
            # Doing this only on the center of the image to avoid
            # the non-tissue parts of the image as much as possible
            #
            kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
            centers = sorted(kmeans.cluster_centers_.flatten())
            threshold = np.mean(centers)
            thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
            #
            # I found an initial erosion helful for removing graininess from some of the regions
            # and then large dialation is used to make the lung region
            # engulf the vessels and incursions into the lung cavity by
            # radio opaque tissue
            #
            eroded = morphology.erosion(thresh_img,np.ones([4,4]))
            dilation = morphology.dilation(eroded,np.ones([10,10]))
            #
            #  Label each region and obtain the region properties
            #  The background region is removed by removing regions
            #  with a bbox that is to large in either dimnsion
            #  Also, the lungs are generally far away from the top
            #  and bottom of the image, so any regions that are too
            #  close to the top and bottom are removed
            #  This does not produce a perfect segmentation of the lungs
            #  from the image, but it is surprisingly good considering its
            #  simplicity.
            #
            labels = measure.label(dilation)
            label_vals = np.unique(labels)
            regions = measure.regionprops(labels)
            good_labels = []
            for prop in regions:
                B = prop.bbox
                if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
                    good_labels.append(prop.label)
            mask = np.ndarray([512,512],dtype=np.int8)
            mask[:] = 0
            #
            #  The mask here is the mask for the lungs--not the nodes
            #  After just the lungs are left, we do another large dilation
            #  in order to fill in and out the lung mask
            #
            for N in good_labels:
                mask = mask + np.where(labels==N,1,0)
            mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
            imgs_to_process[i] = mask
        np.save(img_file.replace("images","lungmask"),imgs_to_process)

def mask_the_images(working_path,set_name):
    """
    Here we're applying the masks and cropping and resizing the image


    :param working_path:
    :return:
    """

    file_list=glob('/media/talhassid/My Passport/haimTal/Unet/test_images_335a834f795a4549ab818dd19090f147.npy')
    out_images = []      #final set of images for all patients
    for fname in file_list:
        out_images_per_patient = []
        print ("working on file ", fname)
        imgs_to_process = np.load(fname.replace("lungmask","images")) # images of one patient
        masks = np.load(fname)
        for i in range(len(imgs_to_process)):
            mask = masks[i]
            img = imgs_to_process[i]
            new_size = [512,512]   # we're scaling back up to the original size of the image
            img= mask*img          # apply lung mask
            #
            # renormalizing the masked image (in the mask region)
            #
            new_mean = np.mean(img[mask>0])
            new_std = np.std(img[mask>0])
            #
            #  Pulling the background color up to the lower end
            #  of the pixel range for the lungs
            #
            old_min = np.min(img)       # background color
            img[img==old_min] = new_mean-1.2*new_std   # resetting backgound color
            img = img-new_mean
            img = img/new_std
            #make image bounding box  (min row, min col, max row, max col)
            labels = measure.label(mask)
            regions = measure.regionprops(labels)
            #
            # Finding the global min and max row over all regions
            #
            min_row = 512
            max_row = 0
            min_col = 512
            max_col = 0
            for prop in regions:
                B = prop.bbox
                if min_row > B[0]:
                    min_row = B[0]
                if min_col > B[1]:
                    min_col = B[1]
                if max_row < B[2]:
                    max_row = B[2]
                if max_col < B[3]:
                    max_col = B[3]
            width = max_col-min_col
            height = max_row - min_row
            if width > height:
                max_row=min_row+width
            else:
                max_col = min_col+height
            #
            # cropping the image down to the bounding box for all regions
            # (there's probably an skimage command that can do this in one line)
            #
            img = img[min_row:max_row,min_col:max_col]
            mask =  mask[min_row:max_row,min_col:max_col]
            if max_row-min_row <5 or max_col-min_col<5:  # skipping all images with no god regions
                pass
            else:
                # moving range to -1 to 1 to accomodate the resize function
                mean = np.mean(img)
                img = img - mean
                min = np.min(img)
                max = np.max(img)
                img = img/(max-min)
                new_img = resize(img,[512,512], mode='constant')
                out_images_per_patient.append(new_img)

        id = re.sub(r'.*_images_(.*)\.npy',r'\1',fname)
        patient_images_and_id = (out_images_per_patient,id)
        out_images.append(patient_images_and_id)

     # num_images = len(out_images)
    # final_images = np.ndarray([num_images,1,512,512],dtype=np.float32)
    num_patients = len(out_images)
    # final_images_and_ids = []
    # for i in range(num_images):
    #     final_images[i,0] = out_images[i][0]
    #     final_images_and_ids = (final_images[i,0],out_images[i][1])

    np.save(working_path+"{}Images.npy".format(set_name),out_images)
K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

img_rows = 512
img_cols = 512

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def get_unet():
    inputs = Input((1,img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def get_mask_from_unet(output_path,data,set_name):
    print('-'*30)
    print('Loading data...')
    print('-'*30)
    # imgs_test = np.load(data).astype(np.float32)
    imgs_test_and_ids = np.load(data)

    print('-'*30)
    print('compiling model...')
    print('-'*30)
    model = get_unet()
    # model = load_model('/home/talhassid/PycharmProjects/DSB3Tutorial-master/unet.hdf5')

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('/home/talhassid/PycharmProjects/DSB3Tutorial-master/unet.hdf5')

    print('-'*30)
    print('Predicting masks on data...')
    print('-'*30)

    # num_test = len(imgs_test)
    num_patients = len(imgs_test_and_ids)


    imgs_mask_test_and_ids = []
    for i in range(num_patients):
        num_test = len(imgs_test_and_ids[i][0])
        imgs_mask_test = np.ndarray([num_test,1,512,512],dtype=np.float32)
        for j in range(num_test):
            imgs_mask_test[j] = model.predict([imgs_test_and_ids[i][0][j]], verbose=0)[0]
        imgs_mask_test_and_ids.append((imgs_mask_test,imgs_test_and_ids[i][1]))
    np.save('{}masks{}Predicted.npy'.format(output_path,set_name), imgs_mask_test_and_ids)

def getRegionFromMap(slice_npy):
    thr = np.where(slice_npy > np.mean(slice_npy),0.,1.0)
    label_image = label(thr)
    labels = label_image.astype(int)
    regions = regionprops(labels)
    return regions

def getRegionMetricRow(fname):
    # fname, numpy array of dimension [#slices, 1, 512, 512] containing the images
    seg = np.load(fname)
    nslices = seg.shape[0]

    #metrics
    totalArea = 0.
    avgArea = 0.
    maxArea = 0.
    avgEcc = 0.
    avgEquivlentDiameter = 0.
    stdEquivlentDiameter = 0.
    weightedX = 0.
    weightedY = 0.
    numNodes = 0.
    numNodesperSlice = 0.
    # crude hueristic to filter some bad segmentaitons
    # do not allow any nodes to be larger than 10% of the pixels to eliminate background regions
    maxAllowedArea = 0.10 * 512 * 512

    areas = []
    eqDiameters = []
    for slicen in range(nslices):
        regions = getRegionFromMap(seg[slicen,0,:,:])
        for region in regions:
            if region.area > maxAllowedArea:
                continue
            totalArea += region.area
            areas.append(region.area)
            avgEcc += region.eccentricity
            avgEquivlentDiameter += region.equivalent_diameter
            eqDiameters.append(region.equivalent_diameter)
            weightedX += region.centroid[0]*region.area
            weightedY += region.centroid[1]*region.area
            numNodes += 1

    weightedX = weightedX / totalArea
    weightedY = weightedY / totalArea
    avgArea = totalArea / numNodes
    avgEcc = avgEcc / numNodes
    avgEquivlentDiameter = avgEquivlentDiameter / numNodes
    stdEquivlentDiameter = np.std(eqDiameters)

    maxArea = max(areas)


    numNodesperSlice = numNodes*1. / nslices


    return np.array([avgArea,maxArea,avgEcc,avgEquivlentDiameter,\
                     stdEquivlentDiameter, weightedX, weightedY, numNodes, numNodesperSlice])

def createFeatureDataset(nodfiles_path):
    # dict with mapping between training examples and true labels
    # the training set is the output masks from the unet segmentation
    print('-'*30)
    print('Create features...')
    print('-'*30)
    labels_df = pd.read_csv('/media/talhassid/My Passport/haimTal/stage1_labels.csv', index_col=0)
    numfeatures = 9
    nodfiles = np.load(nodfiles_path) #.astype(np.float32)
    feature_array = np.zeros((len(nodfiles),numfeatures))
    truth_metric = np.zeros((len(nodfiles)))

    for i,nodfile in enumerate(nodfiles):
        patID = nodfile[1]
        truth_metric[i] = labels_df.get_value(patID, 'cancer')
        feature_array[i] = getRegionMetricRow(nodfile[0])

    np.save("/media/talhassid/My Passport/haimTal/Unet/labels.npy", truth_metric)
    np.save("/media/talhassid/My Passport/haimTal/Unet/masks.npy", feature_array)

import scipy as sp
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


def classifyData():
    X = np.load("/media/talhassid/My Passport/haimTal/Unet/masks.npy")
    Y = np.load("/media/talhassid/My Passport/haimTal/Unet/labels.npy")

    kf = KFold(Y, n_folds=3)
    y_pred = Y * 0
    for train, test in kf:
        X_train, X_test, y_train, y_test = X[train,:], X[test,:], Y[train], Y[test]
        clf = RF(n_estimators=100, n_jobs=3)
        clf.fit(X_train, y_train)
        y_pred[test] = clf.predict(X_test)
    print (classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"]))
    print("logloss",logloss(Y, y_pred))

    # All Cancer
    print ("Predicting all positive")
    y_pred = np.ones(Y.shape)
    print (classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"]))
    print("logloss",logloss(Y, y_pred))

    # No Cancer
    print ("Predicting all negative")
    y_pred = Y*0
    print (classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"]))
    print("logloss",logloss(Y, y_pred))

    # try XGBoost
    print ("XGBoost")
    kf = KFold(Y, n_folds=3)
    y_pred = Y * 0
    for train, test in kf:
        X_train, X_test, y_train, y_test = X[train,:], X[test,:], Y[train], Y[test]
        clf = xgb.XGBClassifier(objective="binary:logistic")
        clf.fit(X_train, y_train)
        y_pred[test] = clf.predict(X_test)
    print (classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"]))
    print("logloss",logloss(Y, y_pred))


"""
preprocessing the data
"""
file_list_test=glob('/media/talhassid/My Passport/haimTal/Unet/test_images_335a834f795a4549ab818dd19090f147.npy')
create_lungmask(file_list_test)
mask_the_images('/media/talhassid/My Passport/haimTal/',"test")

"""
getting mask from unet
"""
get_mask_from_unet(output_path='/media/talhassid/My Passport/haimTal/'
                    ,data='/media/talhassid/My Passport/haimTal/testImages.npy',set_name="Test")

nodfiles = '/media/talhassid/My Passport/haimTal/masksTestPredicted.npy'
createFeatureDataset(nodfiles)
classifyData()

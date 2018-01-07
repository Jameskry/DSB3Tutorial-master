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


def save_data(working_path: str, set_name: str,output_path):
    patients = os.listdir(working_path)
    for num, patient in enumerate(patients):
        path = working_path + patient
        images = [dicom.read_file(path + '/' + s).pixel_array for s in os.listdir(path)]
        np.save("{}{}_images_{}.npy".format(output_path,set_name,patient),images)

def load_data():
    output_path = '/media/talhassid/My Passport/haimTal/Unet/'
    working_path_test = '/media/talhassid/My Passport/haimTal/stage2/stage2/'
    save_data(working_path_test,"test",output_path)
    working_path_train = '/media/talhassid/My Passport/haimTal/train/'
    save_data(working_path_train,"train",output_path)

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

    file_list=glob(working_path+"lungmask_*.npy")
    out_images = []      #final set of images
    for fname in file_list:
        print ("working on file ", fname)
        imgs_to_process = np.load(fname.replace("lungmask","images"))
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
                new_img = resize(img,[512,512])

                id = re.sub(r'.*_lungmask_(.*)\.py',r'\1',fname)
                patient_images_and_id = [new_img,id]

                out_images.append(patient_images_and_id)

    num_images = len(out_images)
    # final_images = np.ndarray([num_images,1,512,512],dtype=np.float32)
    final_images_and_ids = []
    for i in range(num_images):
        # final_images[i,0] = out_images[i][0]
        final_images_and_ids = out_images[i]

    np.save(working_path+"{}Images.npy".format(set_name),final_images_and_ids)

def get_mask_from_unet(output_path,data,set_name):
    print('-'*30)
    print('Loading data...')
    print('-'*30)
    imgs_test = np.load(data).astype(np.float32)

    print('-'*30)
    print('compiling model...')
    print('-'*30)
    model = load_model('./unet.hdf5')

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('./unet.hdf5')

    print('-'*30)
    print('Predicting masks on data...')
    print('-'*30)

    num_test = len(imgs_test)
    imgs_mask_test = np.ndarray([num_test,1,512,512],dtype=np.float32)
    imgs_mask_test_and_ids = []
    for i in range(num_test):
        imgs_mask_test[i] = model.predict([imgs_test[i][0]], verbose=0)[0]
        imgs_mask_test_and_ids.append((imgs_mask_test[i],imgs_test[i][1]))
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

def createFeatureDataset(nodfiles):
    # dict with mapping between training examples and true labels
    # the training set is the output masks from the unet segmentation
    labels_df = pd.read_csv('/media/talhassid/My Passport/haimTal/stage1_labels.csv', index_col=0)
    numfeatures = 9
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

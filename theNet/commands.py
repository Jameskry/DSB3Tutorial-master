import os
import scipy
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import matplotlib.pyplot as plt
import juliandewit.helpers as helpers


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import cv2
import dicom  # for reading dicom files
import math
import numpy as np
import pandas as pd  # for some simple data analysis (right now, just to load in the labels data and quickly reference it)
import tensorflow as tf
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from skimage import data
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

###############################################process data#########################################################
IMG_PX_SIZE = 100 #to make the slices in same size.
SLICE_COUNT = 20  #numbers of slices in each chunk.



def chunks(l, n):
#creates l sized chunks from list n. seperating list to lists.
    n=int(n)
    for i in range(0, len(l), n):
        yield l[i:i + n]

def mean(l):
#mean of a list
    return sum(l) / len(l)

def load_and_reshape(patient,labels_df,data_dir,img_px_size=50,hm_slices=20):
    label = labels_df.get_value(patient, 'cancer') #the value for the cancer column
    path = data_dir + patient
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2])) # sorting the dicom by x image position

    new_slices = []

    slices = [cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE)) for each_slice in slices]

    chunk_number = math.ceil(len(slices) / SLICE_COUNT) #number of chunks

    for slice_chunk in chunks(slices, chunk_number):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    if len(new_slices) == SLICE_COUNT-1:
        new_slices.append(new_slices[-1])

    if len(new_slices) == SLICE_COUNT-2:
        new_slices.append(new_slices[-1])
        new_slices.append(new_slices[-1])

    if len(new_slices) == SLICE_COUNT+2:
        new_val = list(map(mean, zip(*[new_slices[SLICE_COUNT-1],new_slices[SLICE_COUNT],])))
        del new_slices[SLICE_COUNT]
        new_slices[SLICE_COUNT-1] = new_val

    if len(new_slices) == SLICE_COUNT+1:
        new_val = list(map(mean, zip(*[new_slices[SLICE_COUNT-1],new_slices[SLICE_COUNT],])))
        del new_slices[SLICE_COUNT]
        new_slices[SLICE_COUNT-1] = new_val

    #left column nocancer,right column cancer
    if label == 1: label=np.array([0,1])
    elif label == 0: label=np.array([1,0])

    return np.array(new_slices), label

def load_scan(patient,data_dir,img_px_size=50,hm_slices=20):

    path = data_dir + patient
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2])) # sorting the dicom by x image position
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

#the next functions take care of segmentation

def remove_noise_from_segmented_lungs(segmented_ct_scan):
    selem = ball(2)
    binary = binary_closing(segmented_ct_scan, selem)

    label_scan = label(binary)

    areas = [r.area for r in regionprops(label_scan)]
    areas.sort()

    for r in regionprops(label_scan):
        max_x, max_y, max_z = 0, 0, 0
        min_x, min_y, min_z = 1000, 1000, 1000

        for c in r.coords:
            max_z = max(c[0], max_z)
            max_y = max(c[1], max_y)
            max_x = max(c[2], max_x)

            min_z = min(c[0], min_z)
            min_y = min(c[1], min_y)
            min_x = min(c[2], min_x)
        if (min_z == max_z or min_y == max_y or min_x == max_x or r.area > areas[-3]):
            for c in r.coords:
                segmented_ct_scan[c[0], c[1], c[2]] = 0
        else:
            index = (max((max_x - min_x), (max_y - min_y), (max_z - min_z))) / (min((max_x - min_x), (max_y - min_y) , (max_z - min_z)))

#segment a 2D slice
def get_segmented_lungs(im, plot=False):

    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < 604
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone)
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone)
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.bone)

    im[im < 604] = 0

    #im = remove_noise_from_segmented_lungs(im)

    return im

#takes a whole folder and segment all slices. uses the above function.
def segment_lung_from_ct_scan(ct_scan):
    return np.asarray([get_segmented_lungs(slice) for slice in ct_scan])

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def resample(image, scan,img_px_size=50,hm_slices=20):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)
    new_shape = np.array([hm_slices,img_px_size,img_px_size])
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing

def plot_3d(image, threshold=-300):

    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)

    verts, faces = measure.marching_cubes_classic(p, threshold)
    #verts, faces, _, _ = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        biggest = vals[np.argmax(counts)]
    else:
        biggest = None
    return biggest

def segment_lung_mask(image, fill_lung_structures=True):

    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0,0,0]

    #Fill the air around the person
    binary_image[background_label == labels] = 2


    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1


    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image


MIN_BOUND = -1000.0
MAX_BOUND = 400.0

def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

PIXEL_MEAN = 0.25

def zero_center(image):
    image = image - PIXEL_MEAN
    return image


def process_data(patient,data_dir,img_px_size,hm_slices):
    img_data = load_scan(patient,data_dir,img_px_size=img_px_size, hm_slices=hm_slices)
    img_pixels = get_pixels_hu(img_data)
    img_pix_resampled, spacing = resample(img_pixels, img_data, img_px_size=img_px_size, hm_slices=hm_slices)
    #print("Shape before resampling\t", img_pixels.shape)
    #print("Shape after resampling\t", img_pix_resampled.shape)
    #plot_3d(img_pix_resampled, 400)
    #img_pix_segmented = helpers.get_segmented_lungs(img_pix_resampled, plot= True)

    img_pix_resampled =  segment_lung_from_ct_scan(img_pix_resampled)

    # plot_3d(segmented_lungs, 0)
    # plot_3d(segmented_lungs_fill - segmented_lungs, 0)

    img_pix_resampled = normalize(img_pix_resampled)
    img_pix_resampled = zero_center(img_pix_resampled)
    return  img_pix_resampled

def load_and_process_data(patients,labels_df,data_dir,file_name,train_flag):
    print ("starting process data...\n")
    much_data = []
    #just to know where we are, each 100 patient we will print out
    for num, patient in enumerate(patients):
        if num%10==0:
            print(num)
        try:
            img_data = process_data(patient,data_dir,img_px_size=IMG_PX_SIZE, hm_slices=SLICE_COUNT)
            if (train_flag):
                label = labels_df.get_value(patient, 'cancer') #the value for the cancer column
                #left column nocancer,right column cancer
                if label == 1: label=np.array([0,1])
                elif label == 0: label=np.array([1,0])
                much_data.append([img_data,label,patient])
            else:
               much_data.append([img_data,patient])
        except KeyError as e:
            print('This is unlabeled data!')

    np.save('/media/talhassid/My Passport/haimTal/Proccesed Data/{}-{}-{}-{}.npy'
            .format(file_name,IMG_PX_SIZE,IMG_PX_SIZE,SLICE_COUNT), much_data)

###############################################building the net######################################################
def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def maxpool3d(x):
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

def convolutional_neural_network(x, keep_rate=0.8, n_classes=2):
    #                # 3 x 3 x 3 patches, 1 channel, 32 features to compute.
    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,1,32])),
               #       3 x 3 x 3 patches, 32 channels, 64 features to compute.
               'W_conv2':tf.Variable(tf.random_normal([3,3,3,32,64])),
                'W_conv3':tf.Variable(tf.random_normal([3,3,3,64,128])),
               'W_conv4':tf.Variable(tf.random_normal([3,3,3,128,256])),
               'W_conv5':tf.Variable(tf.random_normal([3,3,3,256,512])),
               'W_conv6':tf.Variable(tf.random_normal([3,3,3,512,1024])),
               #                                  64 features
               'W_fc':tf.Variable(tf.random_normal([4096,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_conv3':tf.Variable(tf.random_normal([128])),
              'b_conv4':tf.Variable(tf.random_normal([256])),
              'b_conv5':tf.Variable(tf.random_normal([512])),
              'b_conv6':tf.Variable(tf.random_normal([1024])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    #                            image X      image Y        image Z
    x = tf.reshape(x, shape=[-1, IMG_PX_SIZE, IMG_PX_SIZE, SLICE_COUNT, 1])

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)

    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    conv3 = tf.nn.relu(conv3d(conv2, weights['W_conv3']) + biases['b_conv3'])
    conv3 = maxpool3d(conv3)

    conv4 = tf.nn.relu(conv3d(conv3, weights['W_conv4']) + biases['b_conv4'])
    conv4 = maxpool3d(conv4)

    conv5 = tf.nn.relu(conv3d(conv4, weights['W_conv5']) + biases['b_conv5'])
    conv5 = maxpool3d(conv5)

    conv6 = tf.nn.relu(conv3d(conv5, weights['W_conv6']) + biases['b_conv6'])
    conv6 = maxpool3d(conv6)

    fc = tf.reshape(conv6,[-1, 4096])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)


    output = tf.matmul(fc, weights['out'],name="op_to_restore")+biases['out'] #add name="op_to_restore"

    return output

###############################################train the net##########################################################


def train_neural_network(epochs_count=30, validation_count=100):
# loading data
    print ("loading process data...\n")
    # much_data = np.load('/media/talhassid/My Passport/haimTal/Proccesed Data/sample-100-100-20.npy')
    much_data = np.load('/media/talhassid/My Passport/haimTal/Proccesed Data/train_processed-100-100-20.npy')
    print ("loading process data finished, starting the training...\n")
    train_data = much_data[:-validation_count] #2 for sampleimages and 100 for stage1
    validation_data = much_data[-validation_count:]
# the network
    x = tf.placeholder('float',name="input_tensor") # will consist a tensor of floating point numbers.
    y = tf.placeholder('float',name="target_tensor") # the target output classes will consist a tensor.
    output_layer = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
# training
    hm_epochs = epochs_count

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    with open('/media/talhassid/My Passport/haimTal/Proccesed Data/accuracy_results.txt', 'w') as a:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(hm_epochs):
                epoch_loss = 0
                for data in train_data:
                    try:
                        X = data[0]
                        Y = data[1]
                        _, c= sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                        epoch_loss += c
                    except Exception as e:
                        pass
                print('Epoch', epoch+1, 'completed , loss:',epoch_loss)
                # find predictions on val set
                correct = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}),file=a)

            print('Done. Finishing accuracy:')
            print('Validation Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}),file=a)
            print ("Training complete!")
            # Save the variables to disk.
            save_path = saver.save(sess, "/media/talhassid/My Passport/haimTal/Proccesed Data/model.ckpt")
            print("Model saved in file: %s" % save_path)

############################################################testing#######################################
        # with open('/home/talhassid/PycharmProjects/lung_cancer/sentex/prediction.txt', 'w') as f:
        #     for index in range(10):
        #         test_data = much_data[index]
        #         X = test_data[0]
        #         Y = test_data[1]
        #         feed_dict = {x:X,y:Y}
        #         prediction=tf.nn.softmax(output_layer)
        #         print ("p_id:",much_data[index][2], "prediction[no_cancer , cancer]:",
        #                sess.run(prediction,feed_dict=feed_dict),file=f)

def test():
    sess=tf.Session()
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('/media/talhassid/My Passport/haimTal/Proccesed Data/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('/media/talhassid/My Passport/haimTal/Proccesed Data/'))

    graph = tf.get_default_graph()

    #Now, access the op that you want to run.
    op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
    x_restore = graph.get_tensor_by_name("input_tensor:0")
    y_restore = graph.get_tensor_by_name("target_tensor:0")
    # Now, let's access and create placeholders variables and
    # create feed-dict to feed new data

    # much_data = np.load('/media/talhassid/My Passport/haimTal/Proccesed Data/sample-100-100-20.npy')
    much_data = np.load('/media/talhassid/My Passport/haimTal/Proccesed Data/test_processed-100-100-20.npy')


    with open('/media/talhassid/My Passport/haimTal/Proccesed Data/test_results.csv', 'w') as f:
        print ("id,cancer",file=f)
        for index,_ in enumerate(much_data):
            test_data = much_data[index]
            X = test_data[0]
            feed_dict = {x_restore:X}
            prediction=tf.nn.softmax(op_to_restore)
            pred = sess.run(prediction,feed_dict=feed_dict)
            print (much_data[index][1],",",pred)
            print (much_data[index][1],",",pred[0][1],file=f)
            #print (much_data[index][1],",",(prediction.eval(session=sess,feed_dict=feed_dict)[0][1]))

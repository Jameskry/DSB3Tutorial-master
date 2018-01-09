from tutorial_code.ours_commands import create_lungmask, mask_the_images, save_data, \
    get_mask_from_unet, createFeatureDataset, classifyData
from glob import glob


"""
Loading the data
"""
output_path = '/media/talhassid/My Passport/haimTal/Unet/'

working_path_test = '/media/talhassid/My Passport/haimTal/stage2/stage2/'
# save_data(working_path_test,"test",output_path)

# working_path_train = '/media/talhassid/My Passport/haimTal/train/'
# save_data(working_path_train,"train",output_path)

"""
preprocessing the data
"""
working_path = '/media/talhassid/My Passport/haimTal/Unet/'

file_list_test=glob(working_path+"test_images_*.npy")
create_lungmask(file_list_test)
mask_the_images(working_path+"test_","test")

# file_list_train=glob(working_path+"train_images_*.npy")
# create_lungmask(file_list_train)
# mask_the_images(working_path+"train_","train")

"""
getting mask from unet
"""
# get_mask_from_unet(working_path,working_path+"trainImages.npy","Train")
get_mask_from_unet(working_path,working_path+"testImages.npy","Test")

nodfiles = [working_path+'masksTestPredicted.npy']#, working_path+'masksTrainPredicted.npy']
createFeatureDataset(nodfiles)
classifyData()

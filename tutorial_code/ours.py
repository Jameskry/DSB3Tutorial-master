from theNet.commands import load_scan
import os
import numpy as np


IMG_PX_SIZE = 512



def save_data(working_path: str, set_name: str):
    patients = os.listdir(working_path)
    much_data = []
    for num, patient in enumerate(patients):
        scans_num = len(os.listdir(working_path + patient))
        img_data = load_scan(patient,working_path,img_px_size=IMG_PX_SIZE, hm_slices=scans_num)
        much_data.append([img_data,patient])
    np.save('/media/talhassid/My Passport/haimTal/Unet/{}.npy'.format(set_name), much_data)

working_path_test = '/media/talhassid/My Passport/haimTal/stage2/stage2/'
save_data(working_path_test,"test_set")
working_path_train = '/media/talhassid/My Passport/haimTal/train/'
save_data(working_path_train,"train_set")

#ROI= region of interest

import dicom
dc = dicom.read_file('/media/talhassid/My Passport/haimTal/stage2/stage2/0b8afe447b5f1a2c405f41cf2fb1198e/0a7b6f19e1d1e9616cd76b63c77b661c.dcm')
img = dc.pixel_array


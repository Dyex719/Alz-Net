import os
import pprint
import nibabel as nib

# This is how we will search for the correct file names in the correct directory

def fileList(source):
    matches = []
    for root, dirnames, filenames in os.walk(source):
        for filename in filenames:
            #print(root)
            #print(filenames)
            if ('SUBJ_111') in root:
                # I don't like this hardcoded string,
                # but I don't know how the file labelling is done by medical practitioners
                if 'MR1' in filename and filename.endswith('.img'):
                    matches.append(os.path.join(root,filename))
    return matches

def id_list(source):
    matches = []
    for root, dirnames, filenames in os.walk(source):
        for dirname in dirnames:
            if ('OAS1') and ('MR1') in dirname:
                matches.append(dirname)
    return matches

#As a best practice, data should be stored in data folder
patients = fileList(os.getcwd() + '\\Data\\')
# pprint.pprint(patients)
ids = id_list(os.getcwd() + '/Data/')
#pprint.pprint(ids)


import nibabel as nib
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
import pandas as pd
import math

example_file = os.getcwd() + '\\Data\\OAS1_0351_MR1\\PROCESSED\\MPRAGE\\SUBJ_111\\OAS1_0351_MR1_mpr_n4_anon_sbj_111.img'

# Still have to add for all the files in matches
# This gives me 20+ slices
# 50,nib.load(example_file).shape[2] - 30,5
def all_slice(example_file):
    loaded_file = nib.load(example_file)
    all_slices = []
    for i in range(80,83):
        #plt.figure()
        #plt.axis('on')
        one_slice = list(np.rot90((loaded_file.get_data()[..., i, 0])))
        #plt.imshow(slices, interpolation = "nearest", cmap=plt.cm.gray)
        #plt.show()
        all_slices.append(one_slice)
    return all_slices

# a = all_slice(example_file) # this works
# print(a[128])

labels_file = os.getcwd() + '/Data/OASIS/oasis_cross-sectional.csv'
read_labels_df = pd.read_csv(labels_file, index_col=0)
cdr_df = read_labels_df['CDR'].fillna(0)
#cdr_df = read_labels_df['CDR'].fillna('')


def process_data(labels_df,patient):
    label = labels_df.get_value(patient.split('\\')[2], 'CDR')
    try:
        if label > 0: label=np.array([0.0,1.0])
        elif label == 0: label=np.array([1.0,0.0])
    except:
        print("unlabelled or wrongly labelled data")

    return np.array(all_slice(patient)),label

# img_data,label = process_data(cdr_df,example_file) #
# print(img_data[128])
# print(label)

mri_data = []
for num,patient in enumerate(patients):
    if num % 100 == 0:
        print(num)
    try:
        img_data,label = process_data(cdr_df,patient)
        mri_data.append([img_data,label])
    except KeyError as e:
       print('This is unlabeled data!')


np.save('three_dim_256_3.npy', mri_data)
print("file done")

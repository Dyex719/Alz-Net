from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

example_file = os.getcwd() + '/Data/disc1/OAS1_0027_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_0027_MR1_mpr_n4_anon_sbj_111.img"

def save_to_jpeg(example_file):
    '''To save the image as a JPEG that can be viewed later'''
    image_array = np.rot90(nib.load(example_file).get_data()[..., 88, 0])
    plt.imsave('saved_image.jpg', cmap = plt.cm.gray , arr = image_array)

save_to_jpeg(example_file)

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = load_img('saved_image.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='artificial_expansion', save_prefix='brain', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely

# Deep Learning for Biomedical Image Processing

## The problem statement
1. Using Convolutional Neural Networks to predict Alzheimer's disease.
2. It will help in early diagnosis of the disease and also help to cut costs in its identification by performing the work done by the Radiologist

## About Alzheimer's Disease
Alzheimer’s Disease is a progressive neurodegenerative disease, where dementia symptoms gradually worsen over a number of years. It is  the cause of 60-70% of the cases of Dementia. (Memory Loss)

The greatest known risk factor is increasing age, the majority of people with Alzheimer's being 65 and older. But Alzheimer's is not just a disease of old age. Approximately 200,000 Americans under the age of 65 have younger-onset Alzheimer’s disease. 

It is the sixth leading causing of death in the United States. 

Alzheimer’s has no current cure.

## How can we fight this?
The best way to fight this disease is early detection. This can help target the disease before irreversible brain damage or mental decline has occurred.

Although current Alzheimer's treatments cannot stop Alzheimer's from progressing, they can temporarily slow the worsening of dementia symptoms and improve quality of life for those with Alzheimer's and their caregivers.

The goal of this project was to create a basic neural network that can can differentiate between normal patients and those affected by Alzheimer based on their MRI brain scan.

## The idea behind the project
There are several visual differences between a normal brain and a brain affected with Alzheimer's. The aim was to build a neural network that could identify these differences.


<img src="https://github.com/Dyex719/Alz-Net/blob/master/Pictures/Shrivel.png" height="250" width="250"> 
<img src="https://github.com/Dyex719/Alz-Net/blob/master/Pictures/Shrinkage.png" height="250" width="250"> 
<img src="https://github.com/Dyex719/Alz-Net/blob/master/Pictures/Enlargement.png" height="250" width="250"> 

## Data
The data was procured from oasis-brains.org and consisted of the 3D MRI scans of 416 patients out of which 100 patients were diagnosed with Alzheimer.

<img src="https://github.com/Dyex719/Alz-Net/blob/master/Pictures/Cross-Sections.png" height="250" width="250"> 


## Preprocessing
As the number of slices in each 3D image is very high, in order to shrink the number of slices down to a number that can be dealt with, we averaged the pixel intensity over 10 slices, to create 16 chunks.

 The data that was obtained was of the file format Analyze 7.5, a proprietary neuroimaging file format commonly used in MRI scans and CT scans.
 
The python library Nibabel was used to load the 3D image after which other libraries like Scipy and Numpy were used to resize the image and convert the image into a 3D numpy array.

## 3D Convolutional Net in Tensorflow
In order to fully utilize the three dimensional data we decided to use a three dimensional convolutional network that can capture the spatial information of the MRI brain scan.

## Testing out other convolutional networks
The 2D CNN was built with Keras, taking the 80th slice from all the patients. The 80th slice corresponds to a middle cross section of the brain where the regions affected in the brain in Alzheimers can be viewed clearly.

## Results
The model acheived an accuracy of 79.7% on the test data after training on the 18th epoch. This was also the epoch with the second highest validation accuracy and the lowest validation loss among all 50 epochs the model had been trained on.

The training set consisted of 84 patients out of which 61 were not diseased, thus 72.6% (61/84) was the score to beat.
The results show that the model is still confused with classifying the patients that are suffering from Alzheimer.
However, it has started to learn some of the features for doing so and with more data may yield better results.
![Confusion Matrix](https://github.com/Dyex719/Alz-Net/blob/master/Pictures/Confusion_matrix.png)

## Proceeding from here

Some new approaches that can be tried include:
1. Transfer Learning:
This approach involves changing the last two layers of the pre-trained Google Inception Model that was initially trained on 10000 labels of different categories. This network has more than a dozen layers and may perform better due to its sheer complexity.
2. Artificially expanding the dataset:
Since the dataset size is very small, one can consider creating more data from the existing data by flipping, translating the image etc.

## Conclusion
Neuroimaging is among the most promising areas of research focused on early detection of Alzheimer’s disease and has great potential for treating this disease effectively in combination with appropriate medication.




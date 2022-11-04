# Weather Camera: Video Analysy Algorithms for Weather Monitoring
This repository was created by [Salvatore Ventre](https://github.com/salventre) for the thesis work related to the **Artificial Vision** subject at the University of Salerno.</br>*Academic Year: 2021-2022*

## Problem Description
In order to ensure the safety of travelers, it becomes essential to know in real time the weather conditions on the road they are traveling, so as to inform travelers of any potential danger situations. This is typically done through dedicated weather stations. However, such stations are expensive and are not always able to characterize the state of the road surface. On the other hand, many of the roads are already monitored by surveillance cameras, which can be used for artificial vision applications. Therefore, in recent years there has been a growing interest in artificial intelligence algorithms that can exploit existing cameras to detect the presence of rain, snow or fog.</br>

In this thesis a system for monitoring the weather through the video analysis will be realized, evaluating the goodness of the proposed solution and considering the actual applicability in a real context. Following the general line of work in the literature, the idea is to use a classifier, training it on a dataset of images prepared for the purpose.</br>

Chosen model, [MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2/MobileNetV2), is trained and tested on dataset based on three publicly available datasets:
* [Five-Class Weathrer image dataset (IEEE Dataport)](https://ieee-dataport.org/documents/five-class-weather-image-dataset-1);
* [Road CCTV images with associated weather data](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SV9N9F);
* [Weather Dataset - Kaggle](https://www.kaggle.com/datasets/vijaygiitk/multiclass-weather-dataset?select=dataset).

To reduce the number of erroneous image classifications, it is performed a data augmentation (Rotation, Flip, Translation, Contrast and CutOut) that helped the work of the model and in addition gave more robustness to the trained model because they simulate conditions that may occur in the reality but are not represented in the data. To evaluate the model the chosen method is cross validation: three partition were created and each one contains different samples. Also, the prediction errors were analyzed by hard mining techniques, that show errors caused by different factors and it was used the [Grad-CAM](https://keras.io/examples/vision/grad_cam/) for data interpretability, highlighting with a localization map, the important regions used for predictions.

## Documentation
For more information, read the [thesis report](https://drive.google.com/file/d/1J1d4ueD84uzCL8TG5DfvjT_ZSjMPWo5k/view?usp=share_link).
___

##### Salvatore Ventre


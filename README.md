# Footpath segmentation

This repository consists of an approach to segment footpaths by means of 
machine learning. The model architecture used in this project is based on the
architecture introduced in https://github.com/MarvinTeichmann/KittiSeg.
In comparison to the original work, the code is migrated to TensorFlow 2,
using the high-level API called Keras. KittiSeg has been used as part of
MultiNet https://github.com/MarvinTeichmann/MultiNet, described in [[1]](#1).

---
**Note**

Although the purpose and the examples are focused on footpaths/street 
segmentation, any other two-class segmentation can be trained by this approach.

---

# Example results

The following images show street segmentation after training the model with
the Kitti road dataset. The parameters used to accomplish such results are 
given in the train.ipynb and parameters/training.json files. The example images
below are taken from the testing pool of the Kitti road dataset. 

Original Image             |  Predicted Mask           |  Overlay
:-------------------------:|:-------------------------:|:-------------------------:
![](./data/examples/um_000020.png) | ![](./data/examples/um_000020_pmask.png) | ![](./data/examples/um_000020_overlay.png)
![](./data/examples/umm_000058.png) | ![](./data/examples/umm_000058_pmask.png) | ![](./data/examples/umm_000058_overlay.png)
![](./data/examples/uu_000088.png) | ![](./data/examples/uu_000088_pmask.png) | ![](./data/examples/uu_000088_overlay.png)

# Example results after training the model with different datasets

The following images show results after the model has been trained with
different datasets. The original images are taken from the Kitti road dataset,
the Deep Scene, Freiburg Forest dataset [[2]](#2) and an own dataset showing footpaths 
around Landsberg am Lech (LaL), a city in Bavaria, Germany.

The models used for this comparisons were trained with similar parameters
which differ only in the number of epochs, the dataset and the trainable layers.
The latter one only effects the model referred to as _Kitti 120e; DeepScene-Kitti-Mix 35e_. 
Here the Deep Scene dataset (with a small set of Kitti images) was trained on 
top of the model that was trained exclusively with Kitti images.

The naming convention below is roughly as follows: 
```
DATASET_NUMBER OF EPOCHS; (optional) SECOND DATASET TRAINED ON TOP_NUMBER OF EPOCHS
```

---
**Note**

Although images from the LaL dataset are taken for testing, the here mentioned 
models were not trained with them. The results show how well the models adapt
to other scenery and camera settings.

---


### Kitti images

Original Image             |  Kitti 120e           |  DeepScene 120e           |  Kitti 120e; DeepScene-Kitti-Mix 35e
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------: 
![](./data/examples/comparison/original_images/0-0.png)|![](./data/examples/comparison/kitti-120e/0-0.png)|![](./data/examples/comparison/ds-120e/0-0.png)|![](./data/examples/comparison/kitti-120e-ds-35e/0-0.png) 
![](./data/examples/comparison/original_images/0-1.png)|![](./data/examples/comparison/kitti-120e/0-1.png)|![](./data/examples/comparison/ds-120e/0-1.png)|![](./data/examples/comparison/kitti-120e-ds-35e/0-1.png)
![](./data/examples/comparison/original_images/0-2.png)|![](./data/examples/comparison/kitti-120e/0-2.png)|![](./data/examples/comparison/ds-120e/0-2.png)|![](./data/examples/comparison/kitti-120e-ds-35e/0-2.png)
![](./data/examples/comparison/original_images/0-3.png)|![](./data/examples/comparison/kitti-120e/0-3.png)|![](./data/examples/comparison/ds-120e/0-3.png)|![](./data/examples/comparison/kitti-120e-ds-35e/0-3.png)

### Deep Scene images

Original Image             |  Kitti 120e           |  DeepScene 120e           |  Kitti 120e; DeepScene Kitti Mix 35e
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------: 
![](./data/examples/comparison/original_images/1-0.png)|![](./data/examples/comparison/kitti-120e/1-0.png)|![](./data/examples/comparison/ds-120e/1-0.png)|![](./data/examples/comparison/kitti-120e-ds-35e/1-0.png)
![](./data/examples/comparison/original_images/1-1.png)|![](./data/examples/comparison/kitti-120e/1-1.png)|![](./data/examples/comparison/ds-120e/1-1.png)|![](./data/examples/comparison/kitti-120e-ds-35e/1-1.png)
![](./data/examples/comparison/original_images/1-2.png)|![](./data/examples/comparison/kitti-120e/1-2.png)|![](./data/examples/comparison/ds-120e/1-2.png)|![](./data/examples/comparison/kitti-120e-ds-35e/1-2.png)
![](./data/examples/comparison/original_images/1-3.png)|![](./data/examples/comparison/kitti-120e/1-3.png)|![](./data/examples/comparison/ds-120e/1-3.png)|![](./data/examples/comparison/kitti-120e-ds-35e/1-3.png)

### LaL images

Original Image             |  Kitti 120e           |  DeepScene 120e           |  Kitti 120e; DeepScene Kitti Mix 35e
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------: 
![](./data/examples/comparison/original_images/2-0.png)|![](./data/examples/comparison/kitti-120e/2-0.png)|![](./data/examples/comparison/ds-120e/2-0.png)|![](./data/examples/comparison/kitti-120e-ds-35e/2-0.png)
![](./data/examples/comparison/original_images/2-1.png)|![](./data/examples/comparison/kitti-120e/2-1.png)|![](./data/examples/comparison/ds-120e/2-1.png)|![](./data/examples/comparison/kitti-120e-ds-35e/2-1.png)
![](./data/examples/comparison/original_images/2-2.png)|![](./data/examples/comparison/kitti-120e/2-2.png)|![](./data/examples/comparison/ds-120e/2-2.png)|![](./data/examples/comparison/kitti-120e-ds-35e/2-2.png)
![](./data/examples/comparison/original_images/2-3.png)|![](./data/examples/comparison/kitti-120e/2-3.png)|![](./data/examples/comparison/ds-120e/2-3.png)|![](./data/examples/comparison/kitti-120e-ds-35e/2-3.png)



## References
<a id="1">[1]</a>
M. Teichmann, M. Weber, M. Zoellner, R. Cipolla, R. Urtasun.
MultiNet: Real-time Joint Semantic Reasoning for Autonomous Driving.
In 2018 IEEE Intelligent Vehicles Symposium (IV).
[2018]

<a id="2">[2]</a>
DeepScene - http://deepscene.cs.uni-freiburg.de/
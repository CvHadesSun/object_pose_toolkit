# YCB-Video dataset
It gives the 3d models and rgb images and crossponding depth images
and annotations.
You can get a complete tool in (http://bop.felk.cvut.cz).
(https://github.com/yuxng/YCB_Video_toolbox)


## model
There are 21 objects in models folder,and in per subfolder,one model
contains some files,among them,[textured.obj] and [points.xyz] contains
3d model point cloud datas.

## 0000[for example]
In this folder,a sample include 4 files(label,anntation,rgb,depth)

### label
In *-label.png image,per object is masking by it's object class number.
The class indexes are rearrange in 1~21 by orinal order.we can use
this label to get the masks of one sample.

### Annotation format
The *-meta.mat file in the YCB-Video dataset contains the following fields:
- center: 2D location of the projection of the 3D model origin in the image
- cls_indexes: class labels of the objects
- factor_depth: divde the depth image by this factor to get the actual depth vaule
- intrinsic_matrix: camera intrinsics
- poses: 6D poses of objects in the image
- rotation_translation_matrix: RT of the camera motion in 3D
- vertmap: coordinates in the 3D model space of each pixel in the image

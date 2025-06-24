Dual-Domain Teacher for Unsupervised Domain Adaptation Detection
=
Here's our paper.[paper](https://github.com/user-attachments/files/20877204/Dual-Domain_Teacher_for_Unsupervised_Domain_Adaptation_Detection.pdf)

![paper](https://github.com/user-attachments/assets/fe5e81f8-8d49-4b15-80b1-84e4ceac55ca)
Installation
=
Prerequisites
-
    * Python ≥ 3.7
    * PyTorch ≥ 1.7 and torchvision that matches the PyTorch installation.
    * Detectron2 == 0.5 (The version I used to run my code)
Our tested environment
-
    *2 3090 (4 batch size)
    *2 2080Ti (2 batch size)
Dataset download
=
    1.Download the datasets
    2.Organize the dataset as the Cityscapes and PASCAL VOC format following:
    DDT/
    └── datasets/
        └── cityscapes/
            ├── gtFine/
                ├── train/
                └── target_like/
                └── val/
        ├── leftImg8bit/
            ├── train/
            └── target_like/
            └── val/
       └── cityscapes_foggy/
            ├── gtFine/
               ├── train/
                └── source_like/
                └── val/
            ├── leftImg8bit/
                ├── train/
                └── source_like/
                └── val/
       └── VOC2012/
            ├── Annotations/
            ├── ImageSets/
            └── JPEGImages/
       └── clipark/
             ├── Annotations/
            ├── ImageSets/
            └── JPEGImages/
Training
=

    


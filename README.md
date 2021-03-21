# AutoTruckX
An experimental project for autonomous vehicle driving perception with steering angle prediction and semantic segmentation. 

# Semantic Segmentation

Detailed description can be found at [`./Semantic Segmentation/README.md`](./Semantic%20Segmentation/README.md).

* SETR: A pure transformer encoder model and a variety of decoder unsampling models to perform semantic segmentation tasks. This model was adapted from and implemented based on the paper published in December 2020 by Sixiao Zheng et al., titled [*Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective
with Transformers*](https://arxiv.org/abs/2012.15840). In particular, the SETR-PUP and SETR-MLA variants, that is, the models with progressive upsampling and multi-level feature aggregation decoders, are selected and implemented based on their state-of-the-art performance on benchmark datasets.
* TransUNet: A UNet-transformer hybrid model that uses UNet to extract high-resolution feature maps, a transformer to tokenize and encode images, and a UNet-like mechanism to upsample in decoder using previously-extracted feature maps. This model was adapted from and implemented based on the paper published in February 2021 by Jieneng Chen et al., titled [*TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation*](https://arxiv.org/abs/2102.04306).
* UNet: the well-known UNet model. This variant of UNet, which is 4-layers deep in the architecture, is adapted and implemented based on the paper published in November 2018 by Ari Silburt et al., titled [*Lunar Crater Identification via Deep Learning*](https://arxiv.org/abs/1803.02192).

| SETR | TransUNet | UNet |
| ------------- | ------------- | ------------- |
| ![What is this](./Semantic%20Segmentation/visualizations/SETR_model.png)  | ![What is this](./Semantic%20Segmentation/visualizations/TransUNet_model.png)  | ![What is this](./Semantic%20Segmentation/visualizations/unet_model.png)|

Figures are authored in and extracted from the original papers respectively.

 ![What is this](./Semantic%20Segmentation/visualizations/output1_2.jpg) 
 ![What is this](./Semantic%20Segmentation/visualizations/output2_2.jpg) 
 ![What is this](./Semantic%20Segmentation/visualizations/output3_2.jpg) 
 ![](./Semantic%20Segmentation/visualizations/combined.gif)

It can be observed that the model can perform reasonable semantic segmentation task when inferenced on test image and videos. 

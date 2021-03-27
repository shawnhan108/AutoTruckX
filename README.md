# AutoTruckX
An experimental project for autonomous vehicle driving perception with steering angle prediction and semantic segmentation. 

# Semantic Segmentation

**Detailed description can be found at [`./Semantic Segmentation/README.md`](./Semantic%20Segmentation).**

* SETR: A pure transformer encoder model and a variety of decoder unsampling models to perform semantic segmentation tasks. This model was adapted from and implemented based on the paper published in December 2020 by Sixiao Zheng et al., titled [*Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective
with Transformers*](https://arxiv.org/abs/2012.15840). In particular, the SETR-PUP and SETR-MLA variants, that is, the models with progressive upsampling and multi-level feature aggregation decoders, are selected and implemented based on their state-of-the-art performance on benchmark datasets.
* TransUNet: A UNet-transformer hybrid model that uses UNet to extract high-resolution feature maps, a transformer to tokenize and encode images, and a UNet-like mechanism to upsample in decoder using previously-extracted feature maps. This model was adapted from and implemented based on the paper published in February 2021 by Jieneng Chen et al., titled [*TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation*](https://arxiv.org/abs/2102.04306).
* UNet: the well-known UNet model. This variant of UNet, which is 4-layers deep in the architecture, is adapted and implemented based on the paper published in November 2018 by Ari Silburt et al., titled [*Lunar Crater Identification via Deep Learning*](https://arxiv.org/abs/1803.02192).

| SETR | TransUNet | UNet |
| ------------- | ------------- | ------------- |
| ![What is this](./Semantic%20Segmentation/visualizations/SETR_model.png)  | ![What is this](./Semantic%20Segmentation/visualizations/TransUNet_model.png)  | ![What is this](./Semantic%20Segmentation/visualizations/unet_model.png)|

Figures above are authored in and extracted from the original papers respectively.
 ![](./Semantic%20Segmentation/visualizations/combined.gif)
 ![What is this](./Semantic%20Segmentation/visualizations/output1_2.jpg) 
 ![What is this](./Semantic%20Segmentation/visualizations/output2_2.jpg) 
 ![What is this](./Semantic%20Segmentation/visualizations/output3_2.jpg) 


It can be observed that the model can perform reasonable semantic segmentation task when inferenced on test image and videos. 

# Steering Angle Prediction

**Detailed description can be found at [`./Steering Angle Prediction/README.md`](./Steering%20Angle%20Prediction).**

* TruckNN: A CNN model adapted and modified from NVIDIA's 2016 paper [*End to End Learning for Self-Driving Cars*](https://arxiv.org/abs/1604.07316). The original model was augmented with batch normalization layers and dropout layers.
* TruckResnet50: A CNN transfer learning model utilizing feature maps extracted by ResNet50, connected to additional fully-connected layers. This model was adapated and modified from Du et al.'s 2019 paper [*Self-Driving Car Steering Angle Prediction Based on Image Recognition*](https://arxiv.org/abs/1912.05440). The first 141 layers of the ResNet50 layers (instead of the first 45 layers as in the original paper) were frozen from updating. Dimensions of the fully-connected layers were also modified.
* TruckRNN: A Conv3D-LSTM model, also based on and modified from Du et al.'s 2019 paper mentioned above, was also experimented. The model consumes a sequence of 15 consecutive frames as input, and predicts the steering angle at the last frame. Comparing to the original model, maxpooling layers were omitted and batch normalization layers were introduced. 5 convolutional layers were implemented with the last convolutional layer connected with residual output, followed by two LSTM layers, which is rather different to the model architecture proposed in the paper.

| TruckNN | TruckResnet50 | TruckRNN |
| ------------- | ------------- | ------------- |
| ![What is this](./Steering%20Angle%20Prediction/visualizations/nvidia_model.png)  | ![What is this](./Steering%20Angle%20Prediction/visualizations/3dLSTM_model.png)  | ![What is this](./Steering%20Angle%20Prediction/visualizations/Res_model.png)|

Figures above are authored in and extracted from the original papers respectively.

 ![What is this](./Steering%20Angle%20Prediction/visualizations/model3_output.jpg) ![What is this](./Steering%20Angle%20Prediction/visualizations/model3_output2.jpg) 
 ![](./Steering%20Angle%20Prediction/visualizations/demo.gif)

For further visualization, saliency maps of the last Resnet50 Convolutional layer (layer4) can be observed as below:
![What is this](./Steering%20Angle%20Prediction/visualizations/resnet_salient_map1.png) ![What is this](./Steering%20Angle%20Prediction/visualizations/resnet_salient_map2.png)

The model seems to possess salient features on the road.

# Semantic Segmentation

### 1 Model Architecture

Three models were experimented:
* SETR: A pure transformer encoder model and a variety of decoder unsampling models to perform semantic segmentation tasks. This model was adapted from and implemented based on the paper published in December 2020 by Sixiao Zheng et al., titled [*Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective
with Transformers*](https://arxiv.org/abs/2012.15840). In particular, the SETR-PUP and SETR-MLA variants, that is, the models with progressive upsampling and multi-level feature aggregation decoders, are selected and implemented based on their state-of-the-art performance on benchmark datasets.
* TransUNet: A UNet-transformer hybrid model that uses UNet to extract high-resolution feature maps, a transformer to tokenize and encode images, and a UNet-like mechanism to upsample in decoder using previously-extracted feature maps. This model was adapted from and implemented based on the paper published in February 2021 by Jieneng Chen et al., titled [*TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation*](https://arxiv.org/abs/2102.04306).
* UNet: the well-known UNet model. This variant of UNet, which is 4-layers deep in the architecture, is adapted and implemented based on the paper published in November 2018 by Ari Silburt et al., titled [*Lunar Crater Identification via Deep Learning*](https://arxiv.org/abs/1803.02192).

### 2 Classification Loss

Two loss functions were experimented:
* Cross Entropy between the predicted and groundtruth class assignments of the pixels in a given frame, as implemented in [torch.nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html). 
* As suggested in the TransUNet paper, a combination of the Cross Entropy loss as above, and the Dice loss between predicted and groundtruth class assignments of the pixels in a given frame. The loss function is implemented in `utils.py`. The final loss is a 1:1 weighted sum of the Cross Entropy loss and the Dice loss.


### 3 Dataset

The models were trained on the [Cityscapes dataset](https://www.cityscapes-dataset.com). In particular, the [dataset](https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/) used for training is a subset of processed subsample created for the [Pix2Pix project](https://phillipi.github.io/pix2pix/) and the paper published in Nov 2018 by Phillip et al., titled [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004). The dataset contains 2975 training images and 500 validation images in which each image (256 x 256) is attached with an annotated classification label map. The images were extracted from videos recorded in Germany.

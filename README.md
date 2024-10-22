# Introduction
Semantic segmentation is one of the most important tasks in computer vision. For this project, I worked with the CamVid dataset for semantic segmentation. I step into this project inspired by [1]( https://arxiv.org/pdf/1411.4038) and what I read in this [2](https://github.com/khalequzzamanlikhon/DeepLearning-ComputerVision/blob/master/08-Segmentation-Detection/01-Semantic-Segmentation.ipynb) course. I applied knowledge of Fully Convolutional Networks from both references, keeping resnet50 as the backbone model followed by the custom decoder. Further, I also explored two more models
and compared all of these three models. The other two models are- U-net architecture with attention mechanism and deeplabv3+ architecture explored in [3](https://arxiv.org/pdf/1802.02611). For the loss function, I considered the combined loss function which used both dice and Jaccard loss. Applying suitable data augmentation techniques I got mIoU from the abovementioned architectures are 34.14%, 45.72%, and 38.13% respectively.


# Dataset and Preprocessing 

I used The Cambridge-driving Labeled Video Database (CamVid) dataset for this task which has 32 classes. The dataset comprises 6 folders three of which train,val, and test images, and the others are labels 
associated with these image folders. Moreover, there is also a CSV file that holds the color channel values associated with the classes. I used it to convert images from RGB to classes and vice versa. I applied
different data augmentation techniques which are listed below.

  - **Training Data Transformations:**
    
     - Resize(400, 520): Resize the input images to a larger size.
     - RandomCrop(352, 480): Crops a random region of the image to the target size.
     - HorizontalFlip: Flips the image horizontally with a probability of 0.5.
     - Rotate: Rotates the image randomly within a range of ±15 degrees.
     - GaussianBlur: Applies Gaussian blur with a limit of (3, 5) for a probability of 0.3.
     - ColorJitter: Adjusts brightness, contrast, saturation, and hue.
     - Normalize: Normalize the pixel values using the mean (0.390, 0.405, 0.414) and standard deviation (0.274, 0.285, 0.297).
     - ToTensorV2: Converts the image into a PyTorch tensor.


 - **Validation and Test Data Transformations:**
     - Resize(352, 480): Resize the images to the target size.
     - Normalize Same normalization as the training data.
     - ToTensorV2: Converts the image to a tensor.


# Model Architecture:

I have used three different architectures. FCN, U-net, and deeplabv3+

- **Fully Convolutional Network**:

                                                    
                                                         Input (352, 480, 3)
                                                               |
                                                               V
                                             +-----------------------------------+
                                             |                                   |
                                             |          ResNet-50 Encoder        |
                                             |         (Pretrained Backbone)     |
                                             +-----------------------------------+
                                                               |
                                                               V
                                                 +-----------------------------+
                                                 |   Encoder Output (7x7x2048)  |
                                                 +-----------------------------+
                                                               |
                                                               V
                                                 +-----------------------------+
                                                 |      ConvTranspose Layer     |
                                                 |       (2048 -> 1024)         |
                                                 +-----------------------------+
                                                               |
                                                               V
                                                 +-----------------------------+
                                                 |        BatchNorm + ReLU      |
                                                 +-----------------------------+
                                                               |
                                                               V
                                                 +-----------------------------+
                                                 |      ConvTranspose Layer     |
                                                 |       (1024 -> 512)          |
                                                 +-----------------------------+
                                                               |
                                                               V
                                                 +-----------------------------+
                                                 |        BatchNorm + ReLU      |
                                                 +-----------------------------+
                                                               |
                                                               V
                                                 +-----------------------------+
                                                 |      ConvTranspose Layer     |
                                                 |       (512 -> 256)           |
                                                 +-----------------------------+
                                                               |
                                                               V
                                                 +-----------------------------+
                                                 |        BatchNorm + ReLU      |
                                                 +-----------------------------+
                                                               |
                                                               V
                                                 +-----------------------------+
                                                 |      ConvTranspose Layer     |
                                                 |       (256 -> 128)           |
                                                 +-----------------------------+
                                                               |
                                                               V
                                                 +-----------------------------+
                                                 |        BatchNorm + ReLU      |
                                                 +-----------------------------+
                                                               |
                                                               V
                                                 +-----------------------------+
                                                 |      ConvTranspose Layer     |
                                                 |   (128 -> num_classes=32)    |
                                                 +-----------------------------+
                                                               |
                                                               V
                                                 +-----------------------------+
                                                 |   Upsample to (352, 480)     |
                                                 |  (Bilinear Interpolation)    |
                                                 +-----------------------------+
                                                               |
                                                               V
                                                  Output (352, 480, 32 Classes)

    


 - **U-net with an attention mechanism**

                                                            Input (352, 480, 3)
                                                               |
                                                               V
                                             +-----------------------------------+
                                             |                                   |
                                             |       ResNet-50 Encoder Layers    |
                                             |     (Pretrained Feature Extractor)|
                                             +-----------------------------------+
                                                   |       |       |      |
                                                 e1        e2      e3     e4
                                               (64)       (256)   (512)  (1024)
                                                               |
                                                               V
                                              +-----------------------------------+
                                              |   ResNet-50 Encoder Block (e5)    |
                                              |         (2048 channels)           |
                                              +-----------------------------------+
                                                               |
                                                               V
                                                 +-----------------------------+
                                                 |    Attention Block (2048)    |
                                                 |         + Decoder Block      |
                                                 |    (2048 + 1024 -> 1024)     |
                                                 +-----------------------------+
                                                               |
                                                               V
                                                 +-----------------------------+
                                                 |    Attention Block (1024)    |
                                                 |         + Decoder Block      |
                                                 |     (1024 + 512 -> 512)      |
                                                 +-----------------------------+
                                                               |
                                                               V
                                                 +-----------------------------+
                                                 |    Attention Block (512)     |
                                                 |         + Decoder Block      |
                                                 |     (512 + 256 -> 256)       |
                                                 +-----------------------------+
                                                               |
                                                               V
                                                 +-----------------------------+
                                                 |    Attention Block (256)     |
                                                 |         + Decoder Block      |
                                                 |      (256 + 64 -> 64)        |
                                                 +-----------------------------+
                                                               |
                                                               V
                                                 +-----------------------------+
                                                 |       Final Conv Layer       |
                                                 |   (64 -> num_classes=32)     |
                                                 +-----------------------------+
                                                               |
                                                               V
                                                 +-----------------------------+
                                                 |   Upsample to (352, 480)     |
                                                 |  (Bilinear Interpolation)    |
                                                 +-----------------------------+
                                                               |
                                                               V
                                                  Output (352, 480, 32 Classes)


- **Deeplabv3+**

                                                     Input (352, 480, 3)
                                                               |
                                                               V
                                             +-----------------------------------+
                                             |                                   |
                                             |      ResNet-50 Encoder Backbone   |
                                             |    (Pretrained Feature Extractor) |
                                             +-----------------------------------+
                                                               |
                                                               V
                                                 +-----------------------------+
                                                 |      Atrous Spatial Pyramid  |
                                                 |        Pooling (ASPP)        |
                                                 |   (With Multiple Dilation    |
                                                 |     Rates for Context)       |
                                                 +-----------------------------+
                                                               |
                                                               V
                                                 +-----------------------------+
                                                 |    Upsample + Concatenation  |
                                                 |     with Low-Level Features  |
                                                 |    (Low-Level + ASPP Output) |
                                                 +-----------------------------+
                                                               |
                                                               V
                                                 +-----------------------------+
                                                 |   3x3 Conv Layer + BatchNorm |
                                                 |     + ReLU Activation        |
                                                 +-----------------------------+
                                                               |
                                                               V
                                                 +-----------------------------+
                                                 |      Conv Layer (256 ->      |
                                                 |  num_classes=32)             |
                                                 +-----------------------------+
                                                               |
                                                               V
                                                 +-----------------------------+
                                                 |   Upsample to (352, 480)     |
                                                 |  (Bilinear Interpolation)    |
                                                 +-----------------------------+
                                                               |
                                                               V
                                                  Output (352, 480, 32 Classes)

                                                       
# Training Setup

**4.1 Hyperparameters**

  - Optimizers: Adam optimizer is used for all three models, with a learning rate of 1e-4 and weight decay:
  - weight_decay=1e-4 for FCN
  -  batch_size=4; Used this small batch size due to lack of GPU
  - weight_decay=1e-5 for UNet and DeepLabV3+
  - Learning Rate Scheduler: ReduceLROnPlateau reduces the learning rate by a factor of 0.1 if no improvement is seen for 3 epochs.
  - Early Stopping: Patience of 7 epochs with a delta threshold of 0.001.
  - 
**4.2 Loss Function**
The combined loss function is comprised of dice loss and Jaccard loss used to handle any class imbalance. I used weight .7 and .3 for dice loss and Jaccard loss respectively.

 - **Dice Loss:** Measures the overlap between the predicted and ground truth masks. It helps the model focus on the regions of interest and is effective for class imbalance. The equation for dice loss is:

   $\[\text{Dice Loss} = 1 - \frac{2 \sum_{i=1}^{N} p_i g_i + \epsilon}{\sum_{i=1}^{N} p_i + \sum_{i=1}^{N} g_i + \epsilon}\]$

  - **Jaccard Loss (IoU):** Similar to Dice loss but uses the intersection-over-union measure. This loss is beneficial for ensuring high accuracy in boundary regions. The equation is:
    
      $\[\text{Jaccard Loss} = 1 - \frac{\sum_{i=1}^{N} p_i g_i + \epsilon}{\sum_{i=1}^{N} p_i + \sum_{i=1}^{N} g_i - \sum_{i=1}^{N} p_i g_i + \epsilon}\]$

The final combined loss can be formulated as:  $\[\text{Combined Loss} = \alpha \cdot \text{Dice Loss} + \beta \cdot \text{Jaccard Loss}\]$

# Experiments and Results

**Training Settings**

  - Epochs: The training was conducted for up to 100 epochs with early stopping criteria applied.
  - Batch Size: A batch size of 8 was used.

**Callbacks** 

  - Learning Rate Scheduling and Early Stopping: Optimizers were configured to use learning rate scheduling (ReduceLROnPlateau) and early stopping mechanisms.

**Accuracy and Loss Curves**:


  - **Loss and accuracy curve for Fully Convolutional Network**
  
![loss-acc-fcn](/output_images/loss-acc-fcn.png)

 - **loss and accuracy curve for u-net**

![loss-curve](/output_images/loss-acc-unet.png)

 - **Loss and accuracy curves for deeplabv3+**
 
![loss-acc-curve-deeplab](/output_images/loss-acc-deeplab.png)

**Results:** 

| Model                        | mIoU   |
|------------------------------|--------|
| Fully Convolutional Network  | 34.14% |
| U-Net with Attention         | 45.72% |
| DeepLabV3+                   | 38.13% |


**output-diagram**

- **a prediction on the test set for fcn**

![output-fcn](/output_images/output-fcn.png)

- **a prediction on the test set for u-net**

![output-fcn](/output_images/output-unet.png)

- **a prediction on the test set for deeplabv3+**

![output-fcn](/output_images/output-deeplab.png)


# Future works

with these setups, I didn't get a good average Intersection over Union (IoU) score from any of these three models. Although deeplabv3+ should result in more than 80%, here I got only 38.13%.
I will explore more especially with deeplabv3+ to get a score as it is mentioned in [3]( https://arxiv.org/pdf/1802.02611). Moreover, I will apply post-processing such as Conditional Random Fields (CRF), Morphological Operations (e.g., dilation, erosion), etc. Moreover, I will experiment with different hyperparameters when I get available resources (GPU).

**Note**: you can also find the code in [kaggle](https://www.kaggle.com/code/likhon148/semantic-segmentation-pytorch-scratch)
## References
1. https://arxiv.org/pdf/1411.4038
2. https://github.com/khalequzzamanlikhon/DeepLearning-ComputerVision/blob/master/08-Segmentation-Detection/01-Semantic-Segmentation.ipynb
3. https://arxiv.org/pdf/1802.02611
4. https://www.kaggle.com/code/likhon148/semantic-segmentation-pytorch-scratch

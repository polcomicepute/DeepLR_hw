# DeepLR_hw

- Test Accuracy = 84.48%
  - [W&B_link](https://wandb.ai/polcom/2022707003_%EC%9D%B4%ED%98%9C%EB%AF%BC_pytorchlightning_Cifar)
- Model = Resnet18
- Dataset = Cifar10 
  - 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
 

- Traninig Details
  - Batch Size = 64 
  - LR_scheduler = OneCycle, 0.1
  - Optimizer = Adam
  - Learning Rate = 0.005
  - Device = 4 gpus, DDP

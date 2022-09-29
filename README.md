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

- 고찰
  - Dataset의 크기, resolution이 작으므로 Batchsize 256인 경우보다 64일 때 더 높은 성적
    ![Uploading 스크린샷 2022-09-29 오후 10.33.26.png…]()

  - OneCycle Laerning rate scheduler이용 여부에 따라 차이 발생
  ![스크린샷 2022-09-29 오후 10 36 45](https://user-images.githubusercontent.com/52686915/193046455-bdef6948-aecd-4138-bf9f-36d7c2937dea.png)

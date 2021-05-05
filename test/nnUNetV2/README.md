## Usage Instructions

All the loss functions have been tested with the nnUNetTrainerV2 in the latest [nnUNet](https://github.com/MIC-DKFZ/nnUNet).

0. Prerequisites: install [nnUNet](https://github.com/MIC-DKFZ/nnUNet).
1. Download the loss functions: `git clone https://github.com/JunMa11/SegLoss.git`
2. Copy `SegLoss/test/nnUNetV2/loss_functions` and `SegLoss/test/nnUNetV2/network_training` to `nnUNet/nnunet/training`
3. To Train your model, replacing `nnUNetTrainer` by the new trainer. e.g., if you want to train UNet with Dice loss, run:
> nnUNet_train 3d_fullres nnUNetTrainerV2_Loss_DiceTopK10 TaskXX_MY_DATASET FOLD

## Datasets

- Liver Tumor: [LiTS](https://competitions.codalab.org/competitions/17094) + [MSD-Task08](http://medicaldecathlon.com/)
- Pancreas: [NIH-Pancreas](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT) + [MSD-Task07](http://medicaldecathlon.com/)
- Multi-organ: [Zenodo](http://doi.org/10.5281/zenodo.1169361)


## Results 

> In nnU-Net V2, deep supervision is added to the default U-Net. The optimizer is SGD with momentum rather than Adam.

The associated segmentation [results](https://zenodo.org/record/4738480) have been released.

| Loss       | LiverTumor-DSC | LiverTumor-NSD | Pancreas-DSC | Pancreas-NSD | Multiorgan-DSC | Multiorgan-NSD |
|------------|:--------------:|:--------------:|:------------:|:------------:|:--------------:|:--------------:|
| CE         |     0.6415     |     0.4698     |    0.8338    |    0.6566    |     0.8570     |     0.7368     |
| Dice       |     0.6200     |     0.4592     |    0.8399    |    0.6663    |     0.8577     |     0.7416     |
| DiceCE     |     0.6281     |     0.4678     |    0.8410    |    0.6691    |     0.8626     |     0.7488     |
| DiceFocal  |     0.6303     |     0.4705     |    0.8401    |    0.6691    |     0.8631     |     0.7501     |
| DiceTopK10 |     0.6691     |     0.5095     |    0.8387    |    0.6661    |     0.8636     |     0.7483     |
| TopK10     |     0.6512     |     0.4849     |    0.8383    |    0.6649    |     0.8560     |     0.7378     |




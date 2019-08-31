## Usage Instructions

All the loss functions have been tested with [nnUNet](https://github.com/MIC-DKFZ/nnUNet).

0. Prerequisites: install [nnUNet](https://github.com/MIC-DKFZ/nnUNet).
1. Download the loss functions: `git clone https://github.com/JunMa11/SegLoss.git`
2. Copy `SegLoss/test/loss_functions` and `SegLoss/test/network_training` to `nnUNet/nnunet/training`
3. To Train your model, replacing `nnUNetTrainer` by the new trainer. e.g., if you want to train UNet with Dice loss, run:
> python run/run_training.py 3d_fullres nnUNetTrainer_Dice TaskXX_MY_DATASET FOLD --ndet

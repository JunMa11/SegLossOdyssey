## Usage Instructions

All the loss functions have been tested with [nnUNet](https://github.com/MIC-DKFZ/nnUNet).

0. Prerequisites: install [nnUNet](https://github.com/MIC-DKFZ/nnUNet).
1. Download the loss functions: `git clone https://github.com/JunMa11/SegLoss.git`
2. Copy `SegLoss/test/loss_functions` and `SegLoss/test/network_training` to `nnUNet/nnunet/training`
3. To Train your model, replacing `nnUNetTrainer` by the new trainer. e.g., if you want to train UNet with Dice loss, run:
> python run/run_training.py 3d_fullres nnUNetTrainer_Dice TaskXX_MY_DATASET FOLD --ndet



## Results

|     Loss     |  Liver DSC |  Liver NSD | Liver Tumor DSC | Liver Tumor NSD | Pancreas DSC | Pancreas NDS | Multi-organ DSC | Multi-organ NDS |
|:------------:|:----------:|:----------:|:---------------:|:---------------:|:------------:|:------------:|:---------------:|:---------------:|
|     Asym     |   0.9315   |   0.6905   |      0.6134     |      0.4114     |    0.8234    |    0.6239    |      0.7526     |      0.6088     |
|      CE      |   0.9672   |   0.7962   |      0.5641     |      0.3715     |    0.8221    |    0.6320    |      0.8483     |      0.7200     |
|     Dice     |   0.9547   |   0.7598   |    **0.6208**   |    **0.4326**   |    0.8270    |    0.6428    |      0.8449     |      0.7138     |
|    DiceCE    |   0.9624   |   0.7743   |      0.6009     |      0.4114     |    0.8249    |    0.6298    |      0.8512     |    **0.7293**   |
|   DiceFocal  |   0.9566   |   0.7583   |      0.5951     |      0.4078     |  **0.8416**  |  **0.6721**  |    **0.8554**   |    **0.7339**   |
|   DiceTopK   | **0.9690** | **0.8092** |    **0.6125**   |      0.4208     |    0.8375    |    0.6598    |      0.8512     |    **0.7308**   |
|      ELL     |   0.9347   |   0.7254   |      0.5903     |      0.4047     |    0.8344    |    0.6508    |      0.8375     |      0.6689     |
|     Focal    |   0.9587   |   0.7528   |      0.4675     |      0.2781     |    0.8016    |    0.6033    |      0.8173     |      0.6642     |
| FocalTversky |   0.9320   |   0.6986   |      0.6193     |      0.4178     |    0.8230    |    0.6190    |      0.7497     |      0.6013     |
|     GDice    |   0.9474   |   0.7337   |      0.5486     |      0.3501     |    0.8285    |    0.6478    |      0.0132     |      0.0018     |
|      IoU     |   0.9568   |   0.7709   |    **0.6079**   |      0.4273     |    0.8353    |    0.6605    |      0.8439     |      0.7160     |
|      pCE     |   0.9655   |   0.7852   |      0.5876     |      0.3829     |    0.8358    |    0.6579    |      0.8349     |      0.6967     |
|    pGDice    |   0.6433   |   0.3401   |      0.4526     |      0.2879     |    0.8156    |    0.6203    |      0.0595     |      0.0209     |
|      SS      |   0.9591   |   0.7571   |      0.4527     |      0.1941     |    0.7799    |    0.4781    |      0.7589     |      0.4958     |
|     TopK     |   0.6932   |   0.1080   |      0.5995     |      0.4051     |    0.8406    |    0.6709    |      0.8527     |    **0.7323**   |
|    Tversky   |   0.9390   |   0.6991   |      0.6120     |      0.4045     |    0.8260    |    0.6249    |      0.8371     |      0.6787     |
|      WCE     |   0.8284   |   0.2665   |      0.2697     |      0.0314     |    0.4744    |    0.0496    |      0.6911     |      0.2326     |

![Violin Plot](https://github.com/JunMa11/SegLoss/blob/master/test/LossPK.PNG)


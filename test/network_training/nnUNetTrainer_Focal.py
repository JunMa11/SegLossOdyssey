from nnunet.training.loss_functions.focal_loss import FocalLoss
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper


class nnUNetTrainer_Focal(nnUNetTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        self.apply_nonlin = softmax_helper
        self.loss = FocalLoss(apply_nonlin=self.apply_nonlin, alpha=0.5, gamma=2, smooth=1e-5)

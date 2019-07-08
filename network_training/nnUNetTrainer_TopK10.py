from nnunet.training.loss_functions.TopK_loss import TopKLoss
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_TopK10(nnUNetTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        self.k = 10
        self.loss = TopKLoss(k=self.k)

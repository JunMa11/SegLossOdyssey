from nnunet.training.loss_functions.ND_Crossentropy import WeightedCrossEntropyLoss
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
import torch


class nnUNetTrainerWCE(nnUNetTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super(nnUNetTrainerWCE, self).__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        self.weight = torch.cuda.FloatTensor([0.2,0.8]) # pre-defined according to the task
        self.loss = WeightedCrossEntropyLoss(self.weight)

from nnunet.training.loss_functions.ND_Crossentropy import WeightedCrossEntropyLoss
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
import torch


class nnUNetTrainerWCET4(nnUNetTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super(nnUNetTrainerWCET4, self).__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        self.weight = torch.cuda.FloatTensor([1.0, 179.3,650.6,276.8,1243.0,512.7,30.44,118.1,552.0]) # pre-defined according to the task
        self.loss = WeightedCrossEntropyLoss(self.weight)

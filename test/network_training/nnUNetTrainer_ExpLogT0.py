from nnunet.training.loss_functions.dice_loss import ExpLog_loss
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
import torch


class nnUNetTrainer_ExpLogT0(nnUNetTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        self.weight = torch.cuda.FloatTensor([1.0,42.8])
        self.loss = ExpLog_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'square': False}, {'weight':self.weight})

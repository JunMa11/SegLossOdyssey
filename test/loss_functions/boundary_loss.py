import torch
from nnunet.training.loss_functions.TopK_loss import TopKLoss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.training.loss_functions.ND_Crossentropy import CrossentropyND
from nnunet.utilities.tensor_utilities import sum_tensor
from torch import nn
from scipy.ndimage import distance_transform_edt
from skimage import segmentation as skimage_seg
import numpy as np

def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """

        """
        super(SoftDiceLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return 1-dc

def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM) 
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """

    img_gt = img_gt.astype(np.uint8)

    gt_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(1, out_shape[1]): # channel
            posmask = img_gt[b][c].astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance_transform_edt(posmask)
                negdis = distance_transform_edt(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = negdis - posdis
                sdf[boundary==1] = 0
                gt_sdf[b][c] = sdf

    return gt_sdf


class BDLoss(nn.Module):
    def __init__(self):
        """
        compute boudary loss
        only compute the loss of foreground
        ref: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L74
        """
        super(BDLoss, self).__init__()
        # self.do_bg = do_bg

    def forward(self, net_output, gt):
        """
        net_output: (batch_size, class, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        bound: precomputed distance map, shape (batch_size, class, x,y,z)
        """
        net_output = softmax_helper(net_output)
        with torch.no_grad():
            if len(net_output.shape) != len(gt.shape):
                gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(net_output.shape)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)
            gt_sdf = compute_sdf(y_onehot.cpu().numpy(), net_output.shape)

        phi = torch.from_numpy(gt_sdf)
        if phi.device != net_output.device:
            phi = phi.to(net_output.device).type(torch.float32)
        # pred = net_output[:, 1:, ...].type(torch.float32)
        # phi = phi[:,1:, ...].type(torch.float32)

        multipled = torch.einsum("bcxyz,bcxyz->bcxyz", net_output[:, 1:, ...], phi[:, 1:, ...])
        bd_loss = multipled.mean()

        return bd_loss



class DC_and_BD_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, bd_kwargs, aggregate="sum"):
        super(DC_and_BD_loss, self).__init__()
        self.aggregate = aggregate
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        self.bd = BDLoss(**bd_kwargs)
        

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        bd_loss = self.bd(net_output, target)
        if self.aggregate == "sum":
            result = 0.01*dc_loss + (1-0.01)*bd_loss
        else:
            raise NotImplementedError("nah son") 
        return result       


#####################################################

def compute_gt_dtm(img_gt, out_shape):
    """
    compute the distance transform map of foreground in ground gruth.
    input: segmentation, shape = (batch_size, class, x, y, z)
    output: the foreground Distance Map (SDM) 
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    """

    fg_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(1, out_shape[1]): # class; exclude the background class
            posmask = img_gt[b][c].astype(np.bool)
            if posmask.any():
                posdis = distance_transform_edt(posmask)
                fg_dtm[b][c] = posdis

    return fg_dtm

def compute_pred_dtm(img_gt, out_shape):
    """
    compute the distance transform map of foreground in prediction.
    input: segmentation, shape = (batch_size, class, x, y, z)
    output: the foreground Distance Map (SDM) 
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    """

    fg_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(1, out_shape[1]): # class; exclude the background class
            posmask = img_gt[b][c]>0.5
            if posmask.any():
                posdis = distance_transform_edt(posmask)
                fg_dtm[b][c] = posdis

    return fg_dtm

class HDLoss(nn.Module):
    def __init__(self):
        """
        compute haudorff loss for binary segmentation
        https://arxiv.org/pdf/1904.10030v1.pdf        
        """
        super(HDLoss, self).__init__()


    def forward(self, net_output, gt):
        """
        net_output: (batch_size, c, x,y,z)
        target: ground truth, shape: (batch_size, c, x,y,z)
        """
        net_output = softmax_helper(net_output)
        # one hot code for gt
        with torch.no_grad():
            if len(net_output.shape) != len(gt.shape):
                gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(net_output.shape)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)
        # print('hd loss.py', net_output.shape, y_onehot.shape)

        with torch.no_grad():
            pc_dist = compute_pred_dtm(net_output.cpu().numpy(), net_output.shape)
            gt_dist = compute_gt_dtm(y_onehot.cpu().numpy(), net_output.shape)
            dist = pc_dist**2 + gt_dist**2 # \alpha=2 in eq(8)
            # print('pc_dist.shape: ', pc_dist.shape, 'gt_dist.shape', gt_dist.shape)
        
        pred_error = (net_output - y_onehot)**2

        dist = torch.from_numpy(dist)
        if dist.device != pred_error.device:
            dist = dist.to(pred_error.device).type(torch.float32)

        multipled = torch.einsum("bcxyz,bcxyz->bcxyz", pred_error[:,1:,...], dist[:,1:,...])
        hd_loss = multipled.mean()

        return hd_loss



class DC_and_HD_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, hd_kwargs, aggregate="sum"):
        super(DC_and_HD_loss, self).__init__()
        self.aggregate = aggregate
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        self.hd = HDLoss(**hd_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        hd_loss = self.hd(net_output, target)
        if self.aggregate == "sum":
            with torch.no_grad():
                alpha = hd_loss / (dc_loss + 1e-5 )
            result = alpha * dc_loss + hd_loss
        else:
            raise NotImplementedError("nah son")
        return result  



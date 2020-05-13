import torch
from torch.autograd import Variable
import torch.nn as nn
from munkres import Munkres
import numpy as np
import time
import torch.nn.functional as F
torch.manual_seed(0)
import kornia

def MaskedNLL(target, probs, balance_weights=None):
    # adapted from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
    """
    Args:
        target: A Variable containing a LongTensor of size
            (batch, ) which contains the index of the true
            class for each corresponding step.
        probs: A Variable containing a FloatTensor of size
            (batch, num_classes) which contains the
            softmax probability for each class.
        sw: A Variable containing a LongTensor of size (batch,)
            which contains the mask to apply to each element in a batch.
    Returns:
        loss: Sum of losses with applied sample weight
    """
    log_probs = torch.log(probs)

    if balance_weights is not None:
        balance_weights = balance_weights.cuda()
        log_probs = torch.mul(log_probs, balance_weights)

    losses = -torch.gather(log_probs, dim=1, index=target)
    return losses.squeeze()

def StableBalancedMaskedBCE(target, out, balance_weight = None):
    """
    Args:
        target: A Variable containing a LongTensor of size
            (batch, N) which contains the true binary mask.
        out: A Variable containing a FloatTensor of size
            (batch, N) which contains the logits for each pixel in the output mask.
        sw: A Variable containing a LongTensor of size (batch,)
            which contains the mask to apply to each element in a batch.
    Returns:
        loss: Sum of losses with applied sample weight
    """
    if balance_weight is None:
        num_positive = target.sum()
        num_negative = (1 - target).sum()
        total = num_positive + num_negative
        balance_weight = num_positive / total

    max_val = (-out).clamp(min=0)
    # bce with logits
    loss_values =  out - out * target + max_val + ((-max_val).exp() + (-out - max_val).exp()).log()
    loss_positive = loss_values*target
    loss_negative = loss_values*(1-target)
    losses = (1-balance_weight)*loss_positive + balance_weight*loss_negative

    return losses.squeeze()


def softIoU(target, out, e=1e-6):

    """
    Args:
        target: A Variable containing a LongTensor of size
            (batch, N) which contains the true binary mask.
        out: A Variable containing a FloatTensor of size
            (batch, N) which contains the logits for each pixel in the output mask.
        sw: A Variable containing a LongTensor of size (batch,)
            which contains the mask to apply to each element in a batch.
    Returns:
        loss: Sum of losses with applied sample weight
    """


    out = torch.sigmoid(out)

    num = (out * target).sum(1, True)
    den = (out + target - out * target).sum(1, True) + e
    iou = num / den

    cost = (1 - iou)

    return cost.squeeze()



#def FocalLoss(target, out):

'''print(out.shape)
out = torch.sigmoid(out)

e = 1e-6
out = out + e

alpha = 0.5
gamma = 2.0



BCE_loss = F.binary_cross_entropy(out, target, reduce=False)
print("BCE_loss: ", BCE_loss)
pt = torch.exp(-BCE_loss)
print("pt: ", pt)
F_loss = alpha * (1 - pt) ** gamma * BCE_loss
print("F loss: ", F_loss)
print(F_loss.shape)
losses = F_loss.mean(dim=1)
print("LOSSES", losses)

return losses.squeeze()'''
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):

        input = torch.sigmoid(input)
        logpt = F.binary_cross_entropy(input, target, reduce=False)


        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        #target = (target.view(-1,1))
        if target.dim()>2:
            target = targe.view(target.size(0),target.size(1),-1)  # N,C,H,W => N,C,H*W
            target = target.transpose(1,2)    # N,C,H*W => N,H*W,C
            target = target.contiguous().view(-1,target.size(2))   # N,H*W,C => N*H*W,C

        target = target.type(torch.LongTensor).cuda()
        #logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = Variable(loss, requires_grad = True)
        if self.size_average: return loss.mean()
        else: return loss.sum()







def match(masks, overlaps):
    """
    Args:
        masks - list containing [true_masks, pred_masks], both being [batch_size,T,N]
        overlaps - [batch_size,T,T] - matrix of costs between all pairs
    Returns:
        t_mask_cpu - [batch_size,T,N] permuted ground truth masks
        permute_indices - permutation indices used to sort the above
    """

    overlaps = (overlaps.data).cpu().numpy().tolist()
    m = Munkres()

    t_mask, p_mask = masks

    # get true mask values to cpu as well
    t_mask_cpu = (t_mask.data).cpu().numpy()
    # init matrix of permutations
    permute_indices = np.zeros((t_mask.size(0),t_mask.size(1)),dtype=int)
    # we will loop over all samples in batch (must apply munkres independently)
    for sample in range(p_mask.size(0)):
        # get the indexes of minimum cost
        indexes = m.compute(overlaps[sample])
        for row, column in indexes:
            # put them in the permutation matrix
            permute_indices[sample,column] = row

        # sort ground according to permutation
        t_mask_cpu[sample] = t_mask_cpu[sample,permute_indices[sample],:]
        
    return t_mask_cpu, permute_indices
    
def reorder_mask(y_mask, permutation):

     t_mask_cpu = (y_mask.data).cpu().numpy()
     size = y_mask.size(0)
     for sample in range(size):
         t_mask_cpu[sample] = t_mask_cpu[sample,permutation[sample],:]
         
     return t_mask_cpu

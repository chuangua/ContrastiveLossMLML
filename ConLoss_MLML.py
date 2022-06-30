import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy as sp
import scipy.linalg as linalg
from torch.autograd import Variable, Function

class CLLoss(Function):
    @staticmethod
    def forward(ctx, X, y,lambda_):
        yy = y.cpu().detach().numpy().copy()

        classes = y.shape[1]
        N, D = X.shape
        #lambda_ = 1.
        DELTA = 1.

        # gradients initialization
        Obj_c = 0
        Obj_all = 0

        dX_c = np.zeros((N, D))
        dX_all = np.zeros((N,D))

        eigThd = 1e-6

        # compute objective and gradient for first term \sum ||TX_c||*
        for c in range(classes):
            A = X[y[:,c]==1,:]
            if A.shape!=torch.Size([0]):
                # SVD
                U, S, V = torch.linalg.svd(A, full_matrices=False)
                U=U.cpu().detach().numpy().copy()
                S=S.cpu().detach().numpy().copy()
                V=V.cpu().detach().numpy().copy()
                
                nuclear = np.sum(S)
                ## L_c = max(DELTA, ||TY_c||_*)-DELTA
                V = np.transpose(V)

                if nuclear > DELTA:
                    Obj_c += nuclear
                    # discard small singular values
                    r = np.sum(S < eigThd)
                    uprod=np.dot(U[:, 0:U.shape[1] - r],np.transpose(V[:, 0:V.shape[1] - r]))
                    dX_c[yy[:,c]==1,:] += uprod
                else:
                    Obj_c += DELTA

        # compute objective and gradient for secon term ||TX||*
        U, S, V = torch.linalg.svd(X, full_matrices = False)  # all classes
        U=U.cpu().detach().numpy().copy()
        S=S.cpu().detach().numpy().copy()
        V=V.cpu().detach().numpy().copy()
        Obj_all = np.sum(S)

        V = np.transpose(V)
        r = np.sum(S < eigThd)
        uprod=np.dot(U[:, 0:U.shape[1] - r],np.transpose(V[:, 0:V.shape[1] - r]))
        dX_all = uprod
        dX = (dX_c - lambda_ * dX_all) / N * np.float(lambda_)
        ctx.dX = torch.FloatTensor(dX).cuda()

        obj = (Obj_c  - lambda_*Obj_all)/N*np.float(lambda_)
        obj=torch.FloatTensor([float(obj)])[0].cuda()
        return obj

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.dX,None,None

class CLML(nn.Module):
    r""" CLML loss as described in the paper "Label Structure Preserving Contrastive Embedding for Multi-Label Learning with Missing Labels "
    .. note::
     CLML can be combinded with various multi-label loss functions. 
        """
    def __init__(self,
                 tau: float = 0.7,
                 change_epoch: int = 1,
                 margin: float = 1.0) -> None:
        super(CLML, self).__init__()
        self.tau = tau
        self.change_epoch = change_epoch
        self.margin = margin

    def forward(self, logits: torch.Tensor, targets: torch.LongTensor,feature,epoch,lam) -> torch.Tensor:

        # Subtract margin for positive logits
        logits = torch.where(targets == 1, logits-self.margin, logits)
        
        # CLML missing label correction
        if epoch >= self.change_epoch:
            targets = torch.where(
                torch.sigmoid(logits) > self.tau,
                torch.tensor(1).cuda(), targets)
        
        loss_cl=CLLoss.apply(feature,targets,lam)
        return loss_cl

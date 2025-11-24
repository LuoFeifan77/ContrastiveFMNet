import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.registry import LOSS_REGISTRY
from losses.fmap_loss import SquaredFrobeniusLoss


@LOSS_REGISTRY.register()
class FeatContrastiveLoss(nn.Module):
    def __init__(self, w_inter=1.0, w_intra=1.0, tau_inter=0.07, tau_intra=10.0,  inter_top_k=100, intra_top_k=100, bidirectional=True):
        super(FeatContrastiveLoss, self).__init__()
        self.w_inter = w_inter  # inter contrastive
        self.w_intra = w_intra  # intra contrastive
        self.tau_inter = tau_inter
        self.tau_intra = tau_intra
        self.inter_top_k_min = inter_top_k  # 下界
        self.intra_top_k_min = intra_top_k
        self.bidirectional = bidirectional

    def forward(self, feat_x, feat_y, curr_epoch=0):

        losses = dict()
        feat_x = F.normalize(feat_x, p=2, dim=-1)
        feat_y = F.normalize(feat_y, p=2, dim=-1)

        self.inter_top_k = self.inter_top_k_min
        self.intra_top_k = self.intra_top_k_min

        # logger.info(f'top_k: {top_k}')

        if self.w_inter>0:        
            similarity_xy = torch.bmm(feat_x, feat_y.transpose(1, 2)) #
            vals, idx = similarity_xy.topk(k=self.inter_top_k, dim=2)  #
            similarity_xy_p = torch.full_like(similarity_xy, 0)  
            similarity_xy_p.scatter_(-1, idx, vals) # positive similarity
            similarity_xy_n = similarity_xy - similarity_xy_p  # negetive similarity

            similarity_xy_p /= self.tau_inter 
            similarity_xy_n /= self.tau_inter 

            Sxy_p = torch.sum(similarity_xy_p, dim=-1)/self.inter_top_k
            inter_loss = -torch.sum(Sxy_p- (torch.logsumexp(similarity_xy_n, dim=-1, keepdim=False)))/feat_x.shape[1] 

        
            if self.bidirectional: 
                similarity_yx = similarity_xy.transpose(1, 2)  
                vals, idx = similarity_yx.topk(k=self.inter_top_k, dim=2)  # 
                similarity_yx_p = torch.full_like(similarity_yx, 0)  
                similarity_yx_p.scatter_(-1, idx, vals) # positive similarity
                similarity_yx_n = similarity_yx - similarity_yx_p  # negetive similarity

                similarity_yx_p /= self.tau_inter 
                similarity_yx_n /= self.tau_inter 

                Syx_p = torch.sum(similarity_yx_p, dim=-1)/self.inter_top_k
                inter_loss -= torch.sum(Syx_p- (torch.logsumexp(similarity_yx_n, dim=-1, keepdim=False)))/feat_y.shape[1]
                
            losses['l_inter'] = self.w_inter*inter_loss


        if self.w_intra>0:

            similarity_yy = torch.bmm(feat_y, feat_y.transpose(1, 2)) # consine similarity
            similarity_yy_p = torch.diag(similarity_yy[0].diagonal()).unsqueeze(0) # 
            
            vals, idx = similarity_yy.topk(k=self.intra_top_k, dim=2,) 

            temp = torch.full_like(similarity_yy, 0)  
            temp.scatter_(-1, idx, vals) # positive similarity

            similarity_yy_n = similarity_yy - temp 

            similarity_yy_n /= self.tau_intra

            intra_loss = -torch.sum(- (torch.logsumexp(similarity_yy_n, dim=-1, keepdim=False))) /feat_y.shape[1] 
            
            if self.bidirectional:        
                similarity_xx = torch.bmm(feat_x, feat_x.transpose(1, 2))  # consine similarity
                similarity_xx_p = torch.diag(similarity_xx[0].diagonal()).unsqueeze(0) 
                
                vals, idx = similarity_xx.topk(k=self.intra_top_k, dim=2,) 

                temp = torch.full_like(similarity_xx, 0)  
                temp.scatter_(-1, idx, vals) # positive similarity
                similarity_xx_n = similarity_xx - temp  

                similarity_xx_n /= self.tau_intra

                intra_loss += -torch.sum(- (torch.logsumexp(similarity_xx_n, dim=-1, keepdim=False))) /feat_x.shape[1] 
            
            losses['l_intra'] = self.w_intra*intra_loss

        return losses





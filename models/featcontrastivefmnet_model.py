import torch
import torch.nn.functional as F

from .base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.fmap_util import nn_query, fmap2pointmap

@MODEL_REGISTRY.register()
class FeatContrastiveFMNetModel(BaseModel):
    def __init__(self, opt):
        self.with_refine = opt.get('refine', -1)
        self.partial = opt.get('partial', False)
        self.non_isometric = opt.get('non-isometric', False)
        if self.with_refine > 0:
            opt['is_train'] = True
        super(FeatContrastiveFMNetModel, self).__init__(opt)

    def feed_data(self, data):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        # feature extractor for mesh
        feat_x = self.networks['feature_extractor'](data_x['verts'], data_x['faces'])  # [B, Nx, C]
        feat_y = self.networks['feature_extractor'](data_y['verts'], data_y['faces'])  # [B, Ny, C]

        # get spectral operators
        # evals_x = data_x['evals']
        # evals_y = data_y['evals']
        evecs_x = data_x['evecs']
        evecs_y = data_y['evecs']
        evecs_trans_x = data_x['evecs_trans']  # [B, K, Nx]
        evecs_trans_y = data_y['evecs_trans']  # [B, K, Ny]

        #1 point-to-point Pyx, by softmax 
        Pyx = self.compute_permutation_matrix(feat_x, feat_y, bidirectional=False)

        #2 compute functional maps  
        Cxy_est = torch.bmm(evecs_trans_y, torch.bmm(Pyx, evecs_x))

        rfmnet_loss = torch.linalg.norm(evecs_y-torch.bmm(Pyx, torch.bmm(evecs_x, Cxy_est.transpose(-2, -1))))
        self.loss_metrics = self.losses['rfmnet_loss'](rfmnet_loss)  # 加上权重！
        
        #3 contrastive loss on features
        if 'contrast_loss' in self.losses:
            contrastive_loss = self.losses['contrast_loss'](feat_x, feat_y)
            self.loss_metrics.update(contrastive_loss)  # 合并loss


    def validate_single(self, data, timer):
        # get data pair
        data_x, data_y = to_device(data['first'], self.device), to_device(data['second'], self.device)

        # get previous network state dict
        if self.with_refine > 0:
            state_dict = {'networks': self._get_networks_state_dict()}

        # start record
        timer.start()

        # test-time refinement
        if self.with_refine > 0:
            self.refine(data)

        # feature extractor
        feat_x = self.networks['feature_extractor'](data_x['verts'], data_x.get('faces'))
        feat_y = self.networks['feature_extractor'](data_y['verts'], data_y.get('faces'))

        # get spectral operators
        evecs_x = data_x['evecs'].squeeze()
        evecs_y = data_y['evecs'].squeeze()
        evecs_trans_x = data_x['evecs_trans'].squeeze()
        evecs_trans_y = data_y['evecs_trans'].squeeze()

        feat_x = F.normalize(feat_x, dim=-1, p=2)
        feat_y = F.normalize(feat_y, dim=-1, p=2)

        # nearest neighbour query
        p2p = nn_query(feat_x, feat_y).squeeze()

        if self.non_isometric:
            # compute Pyx from functional map
            Cxy = evecs_trans_y @ evecs_x[p2p]
            Pyx = evecs_y @ Cxy @ evecs_trans_x
        else:
            iter_num = 5  # for near-isometric matching
            for _ in range(iter_num):
                Cxy = evecs_trans_y @ evecs_x[p2p]
                p2p = fmap2pointmap(Cxy, evecs_x, evecs_y)

            # compute Pyx from functional map
            Pyx = evecs_y @ Cxy @ evecs_trans_x

        # finish record
        timer.record()

        # resume previous network state dict
        if self.with_refine > 0:
            self.resume_model(state_dict, net_only=True, verbose=False)
        return p2p, Pyx, Cxy

    def compute_permutation_matrix(self, feat_x, feat_y, bidirectional=False, normalize=True):
        if normalize:
            feat_x = F.normalize(feat_x, dim=-1, p=2)
            feat_y = F.normalize(feat_y, dim=-1, p=2)
        similarity = torch.bmm(feat_y, feat_x.transpose(1, 2))

        # sinkhorn normalization
        Pyx = self.networks['permutation'](similarity)

        if bidirectional:
            Pxy = self.networks['permutation'](similarity.transpose(1, 2))
            return Pyx, Pxy
        else:
            return Pyx
                
    def refine(self, data):
        self.networks['permutation'].hard = False
        self.networks['fmap_net'].bidirectional = True

        with torch.set_grad_enabled(True):
            for _ in range(self.with_refine):
                self.feed_data(data)
                self.optimize_parameters()

        self.networks['permutation'].hard = True
        self.networks['fmap_net'].bidirectional = False

    @torch.no_grad()
    def validation(self, dataloader, tb_logger, update=True):
        # change permutation prediction status
        if 'permutation' in self.networks:
            self.networks['permutation'].hard = True
        if 'fmap_net' in self.networks:
            self.networks['fmap_net'].bidirectional = False
        super(FeatContrastiveFMNetModel, self).validation(dataloader, tb_logger, update)
        if 'permutation' in self.networks:
            self.networks['permutation'].hard = False
        if 'fmap_net' in self.networks:
            self.networks['fmap_net'].bidirectional = True

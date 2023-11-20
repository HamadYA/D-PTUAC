import torch.nn as nn
from timm.models.layers import trunc_normal_
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.task_head.taskhead_base import TRACK_HEADS




class MlpHead(nn.Module):
    def __init__(self, dim, W, H):
        super(MlpHead, self).__init__()
        # self.norm= nn.LayerNorm(dim)
        self.cls_mlp = Mlp(dim, out_features=1, num_layers=3)
        self.reg_mlp = Mlp(dim, out_features=4, num_layers=3)
        self.W = W
        self.H = H        
        self.reset_parameters()
        
    def reset_parameters(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

    def forward(self, x):
        '''
            Args:
                x (torch.Tensor): (B, H * W, C) input feature map
            Returns:
                Dict: {
                    'cls_score' (torch.Tensor): (B, 1, H, W)
                    'bbox' (torch.Tensor): (B, H, W, 4)
                }
        '''
        # x = self.norm(x)
        cls = self.cls_mlp(x)
        reg = self.reg_mlp(x).sigmoid()

        class_score = cls
        B, L, C = class_score.shape
        class_score = class_score.view(B, self.H, self.W, C)
        class_score = class_score.permute(0, 3, 1, 2)
        class_score = class_score.sigmoid()
        bbox = reg.view(B, self.H, self.W, 4)
        return {'class_score': class_score, 'bbox': bbox}



class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 num_layers=2,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.layers = nn.ModuleList(
            [nn.Linear(hidden_features if i != 0 else in_features,
                       hidden_features if i != num_layers - 1 else out_features
                       ) for i in range(num_layers)]
        )

        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        for i, linear in enumerate(self.layers):
            x = linear(x)
            if i != len(self.layers) - 1:
                x = self.act(x)
            x = self.drop(x)
        return x



@TRACK_HEADS.register
class Head(ModuleBase):
    default_hyper_params = dict(
        score_size = 14,
        in_channels = 384
    )

    def __init__(self, ):
        super(Head, self).__init__()
      
    def forward(self, y):
        return self.solve(y)

    def update_params(self):
        super().update_params()
        self.H = self._hyper_params["score_size"]
        self.W = self._hyper_params["score_size"]
        self.dim = self._hyper_params["in_channels"]
        self.solve = MlpHead(self.dim,self.W,self.H)





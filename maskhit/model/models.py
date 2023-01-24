# adapting code from https://github.com/zhoudaquan/dvit_repo

import torch
import torch.nn as nn
from einops import rearrange
from .helper import create_model, parse_archs
from .archs import AggAvgPool, AggAttn, AggMHAttn, AggViT, AggDeepAttnMISL


########################################
# the overall model
########################################
class ResNetExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        self.resnet = create_model(18, True, 1)
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        b = x.size(0)
        is_valid = x.std((2, 3, 4)) != 0
        res = torch.zeros(is_valid.size(0) * is_valid.size(1),
                          512).to(x.device)
        is_valid = rearrange(is_valid, 'b p -> (b p)')
        x = rearrange(x, 'b p d h w -> (b p) d h w')
        res[is_valid] = self.resnet(x[is_valid])
        res = rearrange(res, '(b t p) d -> b t p d', b=b, t=1)
        return res


class HybridModel(nn.Module):

    def __init__(
        self,
        in_dim=512,
        out_dim=1,
        dropout=0.1,
        args=None,
        model_name=None,
        outcome_type='survival',
    ):
        super().__init__()
        self.args = args
        self.model_name = model_name
        self.outcome_type = outcome_type
        if self.args.use_features:
            self.backbone = nn.Identity()
        else:
            self.backbone = ResNetExtractor()

        self.proj = nn.Sequential(nn.Identity())

        arch_specs_attn = parse_archs(args.mil1)
        arch_specs_fuse = parse_archs(args.mil2)
        self.attn = dict_agg[arch_specs_attn['name']](
            in_dim=in_dim,
            h=arch_specs_attn['h'],
            layers=arch_specs_attn['l'],
            dropout=dropout,
            args=args)
        self.fuse = dict_agg[arch_specs_fuse['name']](in_dim=self.attn.dim,
                                                      h=arch_specs_fuse['h'],
                                                      dropout=dropout,
                                                      args=args)

        self.pred = FinalPred(in_dim=self.attn.dim,
                              out_dim=out_dim,
                              outcome_type=outcome_type)

    def forward(self, inputs):
        x = inputs['x']
        pos = inputs['pos']
        ids = inputs['ids']
        n_tiles = inputs['n_tiles']

        x = self.backbone(x)

        # tile-wise aggregation
        x = rearrange(x, 'b t p d -> (b t) p d')
        pos = rearrange(pos, 'b t p d -> (b t) p d')
        stage1_inputs = {
            'x': x,
            'pos': pos
        }

        stage1_outputs = self.attn(stage1_inputs,
                                   avg_pool=self.args.avg_cls,
                                   prob_mask=self.args.prob_mask)

        # enc_cls, enc_seq, org_seq, info

        # wsi-wise aggregation
        ids = ids.unsqueeze(1).expand(-1, n_tiles).flatten()
        stage2_inputs = {
            'x':
            rearrange(stage1_outputs['enc_cls'],
                      '(b n) d -> b n d',
                      n=n_tiles),
            'pos':
            inputs['pos_tile']
        }

        stage2_ouputs = self.fuse(stage2_inputs, avg_pool=False)

        # make the prediction
        out = self.pred(stage2_ouputs['enc_cls'])

        outputs = {
            'out': out,
            'enc_cls': stage1_outputs['enc_cls'],
            'enc_seq': stage1_outputs['enc_seq'],
            'org_seq': stage1_outputs['org_seq'],
            'attn': stage1_outputs.get('attn', None),
            'dots': stage1_outputs.get('dots', None),
            'zeros': stage1_outputs.get('zeros', None),
            'masks': stage1_outputs.get('masks', None),
        }
        return outputs


########################################
# MLP models
########################################

##############################
# final prediction
##############################


class FinalPred(nn.Module):

    def __init__(self, in_dim, out_dim, outcome_type):
        super().__init__()

        if outcome_type == 'survival':
            self.fc = nn.Linear(in_dim, 1, bias=False)
        else:
            self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.fc(x)
        return x


dict_agg = {
    'ap': AggAvgPool,
    'attn': AggAttn,
    'mhattn': AggMHAttn,
    'deepattnmisl': AggDeepAttnMISL,
    'vit': AggViT
}

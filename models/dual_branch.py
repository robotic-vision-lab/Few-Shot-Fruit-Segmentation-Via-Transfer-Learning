import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm2d
from models.model_utils import BasicBlock, Bottleneck, PAPPM, segmenthead, SegmentHead, Light_Bag
import timm

class DualBranch(nn.Module):

    def __init__(self, n_classes, aux_mode='train', in_chans=3, embed_dims=[64, 128, 256, 512],
                 bnums=[2, 2, 2, 2], strides=[1, 2, 2, 2], *args, **kwargs):
        super(DualBranch, self).__init__()
        self.resnet = timm.create_model('resnet18',pretrained=True,features_only=True)
        self.relu = nn.ReLU(inplace=True)
        self.layer5 =  self._make_layer(Bottleneck, embed_dims[3], embed_dims[3], 2, stride=2)
        self.layer5_ = nn.Sequential(
                          nn.Conv2d(embed_dims[2],embed_dims[2],kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(embed_dims[2], momentum=0.1),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(embed_dims[2],embed_dims[2],kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(embed_dims[2], momentum=0.1),
                          nn.ReLU(inplace=True),
                      )
        self.layer3_d = self._make_single_layer(BasicBlock, embed_dims[1], embed_dims[0])
        self.layer4_d = self._make_layer(Bottleneck, embed_dims[0], embed_dims[0], 1)
        self.layer5_d = self._make_layer(Bottleneck, embed_dims[1], embed_dims[1], 1)
        self.diff3 = nn.Sequential(
                                    nn.Conv2d(embed_dims[2], embed_dims[0], kernel_size=3, padding=1, bias=False),
                                    BatchNorm2d(embed_dims[0], momentum=0.1),
                                    )
        self.diff4 = nn.Sequential(
                                 nn.Conv2d(embed_dims[3], embed_dims[1], kernel_size=3, padding=1, bias=False),
                                 BatchNorm2d(embed_dims[1], momentum=0.1),
                                 )
        self.spp = PAPPM(embed_dims[3] * 2, 96, embed_dims[2])
        self.dfm = Light_Bag(embed_dims[2], embed_dims[2])
        self.conv_out = SegmentHead(embed_dims[2], embed_dims[2], n_classes, up_factor=8)
        self.aux_mode = aux_mode
        if self.aux_mode == 'train':
            self.conv_out_ = SegmentHead(embed_dims[3], embed_dims[2], n_classes, up_factor=8)
            self.conv_out16 = SegmentHead(embed_dims[1], embed_dims[1], n_classes, up_factor=8)
            self.seghead_d = segmenthead(embed_dims[1], embed_dims[0], 1)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        width_output = W // 8
        height_output = H // 8
        a,feat4, feat8, feat16, feat32 = self.resnet(x)
        
        feat_ = F.interpolate(
                        self.layer5_(feat16),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=False)
        feat = F.interpolate(
                        self.spp(self.layer5(feat32)),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=False)
                        
            
        
        feat_out = self.conv_out(feat_ + feat)
        if self.aux_mode == 'train':
            feat_out16 = self.conv_out16(feat8)
            feat_out_ = self.conv_out_(feat32)
            return feat_out, feat_out16, feat_out_
        elif self.aux_mode == 'eval':
            return feat_out,
        elif self.aux_mode == 'pred':
            feat_out = feat_out.argmax(dim=1)
            return feat_out
        else:
            raise NotImplementedError

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FeatureFusionModule, BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params
    
    def _make_single_layer(self, block, inplanes, planes, stride=1):
            downsample = None
            if stride != 1 or inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion, momentum=0.1),
                )

            layer = block(inplanes, planes, stride, downsample, no_relu=True)
            
            return layer
    
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.1),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)
  
"""
Interface of DNN models for image translations
"""
import functools
from abc import ABC

import math
import numpy as np
from queue import Queue
from torch import nn
from torch.autograd import Variable
import torch
import collections
from torch.autograd import grad as ta_grad
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from torchvision.models import vgg11, vgg19, vgg11_bn, vgg19_bn, resnet50, resnet101
from torch.nn.modules.batchnorm import _BatchNorm
import torch.utils.model_zoo as model_zoo
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast
from utils import SyncMaster
import functools
from torch.nn.parallel.data_parallel import DataParallel

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass


##################################################################################
# Interfaces
##################################################################################

def get_discriminator(dis_opt, train_mode=None):
    """Get a discriminator"""
    # multi-scale dis
    return MsImageDis(dis_opt)


def get_generator(gen_opt, train_mode=None):
    """Get a generator"""
    return DirectGenMS(gen_opt)


##################################################################################
# Discriminator
##################################################################################

class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, dis_opt):
        super(MsImageDis, self).__init__()
        self.dim = dis_opt.dis.dim
        self.norm = dis_opt.dis.norm
        self.activ = dis_opt.dis.activ
        self.pad_type = dis_opt.dis.pad_type
        self.gan_type = dis_opt.dis.gan_type
        self.n_layers = dis_opt.dis.n_layer
        self.use_grad = dis_opt.dis.use_grad
        self.input_dim = dis_opt.input_dim
        self.num_scales = dis_opt.dis.num_scales
        self.use_wasserstein = dis_opt.dis.use_wasserstein
        self.grad_w = dis_opt.grad_w
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.models = nn.ModuleList()
        self.sigmoid_func = nn.Sigmoid()

        for _ in range(self.num_scales):
            cnns = self._make_net()
            if self.use_wasserstein:
                cnns += [nn.Sigmoid()]

            self.models.append(nn.Sequential(*cnns))

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layers - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        return cnn_x

    def forward(self, x):
        output = None
        for model in self.models:
            out = model(x)
            if output is not None:
                _, _, h, w = out.shape
                output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)
                output = output + out
            else:
                output = out

            x = self.downsample(x)

        output = output / len(self.models)
        output = self.sigmoid_func(output)

        return output

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)

            # Gradient penalty
            grad_loss = 0
            if self.use_grad:
                eps = Variable(torch.rand(1), requires_grad=True)
                eps = eps.expand(input_real.size())
                eps = eps.cuda()
                x_tilde = eps * input_real + (1 - eps) * input_fake
                x_tilde = x_tilde.cuda()
                pred_tilde = self.calc_gen_loss(x_tilde)
                gradients = ta_grad(outputs=pred_tilde, inputs=x_tilde,
                                    grad_outputs=torch.ones(pred_tilde.size()).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
                grad_loss = self.grad_w * gradients

                input_real = self.downsample(input_real)
                input_fake = self.downsample(input_fake)

            loss += ((grad_loss.norm(2, dim=1) - 1) ** 2).mean()

        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1) ** 2)  # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).to(self.device), requires_grad=True)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


##################################################################################
# Generator
##################################################################################

class DirectGenMS(nn.Module):
    def __init__(self, param):
        super(DirectGenMS, self).__init__()
        self.dim = param.gen.dim
        self.norm = param.gen.norm
        self.activ = param.gen.activ
        self.pad_type = param.gen.pad_type
        self.n_layers = param.gen.n_layer
        self.input_dim = param.input_dim
        self.pretrained = param.gen.pretrained
        self.output_dim = param.output_dim_b + param.output_dim_r
        self.decoder_mode = param.gen.decoder_mode

        encoder_name = param.gen.encoder_name

        # Feature extractor as Encoder
        if encoder_name == 'vgg11':
            self.encoder = Vgg11EncoderMS(pretrained=self.pretrained)
        elif encoder_name == 'vgg19':
            self.encoder = Vgg19EncoderMS(pretrained=self.pretrained)
        elif encoder_name == 'resnet50':
            self.encoder = ResNet50AtrousEncoderMS(pretrained=self.pretrained, param=param)
        elif encoder_name == 'resnet101':
            self.encoder = ResNet101AtrousEncoderMS(pretrained=self.pretrained, param=param)
        else:
            raise ValueError('encoder name should in [vgg11/vgg19], but it is: {}'.format(encoder_name))

        self.decoder_sep = DecoderSeparate(param)
        self.decoder_sem = DecoderSemantic(param)

    def _decode(self, feats=None):
        if 'Semantic' in self.decoder_mode or 'Guidance' in self.decoder_mode:
            sem, sem_feats = self.decoder_sem(feats)
        else:
            sem, sem_feats = None, None
        if self.decoder_mode == 'OnlySemantic':
            return None, sem
        return self.decoder_sep(feats, sem_feats), sem

    def _encode(self, x):
        return self.encoder(x)

    def get_encoder_features(self, x):
        with torch.no_grad():
            fea = self.encoder(x)
        return fea

    def forward(self, x):
        feats = self._encode(x)
        out, sem = self._decode(feats)
        if self.output_dim == 6:
            if self.decoder_mode == 'OnlySemantic':
                return None, None, sem, None
            out_a = out[:, :3, :, :]
            out_b = out[:, 3:, :, :]
            return out_a, out_b, sem, feats
        return out, sem, feats


def get_classifier(name='vgg19', num_classes=2, pretrained=False):
    if name == 'vgg19':
        return vgg19(num_classes=num_classes, pretrained=pretrained)
    if name == 'vgg11':
        return vgg11(num_classes=num_classes, pretrained=pretrained)
    if name == 'vgg19_bn':
        return vgg19_bn(num_classes=num_classes, pretrained=pretrained)
    if name == 'vgg11_bn':
        return vgg11_bn(num_classes=num_classes, pretrained=pretrained)
    if name == 'resnet50':
        return resnet50(num_classes=num_classes, pretrained=pretrained)
    if name == 'resnet101':
        return resnet101(num_classes=num_classes, pretrained=pretrained)


##################################################################################
# Encoder and Decoders
##################################################################################

class Vgg11EncoderMS(nn.Module):
    """Vgg encoder wiht multi-scales"""

    def __init__(self, pretrained):
        super(Vgg11EncoderMS, self).__init__()
        features = list(vgg11(pretrained=pretrained).features)
        self.backbone = nn.ModuleList(features)

    def forward(self, x):
        result_dict = {}
        layer_names = ['conv1_1',
                       'conv2_1',
                       'conv3_1', 'conv3_2',
                       'conv4_1', 'conv4_2',
                       'conv5_1', 'conv5_2']
        idx = 0
        for ii, model in enumerate(self.backbone):
            x = model(x)
            if ii in {0, 3, 6, 8, 11, 13, 16, 18}:
                result_dict[layer_names[idx]] = x
                idx += 1

        out_feature = {
            'input': x,
            'shallow': result_dict['conv1_1'],
            'low': result_dict['conv2_1'],
            'mid': result_dict['conv3_2'],
            'deep': result_dict['conv3_2'],
            'out': result_dict['conv5_2'],
            'name': 'vgg11'
        }
        return out_feature


class Vgg19EncoderMS(nn.Module):
    def __init__(self, pretrained):
        super(Vgg19EncoderMS, self).__init__()
        features = list(vgg19(pretrained=pretrained).features)
        self.backbone = nn.ModuleList(features)

    def forward(self, x):
        result_dict = {}
        layer_names = ['conv1_1', 'conv1_2',
                       'conv2_1', 'conv2_2',
                       'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                       'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
                       'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']
        idx = 0
        for ii, model in enumerate(self.backbone):
            x = model(x)
            if ii in {0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34}:
                result_dict[layer_names[idx]] = x
                idx += 1

        out_feature = {
            'input': x,
            'shallow': result_dict['conv1_2'],
            'low': result_dict['conv2_2'],
            'mid': result_dict['conv3_2'],
            'deep': result_dict['conv4_2'],
            'out': result_dict['conv5_2'],
            'name': 'vgg19'
        }
        return out_feature


class ResNet50AtrousEncoderMS(nn.Module):
    def __init__(self, pretrained, param, os=16):
        super(ResNet50AtrousEncoderMS, self).__init__()
        self.backbone = ResNetAtrous(Bottleneck, [3, 4, 6, 3], atrous=[1, 2, 1], os=os)
        if pretrained:
            old_dict = model_zoo.load_url(model_urls['resnet50'])
            model_dict = self.backbone.state_dict()
            old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
            model_dict.update(old_dict)
            self.backbone.load_state_dict(model_dict)
        self.aspp = ASPP(dim_in=param.gen.aspp.input_channel,
                         dim_out=param.gen.aspp.output_dim,
                         rate=16 // param.gen.aspp.output_stride,
                         bn_mom=param.batch_norm_mom)
        self.dropout = nn.Dropout(0.5)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=param.gen.aspp.output_stride // 4)

    def forward(self, x):
        out = self.backbone(x)
        layers = self.backbone.get_layers()
        out = self.aspp(out)
        # out = self.dropout(out)
        out = self.upsample(out)

        out_feature = {'input': x,
                       'shallow': layers[0],
                       'low': layers[1],
                       'mid': layers[2],
                       'deep': layers[3],
                       'out': out,
                       'name': 'resnet50'
                       }

        return out_feature


class ResNet101AtrousEncoderMS(nn.Module):
    def __init__(self, pretrained, param, os=16):
        super(ResNet101AtrousEncoderMS, self).__init__()
        self.backbone = ResNetAtrous(Bottleneck, [3, 4, 23, 3], atrous=[2, 2, 2], os=os)
        if pretrained:
            old_dict = model_zoo.load_url(model_urls['resnet101'])
            model_dict = self.backbone.state_dict()
            old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
            model_dict.update(old_dict)
            self.backbone.load_state_dict(model_dict)
        self.aspp = ASPP(dim_in=param.gen.aspp.input_channel,
                         dim_out=param.gen.aspp.output_dim,
                         rate=16 // param.gen.aspp.output_stride,
                         bn_mom=param.batch_norm_mom)
        # self.dropout = nn.Dropout(0.5)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=param.gen.aspp.output_stride // 4)

    def forward(self, x):
        out = self.backbone(x)
        layers = self.backbone.get_layers()
        out = self.aspp(out)
        # out = self.dropout(out)
        out = self.upsample(out)

        out_feature = {'input': x,
                       'shallow': layers[0],
                       'low': layers[1],
                       'mid': layers[2],
                       'deep': layers[3],
                       'out': out,
                       'name': 'resnet101'
                       }

        return out_feature


class DecoderSeparate(nn.Module):
    def __init__(self, param, res_scale=0.1, se_reduction=8):
        """output_shape = [H, W, C]"""
        super(DecoderSeparate, self).__init__()

        input_dim = param.input_dim
        dim = param.gen.dim
        n_resblocks = param.gen.n_layer
        pad_type = param.gen.pad_type
        norm = param.gen.norm
        activ = param.gen.activ
        act = nn.ReLU(True)
        output_dim = param.output_dim_b + param.output_dim_r
        decoder_mode = param.gen.decoder_mode
        encoder_name = param.gen.encoder_name
        num_classes = param.gen.n_classes
        res_scale = param.gen.res_scale
        se_reduction = param.gen.se_reduction
        norm_layer = nn.BatchNorm2d

        num_classes = num_classes if num_classes is not None else 0
        self.mode = decoder_mode

        self.fusion_block = FusionBlock(input_dim=input_dim, dim=dim, output_dim=dim, pad_type=pad_type,
                                        activ=activ, norm=norm, encoder_name=encoder_name)
        self.channel_attention_block = nn.Sequential(*[ChannelAttentionBlock(
            channels=dim, dilation=1, norm=norm_layer, act=act,
            se_reduction=se_reduction, res_scale=res_scale) for i in range(n_resblocks)])
        self.pyramid_block = PyramidPoolingBlock(in_channels=dim, out_channels=dim, scales=(4, 8, 16, 32),
                                                 ct_channels=dim // 4)
        self.guidance_block = GuidanceBlock(channels=dim, norm=norm_layer, act=act)
        self.output_layer = nn.Sequential(
            ConvLayer(nn.Conv2d, dim, dim, kernel_size=3, stride=1, norm=norm_layer, act=act),
            ConvLayer(nn.Conv2d, dim, output_dim, kernel_size=1, stride=1, norm=None, act=None)
        )

    def forward(self, feat_dict, semantic_map=None):
        x = self.fusion_block(feat_dict)
        if 'Channel' in self.mode:
            x = self.channel_attention_block(x)
        if 'Pyramid' in self.mode:
            x = self.pyramid_block(x)
        if 'Guidance' in self.mode:
            assert semantic_map is not None
            x = self.guidance_block(x, semantic_map['feature'])
            x = self.guidance_block(x, semantic_map['out'])

        x = self.output_layer(x)
        return x


class DecoderSemantic(nn.Module):
    def __init__(self, param):
        super(DecoderSemantic, self).__init__()

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=4)
        indim = 256
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(indim, param.gen.shortcut.dim, param.gen.shortcut.kernel, 1,
                      padding=param.gen.shortcut.kernel // 2, bias=True),
            SynchronizedBatchNorm2d(param.gen.shortcut.dim, momentum=param.batch_norm_mom),
            nn.ReLU(inplace=True),
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(param.gen.aspp.output_dim + param.gen.shortcut.dim, param.gen.aspp.output_dim, 3, 1, padding=1,
                      bias=True),
            SynchronizedBatchNorm2d(param.gen.aspp.output_dim, momentum=param.batch_norm_mom),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(param.gen.aspp.output_dim, param.gen.aspp.output_dim, 3, 1, padding=1, bias=True),
            SynchronizedBatchNorm2d(param.gen.aspp.output_dim, momentum=param.batch_norm_mom),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(param.gen.aspp.output_dim, param.gen.n_classes, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature_dict):
        feature = {}
        feature_shallow = self.shortcut_conv(feature_dict['low'])
        h, w = feature_dict['out'].shape[2:]
        feature_shallow = F.interpolate(feature_shallow, size=(h, w), mode='bilinear', align_corners=False)
        feature_cat = torch.cat([feature_dict['out'], feature_shallow], 1)
        result = self.cat_conv(feature_cat)
        feature['feature'] = result
        result = self.cls_conv(result)

        h, w = feature_dict['input'].shape[2:]
        result = F.interpolate(result, size=(h, w), mode='nearest', align_corners=None)
        feature['out'] = result
        return result, feature


##################################################################################
# Basic Blocks
##################################################################################

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride=1,
                 padding=0, norm='none', activation='relu', pad_type='zero', dilation=1):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                              bias=self.use_bias, dilation=dilation)

    def forward(self, x):
        x = self.conv(self.pad(x))
        # if self.norm:
        #     x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


# define a ResNet(dilated) block
class ResDilateBlock(nn.Module):
    def __init__(self, input_dim, dim, output_dim, rate,
                 padding=0, norm='none', activation='relu', pad_type='zero', use_bias=False):
        super(ResDilateBlock, self).__init__()
        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        feature_, conv_block = self.build_conv_block(input_dim, dim, output_dim, rate,
                                                     pad_type, norm, use_bias)
        self.feature_ = feature_
        self.conv_block = conv_block

    def build_conv_block(self, input_dim, dim, output_dim, rate,
                         padding_type, norm, use_bias=False):

        # branch feature_: in case the output_dim is different from input
        feature_ = [self.pad_layer(padding_type, padding=0),
                    nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=1,
                              bias=False, dilation=1),
                    self.norm_layer(norm, output_dim),
                    ]
        feature_ = nn.Sequential(*feature_)

        # branch convolution:
        conv_block = []

        conv_block += [self.pad_layer(padding_type, padding=0),
                       nn.Conv2d(input_dim, dim, kernel_size=1, stride=1,
                                 bias=False, dilation=1),
                       self.norm_layer(norm, dim),
                       self.activation]
        # dilated conv, padding = dilation_rate, when k=3, s=1
        conv_block += [self.pad_layer(padding_type, padding=rate),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1,
                                 bias=False, dilation=rate),
                       self.norm_layer(norm, dim),
                       self.activation]
        conv_block += [self.pad_layer(padding_type, padding=0),
                       nn.Conv2d(dim, output_dim, kernel_size=1, stride=1,
                                 bias=False, dilation=1),
                       self.norm_layer(norm, output_dim),
                       ]
        conv_block = nn.Sequential(*conv_block)
        return feature_, conv_block

    @staticmethod
    def pad_layer(padding_type, padding):
        if padding_type == 'reflect':
            pad = nn.ReflectionPad2d(padding)
        elif padding_type == 'replicate':
            pad = nn.ReplicationPad2d(padding)
        elif padding_type == 'zero':
            pad = nn.ZeroPad2d(padding)
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        return pad

    @staticmethod
    def norm_layer(norm, norm_dim):
        if norm == 'bn':
            norm_layer_ = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            norm_layer_ = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            norm_layer_ = LayerNorm(norm_dim)
        elif norm == 'none':
            norm_layer_ = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        return norm_layer_

    def forward(self, x):
        feature_ = self.feature_(x)
        conv = self.conv_block(x)
        out = feature_ + conv
        out = self.activation(out)
        return out


class ASPP(nn.Module):

    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            # BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            # BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            # BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            # BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            # BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # self.conv_cat = nn.Sequential(
        #         nn.Conv2d(dim_out*4, dim_out, 1, 1, padding=0),
        #         SynchronizedBatchNorm2d(dim_out),
        #         nn.ReLU(inplace=True),
        # )

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        # global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        # feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3], dim=1)
        result = self.conv_cat(feature_cat)
        return result


bn_mom = 0.0003
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, atrous=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1 * atrous, dilation=atrous, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, atrous=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, atrous)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = SynchronizedBatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = SynchronizedBatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, atrous=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = SynchronizedBatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1 * atrous, dilation=atrous, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = SynchronizedBatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.bn3 = SynchronizedBatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetAtrous(nn.Module):

    def __init__(self, block, layers, atrous=None, os=16):
        super(ResNetAtrous, self).__init__()
        stride_list = None
        if os == 8:
            stride_list = [2, 1, 1]
        elif os == 16:
            stride_list = [2, 2, 1]
        else:
            raise ValueError('resnet_atrous.py: output stride=%d is not supported.' % os)

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = SynchronizedBatchNorm2d(64, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 64, layers[0])
        self.layer2 = self._make_layer(block, 256, 128, layers[1], stride=stride_list[0])
        self.layer3 = self._make_layer(block, 512, 256, layers[2], stride=stride_list[1], atrous=16 // os)
        self.layer4 = self._make_layer(block, 1024, 512, layers[3], stride=stride_list[2],
                                       atrous=[item * 16 // os for item in atrous])
        self.layers = []

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_layers(self):
        return self.layers

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, atrous=None):
        downsample = None
        if atrous == None:
            atrous = [1] * blocks
        elif isinstance(atrous, int):
            atrous_list = [atrous] * blocks
            atrous = atrous_list
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                SynchronizedBatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride=stride, atrous=atrous[0], downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(planes * block.expansion, planes, stride=1, atrous=atrous[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        self.layers = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        self.layers.append(x)
        x = self.layer1(x)
        self.layers.append(x)
        x = self.layer2(x)
        self.layers.append(x)
        x = self.layer3(x)
        self.layers.append(x)
        x = self.layer4(x)
        self.layers.append(x)

        return x


class FusionBlock(nn.Module):
    def __init__(self, input_dim, dim, output_dim, pad_type, activ, norm, encoder_name='vgg19', num_classes=None):
        """output_shape = [H, W, C]"""
        super(FusionBlock, self).__init__()

        conv = nn.Conv2d
        activ = nn.ReLU(True)
        norm = nn.BatchNorm2d

        num_classes = num_classes if num_classes is not None else 0

        # fusion block
        if encoder_name == 'vgg19':
            n_fuse_filters = [512, 768, 384, 192, 96]
        elif encoder_name == 'vgg11':
            n_fuse_filters = [512, 512, 256, 128, 64]
        elif encoder_name == 'resnet50':
            n_fuse_filters = [256, 1280, 640, 320, 96]
        elif encoder_name == 'resnet101':
            n_fuse_filters = [256, 1280, 640, 320, 96]
        else:
            raise ValueError('Encoder name must in [vgg11, vgg19, resnet50, resnet101], '
                             'however it is {}'.format(encoder_name))
        self.fuse_out = ConvLayer(conv, n_fuse_filters[0], 256, kernel_size=3, stride=1, act=activ, norm=norm)
        self.fuse_deep = ConvLayer(conv, n_fuse_filters[1], 128, kernel_size=3, stride=1, act=activ, norm=norm)
        self.fuse_mid = ConvLayer(conv, n_fuse_filters[2], 64, kernel_size=3, stride=1, act=activ, norm=norm)
        self.fuse_low = ConvLayer(conv, n_fuse_filters[3], 32, kernel_size=3, stride=1, act=activ, norm=norm)
        self.fuse_shallow = ConvLayer(conv, n_fuse_filters[4], 16, kernel_size=3, stride=1, act=activ, norm=norm)
        self.fuse_input = ConvLayer(conv, 16 + input_dim + num_classes, dim, kernel_size=3, stride=1, act=activ, norm=norm)

        # use reflection padding in the last conv layer
        self.fusion_layers = [
            ConvLayer(conv, dim, dim, kernel_size=3, stride=1, norm=norm, act=activ)]
        self.fusion_layers += [
            ConvLayer(conv, dim, dim, kernel_size=3, stride=1, norm=norm, act=activ)]
        self.fusion_layers += [
            ConvLayer(conv, dim, output_dim, stride=1, kernel_size=1, norm=None, act=None)]
        self.fusion_layers = nn.Sequential(*self.fusion_layers)

    @staticmethod
    def _fuse_feature(x, feature):
        _, _, h, w = feature.shape
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        x = torch.cat([x, feature], dim=1)
        return x

    def forward(self, feat_dict):
        x = feat_dict['out']
        x = self.fuse_out(x)
        x = self._fuse_feature(x, feat_dict['deep'])
        x = self.fuse_deep(x)
        x = self._fuse_feature(x, feat_dict['mid'])
        x = self.fuse_mid(x)
        x = self._fuse_feature(x, feat_dict['low'])
        x = self.fuse_low(x)
        x = self._fuse_feature(x, feat_dict['shallow'])
        x = self.fuse_shallow(x)
        x = self._fuse_feature(x, feat_dict['input'])
        x = self.fuse_input(x)

        x = self.fusion_layers(x)
        return x


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation layer: https://zhuanlan.zhihu.com/p/65459972
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y


class ConvLayer(torch.nn.Sequential):
    """
    Conv2d -> BatchNorm -> Activation
    """

    def __init__(self, conv, in_channels, out_channels, kernel_size, stride, padding=None,
                 dilation=1, norm=None, act=None):
        super(ConvLayer, self).__init__()
        # padding = padding or kernel_size // 2
        padding = padding or dilation * (kernel_size - 1) // 2
        self.add_module('conv2d', conv(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation))
        if norm is not None:
            self.add_module('norm', norm(out_channels))
            # self.add_module('norm', norm(out_channels, track_running_stats=True))
        if act is not None:
            self.add_module('act', act)


class GuidanceBlock(torch.nn.Module):
    def __init__(self, channels, fea_channels=256, class_num=21, dilation=1, norm=nn.BatchNorm2d, act=nn.ReLU(True)):
        super(GuidanceBlock, self).__init__()
        conv = nn.Conv2d
        self.fea_dim = fea_channels
        self.sem_dim = class_num
        self.conv_fea = ConvLayer(conv, fea_channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm, act=act)
        self.conv_sem = ConvLayer(conv, class_num, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm, act=act)

        self.conv1 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm, act=act)
        self.conv2 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm,
                               act=None)

        self.fuse_conv = ConvLayer(conv, channels * 2, channels, kernel_size=1, stride=1, dilation=dilation, norm=norm,
                                   act=None)

    def forward(self, x, g):
        residual = x
        h, w = x.shape[2], x.shape[3]
        _g = F.interpolate(g, size=(h, w), mode='bilinear', align_corners=True)
        if _g.shape[1] == self.fea_dim:
            _g = self.conv_fea(_g)
        elif _g.shape[1] == self.sem_dim:
            _g = self.conv_sem(_g)
        else:
            raise ValueError('Guidance channels {} not in [{}, {}]'.format(self.sem_dim, self.fea_dim, _g.shape[0]))
        concat = torch.cat([x, _g], dim=1)
        out = self.fuse_conv(concat)
        out = self.conv1(out)
        out = self.conv2(out)
        out = out * _g
        out = out + residual
        return out


class ChannelAttentionBlock(torch.nn.Module):
    def __init__(self, channels, dilation=1, norm=nn.BatchNorm2d, act=nn.LeakyReLU(True), se_reduction=None, res_scale=1):
        super(ChannelAttentionBlock, self).__init__()
        conv = nn.Conv2d
        self.conv1 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm, act=act)
        self.conv2 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm,
                               act=None)
        self.se_layer = None
        self.res_scale = res_scale
        if se_reduction is not None:
            self.se_layer = SELayer(channels, se_reduction)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se_layer:
            out = self.se_layer(out)
        out = out * self.res_scale
        out = out + residual
        return out


class PyramidPoolingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scales=(4, 8, 16, 32), ct_channels=1):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, scale, ct_channels) for scale in scales])
        self.bottleneck = nn.Conv2d(in_channels + len(scales) * ct_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def _make_stage(self, in_channels, scale, ct_channels):
        # prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        prior = nn.AvgPool2d(kernel_size=(scale, scale))
        conv = nn.Conv2d(in_channels, ct_channels, kernel_size=1, bias=False)
        relu = nn.LeakyReLU(0.2, inplace=True)
        return nn.Sequential(prior, conv, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = torch.cat(
            [F.interpolate(input=stage(feats), size=(h, w), mode='nearest') for stage in self.stages] + [feats], dim=1)
        return self.relu(self.bottleneck(priors))


##################################################################################
# Normalization layers
##################################################################################

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


def _get_norm_layer(norm_type='in'):
    if norm_type == 'bn':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'in':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def _get_active_function(act_type='relu'):
    if act_type == 'relu':
        act_func = nn.ReLU(True)
    elif act_type == 'lrelu':
        act_func = nn.LeakyReLU(0.2, True)
    elif act_type == 'prelu':
        act_func = nn.PReLU()
    elif act_type == 'selu':
        act_func = nn.SELU(inplace=True)
    elif act_type == 'sigmoid':
        act_func = nn.Sigmoid()
    elif act_type == 'tanh':
        act_func = nn.Tanh()
    elif act_type == 'none':
        act_func = None
    else:
        raise NotImplementedError('activation function [%s] is not found' % act_type)
    return act_func


def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


def _unsqueeze_ft(tensor):
    """add new dementions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum', 'sum_size'])
_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


class _SynchronizedBatchNorm(_BatchNorm, ABC):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)

        self._sync_master = SyncMaster(self._data_parallel_master)

        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        # If it is not parallel computation or is in evaluation mode, use PyTorch's implementation.
        if not (self._is_parallel and self.training):
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)

        # Resize the input to (B, C, -1).
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)

        # Compute the sum and square-sum.
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)

        # Reduce-and-broadcast the statistics.
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(input_sum, input_ssum, sum_size))

        # Compute the output.
        if self.affine:
            # MJY:: Fuse the multiplication for speed.
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std * self.weight) + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)

        # Reshape it.
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id

        # parallel_id == 0 means master device.
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""

        # Always using same "device order" makes the ReduceAdd operation faster.
        # Thanks to:: Tete Xiao (http://tetexiao.com/)
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())

        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]  # flatten
        target_gpus = [i[1].sum.get_device() for i in intermediates]

        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)

        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)

        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i * 2:i * 2 + 2])))

        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size

        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data

        return mean, bias_var.clamp(self.eps) ** -0.5


class SynchronizedBatchNorm1d(_SynchronizedBatchNorm):
    r"""Applies Synchronized Batch Normalization over a 2d or 3d input that is seen as a
    mini-batch.

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm1d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal BatchNorm

    Args:
        num_features: num_features from an expected input of size
            `batch_size x num_features [x width]`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm1d, self)._check_input_dim(input)


class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
    r"""Applies Batch Normalization over a 4d input that is seen as a mini-batch
    of 3d inputs

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    This module differs from the built-in PyTorch BatchNorm2d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm2d, self)._check_input_dim(input)


class SynchronizedBatchNorm3d(_SynchronizedBatchNorm):
    r"""Applies Batch Normalization over a 5d input that is seen as a mini-batch
        of 4d inputs

        .. math::

            y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

        This module differs from the built-in PyTorch BatchNorm3d as the mean and
        standard-deviation are reduced across all devices during training.

        For example, when one uses `nn.DataParallel` to wrap the network during
        training, PyTorch's implementation normalize the tensor on each device using
        the statistics only on that device, which accelerated the computation and
        is also easy to implement, but the statistics might be inaccurate.
        Instead, in this synchronized version, the statistics will be computed
        over all training samples distributed on multiple devices.

        Note that, for one-GPU or CPU-only case, this module behaves exactly same
        as the built-in PyTorch implementation.

        The mean and standard-deviation are calculated per-dimension over
        the mini-batches and gamma and beta are learnable parameter vectors
        of size C (where C is the input size).

        During training, this layer keeps a running estimate of its computed mean
        and variance. The running sum is kept with a default momentum of 0.1.

        During evaluation, this running mean/variance is used for normalization.

        Because the BatchNorm is done over the `C` dimension, computing statistics
        on `(N, D, H, W)` slices, it's common terminology to call this Volumetric BatchNorm
        or Spatio-temporal BatchNorm

        Args:
            num_features: num_features from an expected input of
                size batch_size x num_features x depth x height x width
            eps: a value added to the denominator for numerical stability.
                Default: 1e-5
            momentum: the value used for the running_mean and running_var
                computation. Default: 0.1
            affine: a boolean value that when set to ``True``, gives the layer learnable
                affine parameters. Default: ``True``

        Shape:
            - Input: :math:`(N, C, D, H, W)`
            - Output: :math:`(N, C, D, H, W)` (same shape as input)

        Examples:
            >>> # With Learnable Parameters
            >>> m = SynchronizedBatchNorm3d(100)
            >>> # Without Learnable Parameters
            >>> m = SynchronizedBatchNorm3d(100, affine=False)
            >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45, 10))
            >>> output = m(input)
        """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm3d, self)._check_input_dim(input)


class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created by original replication.

    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be called ahead of calling the callback
    of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]

    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


class DataParallelWithCallback(DataParallel):
    """
    Data Parallel with a replication callback.

    An replication callback `__data_parallel_replicate__` of each module will be invoked after being created by
    original `replicate` function.
    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
        # sync_bn.__data_parallel_replicate__ will be invoked.
    """

    def replicate(self, module, device_ids):
        modules = super(DataParallelWithCallback, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules


def patch_replication_callback(data_parallel):
    """
    Monkey-patch an existing `DataParallel` object. Add the replication callback.
    Useful when you have customized `DataParallel` implementation.

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallel(sync_bn, device_ids=[0, 1])
        > patch_replication_callback(sync_bn)
        # this is equivalent to
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
    """

    assert isinstance(data_parallel, DataParallel)

    old_replicate = data_parallel.replicate

    @functools.wraps(old_replicate)
    def new_replicate(module, device_ids):
        modules = old_replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules

    data_parallel.replicate = new_replicate


##################################################################################
# Distribution distance measurements and losses blocks
##################################################################################

class KLDivergence(nn.Module):
    def __init__(self, size_average=None, reduce=True, reduction='mean'):
        super(KLDivergence, self).__init__()
        self.eps = 1e-12
        self.log_softmax = nn.LogSoftmax()
        self.kld = nn.KLDivLoss(size_average=size_average, reduce=reduce, reduction=reduction)
        pass

    def forward(self, x, y):
        # normalize
        x = self.log_softmax(x)
        y = self.log_softmax(y)
        return self.kld(x, y)


class JSDivergence(KLDivergence):
    def __init__(self, size_average=True, reduce=True, reduction='mean'):
        super(JSDivergence, self).__init__(size_average, reduce, reduction)

    def forward(self, x, y):
        # normalize
        x = self.log_softmax(x)
        y = self.log_softmax(y)
        m = 0.5 * (x + y)

        return 0.5 * (self.kld(x, m) + self.kld(y, m))


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        """
        SSIM Loss, return 1 - SSIM
        """
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)

    @staticmethod
    def _gaussian(window_size, sigma):
        gauss = torch.Tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    @staticmethod
    def _ssim(img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img_a, img_b):
        (_, channel, _, _) = img_a.size()

        if channel == self.channel and self.window.data.type() == img_a.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)

            if img_a.is_cuda:
                window = window.cuda(img_a.get_device())
            window = window.type_as(img_a)

            self.window = window
            self.channel = channel

        ssim_v = self._ssim(img_a, img_b, window, self.window_size, channel, self.size_average)

        return 1 - ssim_v


class LossAdaptor(object):
    """
    An adaptor aim to balance loss via the std of the loss
    """

    def __init__(self, queue_size=100, param_only=True):
        self.size = queue_size
        self.history = Queue(maxsize=self.size)
        self.param_only = param_only

    def __call__(self, loss_var):
        if self.history.qsize() < self.size:
            param = 1.
            self.history.put(loss_var)
        else:
            self.history.put(loss_var)
            param = np.mean(self.history.queue)

        if self.param_only:
            return param
        else:
            return param * loss_var


class RetinaLoss(nn.Module):
    def __init__(self):

        super(RetinaLoss, self).__init__()
        self.l1_diff = nn.L1Loss()
        self.sigmoid = nn.Sigmoid()
        self.downsample = nn.AvgPool2d(2)
        self.level = 3
        self.eps = 1e-6
        pass

    @staticmethod
    def compute_gradient(img):
        grad_x = img[:, :, 1:, :] - img[:, :, :-1, :]
        grad_y = img[:, :, :, 1:] - img[:, :, :, :-1]
        return grad_x, grad_y

    def compute_exclusion_loss(self, img1, img2):
        """
        NOTE: To make image1 and image2 look different in retina way, TODO: need to be debug in detail, bad-ass
        :param img1:
        :param img2:
        :return:
        """
        gradx_loss = []
        grady_loss = []

        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)

            if torch.mean(torch.abs(gradx2)) < self.eps or torch.mean(torch.abs(gradx2)) < self.eps:
                gradx_loss.append(0)
                grady_loss.append(0)
                continue

            alphax = 2.0 * torch.mean(torch.abs(gradx1)) / (torch.mean(torch.abs(gradx2)) + self.eps)
            alphay = 2.0 * torch.mean(torch.abs(grady1)) / (torch.mean(torch.abs(grady2)) + self.eps)

            if torch.isnan(alphax) or torch.isnan(alphay):
                gradx_loss.append(0)
                grady_loss.append(0)
                continue

            gradx1_s = (self.sigmoid(gradx1) * 2) - 1
            grady1_s = (self.sigmoid(grady1) * 2) - 1
            gradx2_s = (self.sigmoid(gradx2 * alphax) * 2) - 1
            grady2_s = (self.sigmoid(grady2 * alphay) * 2) - 1

            gradx_loss.append(torch.mean(torch.mul(torch.pow(gradx1_s, 2), torch.pow(gradx2_s, 2)) ** 0.25))
            grady_loss.append(torch.mean(torch.mul(torch.pow(grady1_s, 2), torch.pow(grady2_s, 2)) ** 0.25))

            img1 = self.downsample(img1)
            img2 = self.downsample(img2)

        loss = 0.5 * (sum(gradx_loss) / float(len(gradx_loss)) + sum(grady_loss) / float(len(grady_loss)))

        print(loss)

        return loss

    def compute_gradient_loss(self, img1, img2):
        """
        NOTE: To make image1 and image2 look same in retina way
        :param img1:
        :param img2:
        :return:
        """
        losses = []
        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)

            loss = 0.5 * (self.l1_diff(gradx1, gradx2) + self.l1_diff(grady1, grady2))
            losses.append(loss)

        loss = 0 if len(losses) == 0 else sum(losses) / len(losses)
        return loss

    def forward(self, img_b, img_r, mode='exclusion'):
        """  Mode in [exclusion/gradient] """
        with torch.no_grad():
            if mode == 'exclusion':
                loss = self.compute_exclusion_loss(img_b, img_r)
            elif mode == 'gradient':
                loss = self.compute_gradient_loss(img_b, img_r)
            else:
                raise NotImplementedError("mode should in [exclusion/gradient]")
        return loss


class MaskLoss(nn.Module):
    def __init__(self, reduction):
        super(MaskLoss, self).__init__()
        self.loss = None
        self.reduction = reduction

    def forward(self, x, y, mask):
        if self.loss == None:
            raise ValueError('loss.py: MaskLoss.loss has not been implemented')
        count = torch.sum(mask)
        loss = self.loss(x, y)
        loss = loss * mask
        if self.reduction == 'all':
            return torch.sum(loss) / count
        elif self.reduction == 'none':
            return loss


class MaskCrossEntropyLoss(MaskLoss):
    def __init__(self, reduction='all'):
        super(MaskCrossEntropyLoss, self).__init__(reduction)
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')


class MaskBCELoss(MaskLoss):
    def __init__(self, reduction='all'):
        super(MaskBCELoss, self).__init__(reduction)
        self.loss = torch.nn.BCELoss(reduction='none')


class MaskBCEWithLogitsLoss(MaskLoss):
    def __init__(self, reduction='all'):
        super(MaskBCEWithLogitsLoss, self).__init__(reduction)
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='none')


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def forward(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class VggLoss(nn.Module):
    def __init__(self, pretrained):
        super(VggLoss, self).__init__()
        assert type(pretrained) is Vgg11EncoderMS or type(pretrained) is Vgg19EncoderMS \
               or type(pretrained) is ResNet50AtrousEncoderMS or type(pretrained) is ResNet101AtrousEncoderMS
        self.feature_extractor = pretrained
        self.l1_loss = nn.L1Loss()

    def forward(self, input, output):
        with torch.no_grad():
            vgg_real = self.feature_extractor(input)
            vgg_fake = self.feature_extractor(output)

            p0 = self.l1_loss(vgg_real['input'], vgg_fake['input'])
            p1 = self.l1_loss(vgg_real['shallow'], vgg_fake['shallow']) / 2.6
            p2 = self.l1_loss(vgg_real['low'], vgg_fake['low']) / 4.8
            p3 = self.l1_loss(vgg_real['mid'], vgg_fake['mid']) / 3.7
            p4 = self.l1_loss(vgg_real['deep'], vgg_fake['deep']) / 5.6
            p5 = self.l1_loss(vgg_real['out'], vgg_fake['out']) * 10 / 1.5

        return p0 + p1 + p2 + p3 + p4 + p5


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, input_feat, output_feat):
        with torch.no_grad():
            vgg_real = output_feat
            vgg_fake = input_feat

            p0 = self.l1_loss(vgg_real['input'], vgg_fake['input'])
            p1 = self.l1_loss(vgg_real['shallow'], vgg_fake['shallow']) / 2.6
            p2 = self.l1_loss(vgg_real['low'], vgg_fake['low']) / 4.8
            p3 = self.l1_loss(vgg_real['mid'], vgg_fake['mid']) / 3.7
            p4 = self.l1_loss(vgg_real['deep'], vgg_fake['deep']) / 5.6
            p5 = self.l1_loss(vgg_real['out'], vgg_fake['out']) * 10 / 1.5

        return p0 + p1 + p2 + p3 + p4 + p5


##################################################################################
# Test codes
##################################################################################

def test_gen():
    pass


def test_retina_loss():
    import cv2
    img_1 = cv2.imread('/media/ros/Files/ws/Dataset/Reflection/Berkeley/synthetic/transmission_layer/53.jpg')
    img_2 = cv2.imread('/media/ros/Files/ws/Dataset/Reflection/Berkeley/synthetic/reflection_layer/21.jpg')

    img_1 = cv2.resize(img_1, (300, 300))
    img_2 = cv2.resize(img_2, (300, 300))

    img_1 = np.float32(img_1)
    img_2 = np.float32(img_2)

    img_1 = np.transpose(img_1, (2, 0, 1))
    img_2 = np.transpose(img_2, (2, 0, 1))

    v_1 = torch.from_numpy(img_1).unsqueeze(0)
    v_2 = torch.from_numpy(img_2).unsqueeze(0)

    retina = RetinaLoss()

    l1 = retina.compute_exclusion_loss(v_1, v_2)
    l1_1 = retina.compute_exclusion_loss(v_1, v_1)
    l1_2 = retina.compute_exclusion_loss(v_2, v_2)

    l2 = retina.compute_gradient_loss(v_1, v_2)
    l2_1 = retina.compute_gradient_loss(v_1, v_1)
    l2_2 = retina.compute_gradient_loss(v_2, v_2)

    print(l1.item())
    print(l1_1.item())
    print(l1_2.item())

    print(l2.item())
    print(l2_1.item())
    print(l2_2.item())


def test_model():
    from utils import get_config
    config = get_config('configs/semantic_rr.yaml')
    img_feature_dict = {
        'input': Variable(torch.rand((1, 3, 224, 224))).cuda(),
        'shallow': Variable(torch.rand((1, 64, 56, 56))).cuda(),
        'low': Variable(torch.rand((1, 256, 56, 56))).cuda(),
        'mid': Variable(torch.rand((1, 512, 28, 28))).cuda(),
        'deep': Variable(torch.rand((1, 1024, 14, 14))).cuda(),
        'out': Variable(torch.rand((1, 256, 56, 56))).cuda(),
    }
    sem_feature_dict = {
        'feature': Variable(torch.rand((1, 256, 56, 56))).cuda(),
        'out': Variable(torch.rand((1, 21, 224, 224))).cuda(),
    }

    test_in = Variable(torch.rand((1, 64, 56, 56))).cuda()

    # module1 = GuidanceBlock(channels=64).cuda()
    # module2 = GuidanceBlock(channels=64).cuda()
    #
    # out1 = module1(test_in, sem_feature_dict['feature'])
    # out2 = module2(test_in, sem_feature_dict['out'])
    #
    # print(out1)
    # print(out2)

    model = DirectGenMS(param=config).cuda()
    out = model(img_feature_dict['input'])
    print(out)


if __name__ == '__main__':
    test_model()

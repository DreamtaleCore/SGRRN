from networks import get_generator, get_discriminator, RetinaLoss, VggLoss, LayerNorm, get_classifier, PerceptualLoss
from torchvision.models import vgg11, vgg19, resnet50, resnet101
from utils import weights_init, get_scheduler, get_model_list, label2colormap_batch, compute_miou, to_number
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
# from skimage.measure import compare_psnr


class Trainer(nn.Module):
    def __init__(self, param):
        super(Trainer, self).__init__()
        lr_d = param['lr_d']
        # Initiate the networks
        self.generator = get_generator(param)
        self.discriminator_bg = get_discriminator(param)
        self.discriminator_rf = get_discriminator(param)

        # ############################################################################
        # from thop import profile
        # from thop import clever_format
        # input_i = torch.randn(1, 3, 224, 224)
        # macs, params = profile(self.discriminator_bg, inputs=(input_i, ))
        # print('========================')
        # print('MACs: ',   macs)
        # print('PARAMs: ', params)
        # print('------------------------')
        # macs, params = clever_format([macs, params], "%.3f")
        # print('Clever MACs: ',   macs)
        # print('Clever PARAMs: ', params)
        # print('========================')
        # ############################################################################

        # Setup the optimizers
        beta1 = param['beta1']
        beta2 = param['beta2']
        dis_params = list(self.discriminator_bg.parameters()) + list(self.discriminator_rf.parameters())
        self.dis_opt = torch.optim.Adam(dis_params,
                                        lr=lr_d, betas=(beta1, beta2), weight_decay=param['weight_decay'])
        self.gen_opt = torch.optim.SGD(
            params=[
                {'params': self.get_params(self.generator, key='1x'), 'lr': param.lr_g},
                {'params': self.get_params(self.generator, key='10x'), 'lr': 10 * param.lr_g}
            ],
            momentum=param.momentum
        )
        self.dis_scheduler = get_scheduler(self.dis_opt, param)
        self.gen_scheduler = get_scheduler(self.gen_opt, param)
        # self.dis_scheduler = None
        # self.gen_scheduler = None

        # Network weight initialization
        # self.apply(weights_init(param['init']))
        self.discriminator_bg.apply(weights_init('gaussian'))
        self.discriminator_rf.apply(weights_init('gaussian'))
        self.best_result = float('inf')

        self.perceptual_criterion = PerceptualLoss()
        self.retina_criterion = RetinaLoss()
        self.semantic_criterion = nn.CrossEntropyLoss(ignore_index=255)

        self.best_result = 0

    def get_params(self, model, key):
        for m in model.named_modules():
            if key == '1x':
                if 'backbone' in m[0] and (isinstance(m[1], nn.Conv2d) or
                                           isinstance(m[1], nn.BatchNorm2d) or
                                           isinstance(m[1], nn.InstanceNorm2d) or
                                           isinstance(m[1], LayerNorm)):
                    for p in m[1].parameters():
                        yield p
            elif key == '10x':
                if 'backbone' not in m[0] and (isinstance(m[1], nn.Conv2d) or
                                               isinstance(m[1], nn.BatchNorm2d) or
                                               isinstance(m[1], nn.InstanceNorm2d) or
                                               isinstance(m[1], LayerNorm)):
                    for p in m[1].parameters():
                        yield p
            else:
                raise ValueError('key must in [1x, 10x], but it is {}'.format(key))

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_i):
        self.eval()
        bg, rf, sem = self.generator(x_i)
        self.train()
        return bg, rf, sem

    def print_loss(self):
        info = 'Loss: B-vgg: {:.4f} | B-pixel: {:.4f} | B-retina: {:.4f} | B-gen: {:.4f} | B-dis: {:.4f} | ' \
               'B-sem_acc: {:.4f} | B-sem_miou: {:.4f} | B-ce_loss: {:.4f} | R-vgg: {:.4f} | R-pixel: {:.4f} | ' \
               'R-retina: {:.4f} | R-gen: {:.4f} | R-dis: {:.4f}'.format(
                   to_number(self.loss_percep_bg),
                   to_number(self.loss_pixel_bg),
                   to_number(self.loss_retina_bg),
                   to_number(self.loss_gen_bg),
                   to_number(self.loss_dis_bg),
                   to_number(self.loss_sem_acc),
                   to_number(self.loss_sem_miou),
                   to_number(self.loss_semantic),
                   to_number(self.loss_vgg_rf),
                   to_number(self.loss_pixel_rf),
                   to_number(self.loss_retina_rf),
                   to_number(self.loss_gen_rf),
                   to_number(self.loss_dis_rf))
        return info

    # noinspection PyAttributeOutsideInit
    def gen_update(self, x_in, x_bg, x_rf, x_sm, param):
        self.gen_opt.zero_grad()

        pred_bg, pred_rf, pred_sem, feats = self.generator(x_in)

        # loss constraints
        self.loss_percep_bg = self.perceptual_criterion(self.generator.get_encoder_features(pred_bg),
                                                        self.generator.get_encoder_features(x_bg)) if param['vgg_w'] != 0 else 0
        self.loss_pixel_bg = self.recon_criterion(pred_bg, x_bg) if param['pixel_w'] != 0 else 0
        self.loss_retina_bg = self.retina_criterion(pred_bg, x_bg,
                                                    'gradient') if param['retina_w'] != 0 else 0
        self.loss_gen_bg = self.discriminator_bg.calc_gen_loss(pred_bg) if param['gan_w'] != 0 else 0

        if x_rf is not None:
            self.loss_vgg_rf = self.perceptual_criterion(self.generator.get_encoder_features(pred_rf),
                                                         self.generator.get_encoder_features(x_rf)) if param['vgg_w'] != 0 else 0
            self.loss_pixel_rf = self.recon_criterion(pred_rf, x_rf) if param['pixel_w'] != 0 else 0
            self.loss_retina_rf = self.retina_criterion(pred_bg, pred_rf,
                                                        'gradient') if param['retina_w'] != 0 else 0
            self.loss_gen_rf = self.discriminator_rf.calc_gen_loss(pred_rf) if param['gan_w'] != 0 else 0
        else:
            self.loss_vgg_rf, self.loss_pixel_rf, self.loss_retina_rf, self.loss_gen_rf = 0, 0, 0, 0

        if pred_sem is not None:
            x_sm = x_sm.squeeze(1)
            self.loss_semantic = self.semantic_criterion(pred_sem, x_sm)
            pred_sm = torch.argmax(pred_sem[0], dim=0)
            self.loss_sem_acc = torch.sum(x_sm[0] == pred_sm) / (x_sm.shape[1] * x_sm.shape[2])
            self.loss_sem_miou, self.sem_iou = compute_miou(pred_sm, x_sm)
        else:
            self.loss_semantic = 0
            self.loss_sem_acc = 0
            self.loss_sem_miou = 0
            self.sem_iou = None

        loss_bg = param['vgg_w'] * self.loss_percep_bg + \
                  param['pixel_w'] * self.loss_pixel_bg + \
                  param['retina_w'] + self.loss_retina_bg + \
                  param['gan_w'] + self.loss_gen_bg
        loss_rf = param['vgg_w'] * self.loss_vgg_rf + \
                  param['pixel_w'] * self.loss_pixel_rf + \
                  param['retina_w'] + self.loss_retina_rf + \
                  param['gan_w'] + self.loss_gen_rf
        loss_sem = param['semantic_w'] * self.loss_semantic

        # total loss
        self.loss_total = loss_sem + loss_bg + loss_rf

        self.loss_total.backward()
        self.gen_opt.step()

    def sample(self, x_in, x_bg, x_rf, x_sm):
        self.eval()
        xs_bg, xs_rf, xs_sm = [], [], []
        for i in range(x_in.size(0)):
            _bg, _rf, _sm, _fea = self.generator(x_in[i].unsqueeze(0))

            xs_bg.append(_bg)
            xs_rf.append(_rf)

            if x_sm is not None:
                xs_sm.append(_sm)
                pred_bg, pred_rf, pred_sm = torch.cat(xs_bg), torch.cat(xs_rf), torch.cat(xs_sm)
            else:
                pred_bg, pred_rf = torch.cat(xs_bg), torch.cat(xs_rf)
        self.train()

        x_in = x_in / 2. + 0.5
        x_bg = x_bg / 2. + 0.5
        x_rf = x_rf / 2. + 0.5
        pred_bg = pred_bg / 2. + 0.5
        pred_rf = pred_rf / 2. + 0.5

        if x_sm is not None:
            x_sm_color = label2colormap_batch(x_sm)
            pred_sm = torch.argmax(pred_sm[0], dim=0).detach().long()
            pred_sm_color = label2colormap_batch(pred_sm.unsqueeze(0).unsqueeze(0))

            x_sm_color = x_sm_color.contiguous().float().cuda() / 255.
            pred_sm_color = pred_sm_color.contiguous().float().cuda() / 255.

            return 'in-gt_bg-pred_bg-gt_rf-pred_rf-gt_sem-pred_sem', \
                   (x_in, x_bg, pred_bg, x_rf, pred_rf, x_sm_color, pred_sm_color)
        else:
            return 'in-gt_bg-pred_bg-gt_rf-pred_rf', (x_in, x_bg, pred_bg, x_rf, pred_rf)

    # noinspection PyAttributeOutsideInit
    def dis_update(self, x_in, x_bg, x_rf, param):
        self.dis_opt.zero_grad()
        pred_bg, pred_rf = self.generator(x_in)[:2]
        # D loss
        if param['gan_w'] != 0:
            self.loss_dis_bg = self.discriminator_bg.calc_dis_loss(pred_bg.detach(), x_bg)
            self.loss_dis_rf = self.discriminator_rf.calc_dis_loss(pred_rf.detach(), x_rf) if x_rf is not None else 0
            self.loss_dis_total = param['bg_w'] * self.loss_dis_bg + param['rf_w'] * self.loss_dis_rf
            self.loss_dis_total.backward()
            self.dis_opt.step()
        else:
            self.loss_dis_bg = 0
            self.loss_dis_rf = 0
            self.loss_dis_total = 0

    def evaluate(self, xs, bgs, rfs):
        list_bg_psnr = []
        with torch.no_grad():
            for i in range(xs.size(0)):
                pred_bg, pred_rf, _sm, _fea = self.generator(xs[i].unsqueeze(0))

                pred_bg = pred_bg / 2. + 0.5
                gt_bg = bgs[i].unsqueeze(0) / 2. + 0.5

                list_bg_psnr.append(compare_psnr(np.uint8(np.clip(pred_bg.cpu().numpy(), 0, 1) * 255),
                                                 np.uint8(np.clip(gt_bg.cpu().numpy(), 0, 1) * 255)))
        mean_psnr = np.mean(list_bg_psnr)
        return mean_psnr

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, param):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.generator.load_state_dict(state_dict['generator'])
        self.best_result = state_dict['best_result']
        epoch = int(last_model_name[-6: -3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.discriminator_bg.load_state_dict(state_dict['bg'])
        self.discriminator_rf.load_state_dict(state_dict['rf'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        try:
            self.dis_scheduler = get_scheduler(self.dis_opt, param, epoch)
            self.gen_scheduler = get_scheduler(self.gen_opt, param, epoch)
        except Exception as e:
            print('Warning: {}'.format(e))
        print('Resume from epoch %d' % epoch)
        return epoch

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%03d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%03d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'generator': self.generator.state_dict(), 'best_result': self.best_result}, gen_name)
        torch.save({'bg': self.discriminator_bg.state_dict(), 'rf': self.discriminator_rf.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)


class ClassifyTrainer(nn.Module):
    def __init__(self, param):
        super(ClassifyTrainer, self).__init__()
        lr = param['lr_g']
        # Initiate the networks
        self.model = get_classifier(name=param['model_name'], pretrained=bool(param['pretrained']), num_classes=2)

        self.gen_opt = torch.optim.SGD(lr=lr, params=self.model.parameters(), momentum=param.momentum)
        self.gen_scheduler = get_scheduler(self.gen_opt, param)

        # Network weight initialization
        self.best_result = 0
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x_i, top_k=1):
        self.eval()
        preds = self.model(x_i)
        _, predicted = torch.max(preds.data, top_k)
        self.train()
        return predicted

    def evaluate(self, x_i, y_i):
        with torch.no_grad():
            preds = self.model(x_i)
            predicted = torch.argmax(preds.data, 1)
            rlt = predicted == y_i
            accuracy = float(to_number(rlt.sum())) / y_i.shape[0]
        return accuracy

    # noinspection PyAttributeOutsideInit
    def gen_update(self, x_in, y_gt):
        self.gen_opt.zero_grad()

        preds = self.model(x_in)

        # loss constraints
        self.loss = self.criterion(preds, y_gt)

        self.loss.backward()
        self.gen_opt.step()

    def update_learning_rate(self):
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, param):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "model")
        state_dict = torch.load(last_model_name)
        self.model.load_state_dict(state_dict['model'])
        self.best_result = state_dict['best_result']
        epoch = int(last_model_name[-6: -3])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.gen_opt.load_state_dict(state_dict['model'])
        # Re-initilize schedulers
        try:
            self.gen_scheduler = get_scheduler(self.gen_opt, param, epoch)
        except Exception as e:
            print('Warning: {}'.format(e))
        print('Resume from epoch %d' % epoch)
        return epoch

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'model_%03d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'model': self.model.state_dict(), 'best_result': self.best_result}, gen_name)
        torch.save({'model': self.gen_opt.state_dict()}, opt_name)

    def print_loss(self):
        info = 'Loss: {:.4f}'.format(to_number(self.loss))
        return info


class SemanticTrainer(nn.Module):
    def __init__(self, param):
        super(SemanticTrainer, self).__init__()
        lr_d = param['lr_d']
        # Initiate the networks
        self.generator = get_generator(param)

        # Setup the optimizers
        beta1 = param['beta1']
        beta2 = param['beta2']
        self.gen_opt = torch.optim.SGD(
            params=[
                {'params': self.get_params(self.generator, key='1x'), 'lr': param.lr_g},
                {'params': self.get_params(self.generator, key='10x'), 'lr': 10 * param.lr_g}
            ],
            momentum=param.momentum
        )
        self.gen_scheduler = get_scheduler(self.gen_opt, param)
        # self.dis_scheduler = None
        # self.gen_scheduler = None

        # Network weight initialization
        # self.apply(weights_init(param['init']))
        self.best_result = 0

        self.semantic_criterion = nn.CrossEntropyLoss(ignore_index=255)

    def get_params(self, model, key):
        for m in model.named_modules():
            if key == '1x':
                if 'backbone' in m[0] and (isinstance(m[1], nn.Conv2d) or
                                           isinstance(m[1], nn.BatchNorm2d) or
                                           isinstance(m[1], nn.InstanceNorm2d) or
                                           isinstance(m[1], LayerNorm)):
                    for p in m[1].parameters():
                        yield p
            elif key == '10x':
                if 'backbone' not in m[0] and (isinstance(m[1], nn.Conv2d) or
                                               isinstance(m[1], nn.BatchNorm2d) or
                                               isinstance(m[1], nn.InstanceNorm2d) or
                                               isinstance(m[1], LayerNorm)):
                    for p in m[1].parameters():
                        yield p
            else:
                raise ValueError('key must in [1x, 10x], but it is {}'.format(key))

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_i):
        self.eval()
        _, _, sem = self.generator(x_i)
        self.train()
        return sem

    def print_loss(self):
        info = 'Loss: ' + \
               'B-sem_acc: {:.4f} | B-sem_miou: {:.4f} | B-ce_loss: {:.4f} | '.format(
                   to_number(self.loss_sem_acc),
                   to_number(self.loss_sem_miou),
                   to_number(self.loss_semantic))
        return info

    # noinspection PyAttributeOutsideInit
    def gen_update(self, x_in, x_sm, param):
        self.gen_opt.zero_grad()

        _, _, pred_sem, _ = self.generator(x_in)

        x_sm = x_sm.squeeze(1)
        self.loss_semantic = self.semantic_criterion(pred_sem, x_sm)
        pred_sm = torch.argmax(pred_sem[0], dim=0)
        self.loss_sem_acc = float(to_number(torch.sum(x_sm[0] == pred_sm))) / (x_sm.shape[1] * x_sm.shape[2])
        self.loss_sem_miou, self.sem_iou = compute_miou(pred_sm, x_sm)

        loss_sem = param['semantic_w'] * self.loss_semantic

        # total loss
        self.loss_total = loss_sem

        self.loss_total.backward()
        self.gen_opt.step()

    def evaluate(self, xs, ys):
        with torch.no_grad():
            _, _, _sm, _ = self.generator(xs)
            x_sm = ys.squeeze(1)
            pred_sm = torch.argmax(_sm[0], dim=0)
            loss_sem_miou, sem_iou = compute_miou(pred_sm, x_sm)
        return loss_sem_miou

    def sample(self, x_in, x_sm):
        self.eval()
        xs_sm = []
        for i in range(x_in.size(0)):
            _, _, _sm, _ = self.generator(x_in[i].unsqueeze(0))

            xs_sm.append(_sm)
        pred_sm = torch.cat(xs_sm)

        x_in = x_in / 2. + 0.5
        x_sm_color = label2colormap_batch(x_sm)
        pred_sm = torch.argmax(pred_sm[0], dim=0).detach().long()
        pred_sm_color = label2colormap_batch(pred_sm.unsqueeze(0).unsqueeze(0))

        x_sm_color = x_sm_color.contiguous().float().cuda() / 255.
        pred_sm_color = pred_sm_color.contiguous().float().cuda() / 255.

        self.train()
        return 'input-label-inference', (x_in, x_sm_color, pred_sm_color)

    def update_learning_rate(self):
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, param):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.generator.load_state_dict(state_dict['generator'])
        self.best_result = state_dict['best_result']
        epoch = int(last_model_name[-6: -3])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        try:
            self.gen_scheduler = get_scheduler(self.gen_opt, param, epoch)
        except Exception as e:
            print('Warning: {}'.format(e))
        print('Resume from epoch %d' % epoch)
        return epoch

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%03d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'generator': self.generator.state_dict(), 'best_result': self.best_result}, gen_name)
        torch.save({'gen': self.gen_opt.state_dict()}, opt_name)

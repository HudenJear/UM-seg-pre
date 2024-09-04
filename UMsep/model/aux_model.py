import pandas as pd
import torch, time
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from .base_model import BaseModel
from .build_utils import build_network,build_loss,calculate_metric,csv_write,tensor2img,img_write
from .logger_utils import get_root_logger

class AuxModel(BaseModel):
    """Catintell model for dehaze."""

    def __init__(self, opt):
        super(AuxModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        self.net_d = build_network(opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        load_path_d = self.opt['path'].get('pretrain_network_d', None)
        if load_path_d is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path_d, self.opt['path'].get('strict_load_d', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        self.net_d.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None


        if train_opt.get('perceptual_opt'):
            self.cri_per = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_per = None
        if train_opt.get('dice_opt'):
            self.cri_dice = build_loss(train_opt['dice_opt']).to(self.device)
        else:
            self.cri_dice = None
        if train_opt.get('grad_opt'):
            self.cri_grad = build_loss(train_opt['grad_opt']).to(self.device)
        else:
            self.cri_grad = None
        if train_opt.get('focal_opt'):
            self.cri_focal = build_loss(train_opt['focal_opt']).to(self.device)
        else:
            self.cri_focal = None


        if train_opt.get('net_d_opt'):
            self.cri_d = build_loss(train_opt['net_d_opt']).to(self.device)
        else:
            self.cri_d = None

        if self.cri_per is None and self.cri_pix is None and self.cri_d is None:
            raise ValueError('No loss found. Please use GAN loss and Perceptual loss in train setting.')

        # set up optimizers and schedulers
        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        


        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():  # key ,value
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)


        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.hq = data['hq'].to(self.device)
        self.lq = data['lq'].to(self.device)
        self.label=data['label'].to(self.device)
        # self.DR_grade = data['DR_grade'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        # prediction
        # while 'paras['target_in_vec']' is the input to target net
        self.output = self.net_g(self.lq)


        l_g_total = 0
        loss_dict = OrderedDict()
        # segmentation model is set to work from beginning 
        if self.cri_pix:
            l_g_pix = self.cri_pix(self.output, self.hq)
            l_g_total += l_g_pix
            loss_dict['l_g_pix'] = l_g_pix
          # perceptual loss
        if self.cri_per:
            l_g_percep, l_g_style = self.cri_per(self.output, self.hq)
            if l_g_percep is not None:
                l_g_total += l_g_percep
                loss_dict['l_g_percep'] = l_g_percep
            if l_g_style is not None:
                l_g_total += l_g_style
                loss_dict['l_g_style'] = l_g_style

        if self.cri_dice:
            l_g_dice = self.cri_dice(self.output, self.hq)
            l_g_total += l_g_dice
            loss_dict['l_g_dice'] = l_g_dice
        if self.cri_grad:
            l_g_grad = self.cri_grad(self.output, self.hq)
            l_g_total += l_g_grad
            loss_dict['l_g_grad'] = l_g_grad
        if self.cri_focal:
            l_g_focal = self.cri_focal(self.output, self.hq)
            l_g_total += l_g_focal
            loss_dict['l_g_focal'] = l_g_focal
          
        l_g_total.backward()
        self.optimizer_g.step()


        # optimize net_d after xxxx round, this setting can be modified in the yml file
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):

        # open grad
          for p in self.net_d.parameters():
              p.requires_grad = True

          self.optimizer_d.zero_grad()
          # real
          self.pred = self.net_d(self.output.detach().repeat((1,3,1,1))*self.lq)
          loss_d = self.cri_d(self.pred, self.label)
          loss_dict['loss_d'] = loss_d
          loss_dict['out_d'] = torch.mean(self.pred.detach())
          loss_d.backward()
          
          self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            self.net_d.eval()
            with torch.no_grad():
                # prediction
                self.output =  self.net_g_ema(self.lq)
                self.pred = self.net_d(self.output.detach().repeat((1,3,1,1))*self.lq)
            self.net_d.train()
            
        else:
            self.net_g.eval()
            self.net_d.eval()
            with torch.no_grad():
                # prediction
                self.output = self.net_g(self.lq)
                self.pred = self.net_d(self.output.detach().repeat((1,3,1,1))*self.lq)
            self.net_g.train()
            self.net_d.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        if self.opt['val'].get('use_pbar'):
            self.use_pbar = self.opt['val']['use_pbar']
        else:
            self.use_pbar = False

        with_metrics = self.opt['val'].get('metrics') is not None
        with_iqa_met= self.opt['val'].get('iqa_metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            if with_iqa_met:
                iqa_met={metric: 0 for metric in self.opt['val']['iqa_metrics'].keys()}
                self.metric_results.update(iqa_met)
                # prepare for data gathering
                pred=[]
                iqa_gt=[]
                name_list=[]

            self._initialize_best_metric_results(dataset_name)
        if self.use_pbar: 
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img(visuals['result'])
            if 'hq' in visuals:
                hq_img = tensor2img(visuals['hq'])
                del self.hq

            if with_iqa_met:
                pred.append(self.pred.detach().cpu().numpy().flatten()[0])
                iqa_gt.append(self.label.cpu().numpy().flatten()[0])
                name_list.append(img_name)
            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                img_write(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data = dict(img1=sr_img, img2=hq_img) # img2 is set to be GT img
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if self.use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        

        # launch iqa metrics after gathering all predictions
        s_ratio=self.opt['datasets']['val']['score_scaler'] if 'score_scaler' in self.opt['datasets']['val'] else 1
        iqa_data={}
        iqa_data['pred']=np.array(pred)*s_ratio
        iqa_data['gt']=np.array(iqa_gt)*s_ratio
        
        if with_iqa_met:
            for name, opt_ in self.opt['val']['iqa_metrics'].items():
                    self.metric_results[name] += calculate_metric(iqa_data, opt_)
                    # print(name,':',self.metric_results[name])
        if self.opt['val']['save_prediction']:
            if self.opt['is_train']:
                # save pred is disabled in the training state.
                sav_path = osp.join(self.opt['path']['visualization'], f'_prediction_res.csv')
            else:
                if self.opt['val']['suffix']:
                    sav_path = osp.join(self.opt['path']['visualization'], dataset_name,f'_prediction_{self.opt["val"]["suffix"]}.csv')
                else:
                    sav_path = osp.join(self.opt['path']['visualization'], dataset_name,f'_prediction_res.csv')
            sav_csv = {
                "file_name": name_list,
                "prediction": iqa_data['pred'],
                "ground_truth": iqa_data['gt'],
            }
            if with_iqa_met:
              for name, opt_ in self.opt['val']['iqa_metrics'].items():
                sav_csv[name]=self.metric_results[name]
            # print(sav_csv)
            sav_csv=pd.DataFrame(sav_csv)
            # sav=sav_csv.to_csv(sav_path)
            sav = csv_write(sav_csv, sav_path)
        

        if self.use_pbar:
            pbar.close()

        if with_metrics:
            for metric_name, opt_ in self.opt['val']['metrics'].items():
                self.metric_results[metric_name] /= (idx + 1)
                self._update_best_metric_result(dataset_name, metric_name, self.metric_results[metric_name], current_iter)
        if with_iqa_met :
            for metric_name, opt_ in self.opt['val']['iqa_metrics'].items():
                self._update_best_metric_result(dataset_name, metric_name, self.metric_results[metric_name], current_iter)
        if with_iqa_met or with_metrics:
            self._log_validation_metric_values(current_iter , dataset_name, tb_logger)

    def _initialize_best_metric_results(self, dataset_name):
        """Initialize the best metric results dict for recording the best metric value and iteration."""
        if hasattr(self, 'best_metric_results') and dataset_name in self.best_metric_results:
            return
        elif not hasattr(self, 'best_metric_results'):
            self.best_metric_results = dict()

        # add a dataset record
        record = dict()
        # print(self.opt['val']['metrics'].items())
        for metric, content in self.opt['val']['metrics'].items():
            better = content.get('better', 'higher')
            init_val = float('-inf') if better == 'higher' else float('inf')
            record[metric] = dict(better=better, val=init_val, iter=-1)
        if self.opt['val'].get('iqa_metrics') is not None:
            for metric, content in self.opt['val']['iqa_metrics'].items():
              better = content.get('better', 'higher')
              init_val = float('-inf') if better == 'higher' else float('inf')
              record[metric] = dict(better=better, val=init_val, iter=-1)
        self.best_metric_results[dataset_name] = record

    def _update_best_metric_result(self, dataset_name, metric, val,current_iter):
        if self.best_metric_results[dataset_name][metric]['better'] == 'higher':
            if val >= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['iter'] = current_iter
        else:
            if val <= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['iter'] = current_iter

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        # print(out_dict['lq'].shape)
        out_dict['result'] = self.output.detach().cpu()
        # print(out_dict['result'].shape)
        if hasattr(self, 'hq'):
            out_dict['hq'] = self.hq.detach().cpu()
        return out_dict
        
        
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        # print(self.metric_results.items())
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)


    def save(self, epoch, current_iter,multi_round=0):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter,round=multi_round, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter,round=multi_round)
        self.save_network(self.net_d, 'net_d', current_iter,round=multi_round)
        self.save_training_state(epoch, current_iter)


    # rewrite the saving
    def save_network(self, net, net_label, current_iter,round, param_key='params'):
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{round}_{net_label}_{current_iter}.pth'
        save_path = osp.join(self.opt['path']['models'], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                torch.save(save_dict, save_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'Save model error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f'Still cannot save {save_path}. Just ignore it.')
            # raise IOError(f'Cannot save {save_path}.')

class ApartModel(AuxModel):
    """Catintell model for dehaze."""

    
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        # prediction
        # while 'paras['target_in_vec']' is the input to target net
        self.output = self.net_g(self.lq)


        l_g_total = 0
        loss_dict = OrderedDict()
        # segmentation model is set to work from beginning 
        if self.cri_pix:
            l_g_pix = self.cri_pix(self.output, self.hq)
            l_g_total += l_g_pix
            loss_dict['l_g_pix'] = l_g_pix
          # perceptual loss
        if self.cri_per:
            l_g_percep, l_g_style = self.cri_per(self.output, self.hq)
            if l_g_percep is not None:
                l_g_total += l_g_percep
                loss_dict['l_g_percep'] = l_g_percep
            if l_g_style is not None:
                l_g_total += l_g_style
                loss_dict['l_g_style'] = l_g_style

        if self.cri_dice:
            l_g_dice = self.cri_dice(self.output, self.hq)
            l_g_total += l_g_dice
            loss_dict['l_g_dice'] = l_g_dice
        if self.cri_grad:
            l_g_grad = self.cri_grad(self.output, self.hq)
            l_g_total += l_g_grad
            loss_dict['l_g_grad'] = l_g_grad
        if self.cri_focal:
            l_g_focal = self.cri_focal(self.output, self.hq)
            l_g_total += l_g_focal
            loss_dict['l_g_focal'] = l_g_focal
          
        l_g_total.backward()
        self.optimizer_g.step()


        # optimize net_d after xxxx round, this setting can be modified in the yml file
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):

        # open grad
          for p in self.net_d.parameters():
              p.requires_grad = True

          self.optimizer_d.zero_grad()
          # real
          self.pred = self.net_d(self.lq)
          loss_d = self.cri_d(self.pred, self.label)
          loss_dict['loss_d'] = loss_d
          loss_dict['out_d'] = torch.mean(self.pred.detach())
          loss_d.backward()
          
          self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            self.net_d.eval()
            with torch.no_grad():
                # prediction
                self.output =  self.net_g_ema(self.lq)
                self.pred = self.net_d(self.lq)
            self.net_d.train()
            
        else:
            self.net_g.eval()
            self.net_d.eval()
            with torch.no_grad():
                # prediction
                self.output = self.net_g(self.lq)
                self.pred = self.net_d(self.lq)
            self.net_g.train()
            self.net_d.train()

    

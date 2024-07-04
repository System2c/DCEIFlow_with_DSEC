import time
import numpy as np
import torch as th
from torchvision import utils
from tqdm import tqdm
from utils.helper_functions import *
from utils.utils import InputPadder, build_module
import utils.visualization as visualization
import utils.filename_templates as TEMPLATES
import utils.helper_functions as helper
import utils.logger as logger
from utils import image_utils
import pdb
import torch.nn as nn

# import torch.distributed as dist
# dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)

def reduce_list(lists, nprocs):
    new_lists = {}
    for key, value in lists.items():
        rt = value.clone()
        # dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= nprocs
        new_lists[key] = rt.item()
    return new_lists

def ensure_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)

class Train:
    """
    Train class

    """

    def __init__(self, model, config,
                 data_loader, loss=None, optimizer=None, train_logger=None, lr_scheduler=None, scaler=None):
        self.downsample = False # Downsampling for Rebuttal
        self.model = model
        self.config = config
        self.data_loader = data_loader
        self.loss = loss
        self.optimizer = optimizer
        self.logger = logger
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler
        if config['cuda'] and not torch.cuda.is_available():
            print('Warning: There\'s no CUDA support on this machine, '
                                'training is performed on CPU.')
        else:
            self.gpu = torch.device('cuda:' + str(config['gpu']))
            self.model = self.model.to(self.gpu)
        # if self.logger is None:
        #     def print_line(line, subname=None):
        #             print(line)
        #     self.log_info = print_line
        # else:
        #     self.log_info = self.logger.log_info

    def summary(self):
            self.logger.write_line("====================================== TRAIN SUMMARY ======================================", True)
            self.logger.write_line("Model:\t\t\t" + self.model.__class__.__name__, True)
            self.logger.write_line("Trainer:\t\t" + self.__class__.__name__, True)
            self.logger.write_line("Train Set:\t" + self.data_loader.dataset.__class__.__name__, True)
            self.logger.write_line("\t-Dataset length:\t"+str(len(self.data_loader)), True)
            self.logger.write_line("\t-Batch size:\t\t" + str(self.data_loader.batch_size), True)
            self.logger.write_line("==========================================================================================", True)

    # 可以设置使得某些神经网络参数冻结，p.requires_grad = False 用于设置参数不参与梯度更新
    def weight_fix(self, way, refer_dict=None):

        # fix weights
        if way == 'checkpoint':
            assert refer_dict is not None
            for n, p in self.model.named_parameters():
                if n in refer_dict.keys():
                    p.requires_grad = False
        elif way == 'encoder':
            for n, p in self.model.named_parameters():
                if 'fnet' in n or 'cnet' in n or 'enet' in n or 'fusion' in n:
                    p.requires_grad = False
        elif way == 'event':
            for n, p in self.model.named_parameters():
                if 'enet' in n or 'fusion' in n:
                    p.requires_grad = False
        elif way == 'eventencoder':
            for n, p in self.model.named_parameters():
                if 'enet' in n:
                    p.requires_grad = False
        elif way == 'eventfusion':
            for n, p in self.model.named_parameters():
                if 'fusion' in n:
                    p.requires_grad = False
        elif way == 'imageencoder':
            for n, p in self.model.named_parameters():
                if 'fnet' in n or 'cnet' in n:
                    p.requires_grad = False
        elif way == 'raft':
            for n, p in self.model.named_parameters():
                if 'fnet' in n or 'cnet' in n or 'update_block' in n:
                    p.requires_grad = False
        elif way == 'allencoder':
            for n, p in self.model.named_parameters():
                if 'enet' in n or 'fusion' in n or 'fnet' in n or 'cnet' in n:
                    p.requires_grad = False
        elif way == 'update':
            for n, p in self.model.named_parameters():
                if 'update_block' in n:
                    p.requires_grad = False

        self.log_info("Weight fix way: {} complete.".format(way if way != "" else "None"), "trainer")


    # 部分加载模型参数。这个方法允许从指定路径加载预训练的模型参数，同时可以指定只加载模型的某些部分，并且可以选择是否将这些参数应用到当前模型上
    def partial_load(self, path, weight_fix=None, not_load=False):
        # partial parameters loading
        assert path != ''
        load_dict = torch.load(path, map_location=torch.device("cpu"))
        try:
            if "model" not in load_dict.keys():
                pretrained_dict = {k: v for k, v in load_dict.items() if k in self.model.state_dict().keys() \
                    and k != 'module.update_block.encoder.conv.weight' \
                    and k != 'module.update_block.encoder.conv.bias' \
                    and not k.startswith('module.update_block.flow_enc')}
            else:
                pretrained_dict = {k: v for k, v in load_dict.pop("model").items() if k in self.model.state_dict().keys() \
                    and k != 'module.update_block.encoder.conv.weight' \
                    and k != 'module.update_block.encoder.conv.bias' \
                    and not k.startswith('module.update_block.flow_enc')}
            assert len(pretrained_dict.keys()) > 0
            if not not_load:
                self.model.load_state_dict(pretrained_dict, strict=False)
                self.log_info("Partial load model from {} complete.".format(path), "trainer")
            else:
                self.log_info("Partial load dict from {} only for weight fix, but not load to model.".format(path), "trainer")
        except:
            raise KeyError("'model' not in or mismatch state_dict.keys(), please check partial checkpoint path {}".format(path))

        self.weight_fix(weight_fix, pretrained_dict)

    def load(self, path, only_model=True):
        assert path != ''
        state_dict = torch.load(path, map_location=torch.device("cpu"))

        # 尝试装载预训练模型
        try:
            if "model" in state_dict.keys():
                state_dict = state_dict.pop("model")
            elif 'model_state_dict' in state_dict.keys():
                state_dict = state_dict.pop("model_state_dict")

            if "module." in list(state_dict.keys())[0]:
                for key in list(state_dict.keys()):
                    state_dict.update({key[7:]: state_dict.pop(key)})
            # print(state_dict)
            # del state_dict['fnet.conv1.weight']
            # del state_dict['cnet.conv1.weight']
            padding_tensor = torch.zeros(64, 12, 7, 7)
            state_dict['fnet.conv1.weight'] = torch.cat((state_dict['fnet.conv1.weight'], padding_tensor), dim=1)
            state_dict['cnet.conv1.weight'] = torch.cat((state_dict['cnet.conv1.weight'], padding_tensor), dim=1)
            self.model.load_state_dict(state_dict)
            # print(12222222222321)
        except:
            raise KeyError("'model' not in or mismatch state_dict.keys(), please check checkpoint path {}".format(path))
            self.model.load_state_dict(state_dict)

        index = 0

        # 尝试装载loss、optimizer等参数
        if not only_model:
            try:
                self.optimizer.load_state_dict(state_dict.pop("optimizer"))
            except:
                self.log_info("'optimizer' not in state_dict.keys(), skip it.", "trainer")

            try:
                self.lr_scheduler.load_state_dict(state_dict.pop("lr_scheduler"))
            except:
                self.log_info("'lr_scheduler' not in state_dict.keys(), skip it.", "trainer")

            try:
                index = state_dict.pop("index")
            except:
                self.log_info("'index' not in state_dict.keys(), set to 0.", "trainer")

            self.log_info("Load model/optimizer/index from {} complete, index {}".format(path, index), "trainer")
        else:
            # (self.log_info
             print("Load model from {} complete, index {}".format(path, index), "trainer")

        return index
    
    # 保存相关参数的
    def store(self, path, name, index=None):
        if path != "" and name != "":
            checkpoint = {}
            checkpoint["model"] = self.model.state_dict()
            checkpoint["optimizer"] = self.optimizer.state_dict()
            checkpoint["lr_scheduler"] = self.lr_scheduler.state_dict()
            checkpoint["index"] = index
            
            # 确保文件是否存在
            ensure_folder(path)
            save_path = os.path.join(path, "{}_{}.pth".format(name, checkpoint["index"]))
            torch.save(checkpoint, save_path)
            print("<<< Save model to {} complete".format(save_path), "trainer")

    def move_batch_to_cuda(self, batch):
        return move_dict_to_cuda(batch, self.gpu)

    def _train(self, args):
        start = time.time()     # 开始时间
        self.model.train()
        # 评价函数
        metric_fun = build_module("core.metric", "Combine")(args)
        bar = tqdm(total=len(self.data_loader), position=0, leave=True)
        # 在需要设置断点的位置添加以下语句
        # pdb.set_trace()
        name = ["DSEC/thun_00_a"]
        for batch_idx, batch in enumerate(self.data_loader):

            if 'flow_gt' not in batch.keys():
                break

            # 声明im1，im2
            im1 = batch['event_volume_old']
            im2 = batch['event_volume_new']
            # Move Data to GPU
            pad = 8
            padder = InputPadder(im1.shape, div=pad)
            # pad_batch = padder.pad_batch(batch)
            
            if next(self.model.parameters()).is_cuda:
                batch = self.move_batch_to_cuda(batch)
            # Network Forward Pass
            tm = time.time()        # 输入时间
            output = self.run_network(batch)



            # 训练参数更新
            loss = self.loss(output, batch)
            self.optimizer.zero_grad()
            reduced_loss = reduce_list(loss, args.nprocs)
            self.scaler.scale(loss['loss']).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
            self.scaler.step(self.optimizer)
            self.lr_scheduler.step()
            self.scaler.update()

            elapsed = time.time() - tm      # 输入到输出的时间
            name = ["DSEC/thun_00_a"]       # 根据需要修改名字
            batch['basename'] = name
            output['flow_pred'] = padder.unpad(output['flow_final'])
            batch['flow_est'] = output['flow_pred']
            if batch['save_submission']:
                metric_each = metric_fun.calculate(output, batch, name)
                # pdb.set_trace()
                # print("Sample {}/{}".format(batch_idx + 1, len(self.data_loader)))
            
                reduced_metric_each = metric_each
                # 记录和保存训练过程中的指标
                metric_fun.push(reduced_metric_each)

                reduced_metric_each.update({'time': elapsed})

                bar.set_description("{}/{}[{}:{}],time:{:8.6f}, epe:{:8.6f}".format((batch_idx + 1) * len(batch['basename']), \
                    len(self.data_loader.dataset), batch_idx + 1, batch['basename'][0], elapsed, metric_each['epe']))
            else:
                bar.set_description("{}/{}[{}:{}],time:{:8.6f}, epe:{}".format((batch_idx + 1) * len(batch['basename']), \
                    len(self.data_loader.dataset), batch_idx + 1, batch['basename'][0], elapsed, "No results."))
            bar.update(1)
            
            
        # 测试数据
        metrics_str, all_metrics = metric_fun.summary()
        metric_fun.clear()
        self.model.train()
        bar.close()

        # 日志的收集
        log_info = []
        log_info.append("<<< In {} eval: {} (100X F1), with time {}s.".format(name, metrics_str, time.time() - start))
        # Log Generation
        log = {}
        log['evaluation_info'] = log_info

        return log

    def run_network(self, batch):
        # RAFT just expects two images as input. cleanest. code. ever.
        if not self.downsample:
                im1 = batch['event_volume_old']
                im2 = batch['event_volume_new']
        else:
            im1 = torch.nn.functional.interpolate(batch['event_volume_old'], scale_factor=0.5)
            im2 = torch.nn.functional.interpolate(batch['event_volume_new'], scale_factor=0.5)
        # print(im1.shape)
        # pdb.set_trace()
        output = self.model(image1=im1, image2=im2, batch=batch)
        return output

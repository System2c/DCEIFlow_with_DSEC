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

import torch.distributed as dist

def reduce_list(lists, nprocs):
    new_lists = {}
    for key, value in lists.items():
        rt = value.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= nprocs
        new_lists[key] = rt.item()
    return new_lists



class Test(object):
    """
    Test class

    """

    def __init__(self, model, config,
                 data_loader, visualizer, test_logger=None, save_path=None, additional_args=None):
        self.downsample = False # Downsampling for Rebuttal
        self.model = model
        self.config = config
        self.data_loader = data_loader
        self.additional_args = additional_args
        if config['cuda'] and not torch.cuda.is_available():
            print('Warning: There\'s no CUDA support on this machine, '
                                'training is performed on CPU.')
        else:
            self.gpu = torch.device('cuda:' + str(config['gpu']))
            self.model = self.model.to(self.gpu)
        if save_path is None:
            self.save_path = helper.create_save_path(config['save_dir'].lower(),
                                           config['name'].lower())
        else:
            self.save_path=save_path
        if logger is None:
            self.logger = logger.Logger(self.save_path)
        else:
            self.logger = test_logger
        if isinstance(self.additional_args, dict) and 'name_mapping_test' in self.additional_args.keys():
            visu_add_args = {'name_mapping' : self.additional_args['name_mapping_test']}
        else:
            visu_add_args = None
        self.visualizer = visualizer(data_loader, self.save_path, additional_args=visu_add_args)

    def summary(self):
        self.logger.write_line("====================================== TEST SUMMARY ======================================", True)
        self.logger.write_line("Model:\t\t\t" + self.model.__class__.__name__, True)
        self.logger.write_line("Tester:\t\t" + self.__class__.__name__, True)
        self.logger.write_line("Test Set:\t" + self.data_loader.dataset.__class__.__name__, True)
        self.logger.write_line("\t-Dataset length:\t"+str(len(self.data_loader)), True)
        self.logger.write_line("\t-Batch size:\t\t" + str(self.data_loader.batch_size), True)
        self.logger.write_line("==========================================================================================", True)

    def run_network(self, epoch):
        raise NotImplementedError

    def move_batch_to_cuda(self, batch):
        raise NotImplementedError

    def visualize_sample(self, batch):
        self.visualizer(batch)

    def visualize_sample_dsec(self, batch, batch_idx):
        self.visualizer(batch, batch_idx, None)

    def get_estimation_and_target(self, batch):
        # Returns the estimation and target of the current batch
        raise NotImplementedError

    def _test(self, args):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        start = time.time()     # 开始时间
        self.model.eval()
        # 评价函数
        metric_fun = build_module("core.metric", "Combine")(args)

        bar = tqdm(total=len(self.data_loader), position=0, leave=True)
        with torch.no_grad():
            # 在需要设置断点的位置添加以下语句
            # pdb.set_trace()
            for batch_idx, batch in enumerate(self.data_loader):
                
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
                # 获取运行时间
                elapsed = time.time() - tm  # 输入到输出的时间
                output['flow_pred'] = padder.unpad(output['flow_final'])
                
                name = ["DSEC/thun_00_a"]       # 根据需要修改名字
                batch['basename'] = name
                
                # 下标符合提交要求，才计算误差
                
                if batch['save_submission']:
                    metric_each = metric_fun.calculate(output, batch, name)
                    # pdb.set_trace()
                    print("Sample {}/{}".format(batch_idx + 1, len(self.data_loader)))
                
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
            
                batch['flow_est'] = output['flow_pred']
                # Visualize
                self.visualize_sample_dsec(batch, batch_idx)
                # pdb.set_trace()  

            # 测试数据
            metrics_str, all_metrics = metric_fun.summary()
            metric_fun.clear()
            self.model.train()
            bar.close()
            
            # _print("<<< In {} eval: {} (100X F1), with time {}s.".format(name, metrics_str, time.time() - start), "evaluate")

        
        # 日志的收集
        log_info = []
        log_info.append("<<< In {} eval: {} (100X F1), with time {}s.".format(name, metrics_str, time.time() - start))
        # Log Generation
        log = {}
        log['evaluation_info'] = log_info

        return log

class TestRaftEvents(Test):
    def move_batch_to_cuda(self, batch):
        return move_dict_to_cuda(batch, self.gpu)    
    
    def get_estimation_and_target(self, batch):
        if not self.downsample:
            if 'gt_valid_mask' in batch.keys():
                return batch['flow_est'].cpu().data, (batch['flow'].cpu().data, batch['gt_valid_mask'].cpu().data)
            return batch['flow_est'].cpu().data, batch['flow'].cpu().data
        else:
            f_est = batch['flow_est'].cpu().data
            f_gt = torch.nn.functional.interpolate(batch['flow'].cpu().data, scale_factor=0.5)
            if 'gt_valid_mask' in batch.keys():
                f_mask = torch.nn.functional.interpolate(batch['gt_valid_mask'].cpu().data, scale_factor=0.5)
                return f_est, (f_gt, f_mask)
            return f_est, f_gt

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
        # print(batch['flow_list'])
        # pdb.set_trace()
        # batch['flow_est'] = batch['flow_list'][-1]

class TestRaftEventsWarm(Test):
    def __init__(self, model, config,
                 data_loader, visualizer, test_logger=None, save_path=None, additional_args=None):
        super(TestRaftEventsWarm, self).__init__(model, config,
                                                 data_loader, visualizer, test_logger, save_path,
                                                 additional_args=additional_args)
        self.subtype = config['subtype'].lower()
        print('Tester Subtype: {}'.format(self.subtype))
        self.net_init = None # Hidden state of the refinement GRU
        self.flow_init = None
        self.idx_prev = None
        self.init_print=False
        assert self.data_loader.batch_size == 1, 'Batch size for recurrent testing must be 1'

    def move_batch_to_cuda(self, batch):
        return move_list_to_cuda(batch, self.gpu)

    def get_estimation_and_target(self, batch):
        if not self.downsample:
            if 'gt_valid_mask' in batch[-1].keys():
                return batch[-1]['flow_est'].cpu().data, (batch[-1]['flow'].cpu().data, batch[-1]['gt_valid_mask'].cpu().data)
            return batch[-1]['flow_est'].cpu().data, batch[-1]['flow'].cpu().data
        else:
            f_est = batch[-1]['flow_est'].cpu().data
            f_gt = torch.nn.functional.interpolate(batch[-1]['flow'].cpu().data, scale_factor=0.5)
            if 'gt_valid_mask' in batch[-1].keys():
                f_mask = torch.nn.functional.interpolate(batch[-1]['gt_valid_mask'].cpu().data, scale_factor=0.5)
                return f_est, (f_gt, f_mask)
            return f_est, f_gt

    def visualize_sample(self, batch):
        self.visualizer(batch[-1])

    def visualize_sample_dsec(self, batch, batch_idx):
        self.visualizer(batch[-1], batch_idx, None)

    def check_states(self, batch):
        # 0th case: there is a flag in the batch that tells us to reset the state (DSEC)
        if 'new_sequence' in batch[0].keys():
            if batch[0]['new_sequence'].item() == 1:
                self.flow_init = None
                self.net_init = None
                self.logger.write_line("Resetting States!", True)
        else:
            # During Validation, reset state if a new scene starts (index jump)
            if self.idx_prev is not None and batch[0]['idx'].item() - self.idx_prev != 1:
                self.flow_init = None
                self.net_init = None
                self.logger.write_line("Resetting States!", True)
            self.idx_prev = batch[0]['idx'].item()

    def run_network(self, batch):
        self.check_states(batch)
        for l in range(len(batch)):
            # Run Recurrent Network for this sample

            if not self.downsample:
                im1 = batch[l]['event_volume_old']
                im2 = batch[l]['event_volume_new']
            else:
                im1 = torch.nn.functional.interpolate(batch[l]['event_volume_old'], scale_factor=0.5)
                im2 = torch.nn.functional.interpolate(batch[l]['event_volume_new'], scale_factor=0.5)
            flow_low_res, batch[l]['flow_list'] = self.model(image1=im1,
                                                                image2=im2,
                                                                flow_init=self.flow_init)

        batch[l]['flow_est'] = batch[l]['flow_list'][-1]
        self.flow_init = image_utils.forward_interpolate_pytorch(flow_low_res)
        batch[l]['flow_init'] = self.flow_init

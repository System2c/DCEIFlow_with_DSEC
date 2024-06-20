import os
import hdf5plugin
os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"]="1"
from loader.loader_mvsec_flow import *
from loader.loader_dsec import *
from utils.logger import *
import utils.helper_functions as helper
import json
from torch.utils.data.dataloader import DataLoader
from utils import visualization as visu
import argparse
from test import *
# import git
import torch.nn
from model import eraft
from core import DCEIFlow
from utils.utils import setup_seed, count_parameters, count_all_parameters, build_module
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pdb

def initialize_tester(config):
    # Warm Start
    if config['subtype'].lower() == 'warm_start':
        return TestRaftEventsWarm
    # Classic
    else:
        return TestRaftEvents

def get_visualizer(args):
    # DSEC dataset
    if args.dataset.lower() == 'dsec':
        return visualization.DsecFlowVisualizer
    # MVSEC dataset
    else:
        return visualization.FlowVisualizerEvents
        
def test(args):
    # Choose correct config file
    if args.dataset.lower()=='dsec':
        if args.type.lower()=='warm_start':
            config_path = 'config/dsec_warm_start.json'
        elif args.type.lower()=='standard':
            config_path = 'config/dsec_standard.json'
        else:
            raise Exception('Please provide a valid argument for --type. [warm_start/standard]')
    elif args.dataset.lower()=='mvsec':
        if args.frequency==20:
            config_path = 'config/mvsec_20.json'
        elif args.frequency==45:
            config_path = 'config/mvsec_45.json'
        else:
            raise Exception('Please provide a valid argument for --frequency. [20/45]')
        if args.type=='standard':
            raise NotImplementedError('Sorry, this is not implemented yet, please choose --type warm_start')
    else:
        raise Exception('Please provide a valid argument for --dataset. [dsec/mvsec]')


    # Load config file
    config = json.load(open(config_path))
    # Create Save Folder
    save_path = helper.create_save_path(config['save_dir'].lower(), config['name'].lower())
    print('Storing output in folder {}'.format(save_path))
    # Copy config file to save dir
    json.dump(config, open(os.path.join(save_path, 'config.json'), 'w'),
              indent=4, sort_keys=False)
    # Logger
    logger = Logger(save_path)
    logger.initialize_file("test")

    # Instantiate Dataset
    # Case: DSEC Dataset
    additional_loader_returns = None
    if args.dataset.lower() == 'dsec':
        # Dsec Dataloading
        loader = DatasetProvider(
            dataset_path=Path(args.path),
            representation_type=RepresentationType.VOXEL,
            delta_t_ms=100,
            config=config,
            type=config['subtype'].lower(),
            visualize=args.visualize)
        loader.summary(logger)
        test_set = loader.get_test_dataset()
        additional_loader_returns = {'name_mapping_test': loader.get_name_mapping_test()}
    
    # Case: MVSEC Dataset
    else:
        if config['subtype'].lower() == 'standard':
            test_set = MvsecFlow(
                args = config["data_loader"]["test"]["args"],
                type='test',
                path=args.path
            )
        elif config['subtype'].lower() == 'warm_start':
            test_set = MvsecFlowRecurrent(
                args = config["data_loader"]["test"]["args"],
                type='test',
                path=args.path
            )
        else:
            raise NotImplementedError 
        test_set.summary(logger)

    # Instantiate Dataloader
    test_set_loader = DataLoader(test_set,
                                 batch_size=config['data_loader']['test']['args']['batch_size'],
                                 shuffle=config['data_loader']['test']['args']['shuffle'],
                                 num_workers=args.num_workers,
                                 drop_last=True)

    # Load Model
    # model = eraft.ERAFT(
    #     config=config, 
    #     n_first_channels=config['data_loader']['test']['args']['num_voxel_bins']
    # )
    # 构建模型----------------------------------------
    config=config
    n_first_channels=config['data_loader']['test']['args']['num_voxel_bins']
    model = build_module("core", args.model)(args, config, n_first_channels)

    # model = DCEIFlow.DCEIFlow(
    #     config=config, 
    #     n_first_channels=config['data_loader']['test']['args']['num_voxel_bins']
    # )
    # Load Checkpoint
    # checkpoint = torch.load(config['test']['checkpoint'])
    # model.load_state_dict(checkpoint['model'])
    # model.load_state_dict(checkpoint)

    state_dict = torch.load(config['test']['checkpoint'], map_location=torch.device("cpu")) 
    try:
        if "model" in state_dict.keys():
            state_dict = state_dict.pop("model")
        elif 'model_state_dict' in state_dict.keys():
            state_dict = state_dict.pop("model_state_dict")

        if "module." in list(state_dict.keys())[0]:
            for key in list(state_dict.keys()):
                state_dict.update({key[7:]:state_dict.pop(key)})
        # print(state_dict)
        # del state_dict['fnet.conv1.weight']
        # del state_dict['cnet.conv1.weight']
        padding_tensor = torch.zeros(64, 12, 7, 7)
        state_dict['fnet.conv1.weight'] = torch.cat((state_dict['fnet.conv1.weight'], padding_tensor), dim=1)
        state_dict['cnet.conv1.weight'] = torch.cat((state_dict['cnet.conv1.weight'], padding_tensor), dim=1)
        model.load_state_dict(state_dict)
        # print(12222222222321)
    except:
        raise KeyError("'model' not in or mismatch state_dict.keys(), please check checkpoint path {}".format(config['test']['checkpoint']))

        model.load_state_dict(state_dict)
    # Get Visualizer
    visualizer = get_visualizer(args)

    # Initialize Tester
    test = initialize_tester(config)

    test = test(
        model=model,
        config=config,
        data_loader=test_set_loader,
        test_logger=logger,
        save_path=save_path,
        visualizer=visualizer,
        additional_args=additional_loader_returns
    )

    test.summary()
    # 在需要设置断点的位置添加以下语句
    # pdb.set_trace()
    testlog = test._test(args)
    print(testlog['evaluation_info'])

if __name__ == '__main__':
    config_path = "config/config_test.json"
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help="Dataset path", required=True)
    parser.add_argument('-d', '--dataset', default="dsec", type=str, help="Which dataset to use: ([dsec]/mvsec)")
    parser.add_argument('-f', '--frequency', default=20, type=int, help="Evaluation frequency of MVSEC dataset ([20]/45) Hz")
    parser.add_argument('-t', '--type', default='warm_start', type=str, help="Evaluation type ([warm_start]/standard)")
    parser.add_argument('-v', '--visualize', action='store_true', help='Provide this argument s.t. DSEC results are visualized. MVSEC experiments are always visualized.')
    parser.add_argument('-n', '--num_workers', default=0, type=int, help='How many sub-processes to use for data loading')
    # 以下是为DCEIFlow模型适配而添加的参数 
    parser.add_argument('--event_bins', type=int, default=5, \
            help='number of bins in the voxel grid event tensor')
    parser.add_argument('--no_event_polarity', dest='no_event_polarity', action='store_true', \
            default=False, help='Don not divide event voxel by polarity')
    parser.add_argument('--isbi', action='store_true', default=False, help='bidirection flow training')
    parser.add_argument("--model", type=str, default="DCEIFlow", help="")
    parser.add_argument("--mixed_precision", action='store_true', default=False, help="")
    parser.add_argument("--metric", type=str, nargs='+', default=["epe"], help="")
    parser.add_argument('--gpus', type=int, nargs='+', default=[-1])
    
    args = parser.parse_args()

    if args.gpus[0] == -1:
        args.gpus = [i for i in range(torch.cuda.device_count())]
    args.nprocs = len(args.gpus)
    print(args)
    # Run Test Script
    test(args)

# 运行命令
# python main.py --path ./data/DSEC --type standard --visualize True


import argparse
import os
import sys
import subprocess

import numpy as np
import torch
import json

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from datasets.captioning_dataset import ActivityNetCaptionsDataset
# from datasets.load_features import load_features_from_npy
from datasets.load_features import crop_a_segment, pad_segment
from epoch_loops.captioning_epoch_loops import make_masks
from model.captioning_module import BiModalTransformer, Transformer
from model.proposal_generator import MultimodalProposalGenerator
from epoch_loops.captioning_epoch_loops import (greedy_decoder, inference_validation)
from utilities.proposal_utils import (get_corner_coords,
                                      remove_very_short_segments,
                                      select_topk_predictions, trim_proposals, non_max_suppresion)

from typing import Dict, List, Union
from torch.utils.data import DataLoader

class Config(object):
    # I need this to keep the name defined to load the config objects from model checkpoints.
    def __init__(self, to_log=True):
        pass

def load_features_from_npy(
        feature_paths: Dict[str, str], start: float, end: float, duration: float, pad_idx: int,
        device: int, get_full_feat=False, pad_feats_up_to: Dict[str, int] = None
    ) -> Dict[str, torch.Tensor]:
    '''Loads the pre-extracted features from numpy files.
    This function is conceptually close to `datasets.load_feature.load_features_from_npy` but cleaned up
    for demonstration purpose.

    Args:
        feature_paths (Dict[str, str]): Paths to the numpy files (keys: 'audio', 'rgb', 'flow).
        start (float, None): Start point (in secs) of a proposal, if used for captioning the proposals.
        end (float, None): Ending point (in secs) of a proposal, if used for captioning the proposals.
        duration (float): Duration of the original video in seconds.
        pad_idx (int): The index of the padding token in the training vocabulary.
        device (int): GPU id.
        get_full_feat (bool, optional): Whether to output full, untrimmed, feature stacks. Defaults to False.
        pad_feats_up_to (Dict[str, int], optional): If get_full_feat, pad to this value. Different for audio
                                                    and video modalities. Defaults to None.

    Returns:
        Dict[str, torch.Tensor]: A dict holding 'audio', 'rgb' and 'flow' features.
    '''
    stack_rgb = np.load(feature_paths['rgb'])
    stack_flow = np.load(feature_paths['flow'])

    stack_rgb = torch.from_numpy(stack_rgb).float()
    stack_flow = torch.from_numpy(stack_flow).float()
    
    if get_full_feat:        
        stack_rgb = pad_segment(stack_rgb, pad_feats_up_to['video'], pad_idx)
        stack_flow = pad_segment(stack_flow, pad_feats_up_to['video'], pad_idx=0)    
    else:        
        stack_rgb = crop_a_segment(stack_rgb, start, end, duration)
        stack_flow = crop_a_segment(stack_flow, start, end, duration)
        
    stack_rgb = stack_rgb.to(torch.device(device)).unsqueeze(0)
    stack_flow = stack_flow.to(torch.device(device)).unsqueeze(0)

    return {'rgb': stack_rgb,'flow': stack_flow}

def load_cap_model(pretrained_cap_model_path: str, dataset_path: str, device: int) -> tuple:
    '''Loads captioning model along with the Config used to train it and initiates training dataset
       to build the vocabulary including special tokens.

    Args:
        pretrained_cap_model_path (str): path to pre-trained captioning model.
        device (int): GPU id.

    Returns:
        Config, torch.nn.Module, torch.utils.data.dataset.Dataset: config, captioning module, train dataset.
    '''
    # load and patch the config for user-defined arguments
    cap_model_cpt = torch.load(pretrained_cap_model_path, map_location='cpu')
    cfg = cap_model_cpt['config']
    cfg.device = device
    cfg.pretrained_cap_model_path = pretrained_cap_model_path
    cfg.train_meta_path = dataset_path
    # load train dataset just for special token's indices
    train_dataset = ActivityNetCaptionsDataset(cfg, 'train', get_full_feat=False)

    # define model and load the weights
    print("DEVICE", device)
    model = Transformer(train_dataset, cfg)
    model = torch.nn.DataParallel(model, [device])
    model.load_state_dict(cap_model_cpt['model_state_dict'])  # if IncompatibleKeys - ignore
    model.eval()

    return cfg, model, train_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='One video prediction')
    parser.add_argument('--pretrained_cap_model_path', required=True)
    parser.add_argument('--dataset', required=True)            
    parser.add_argument('--val_1_meta_path', required=True)            
    parser.add_argument('--reference_path', required=True)            
    parser.add_argument('--modality', required=True)            
    #parser.add_argument('--inference_batch_size', required=True)            
    parser.add_argument('--device_id', type=int, default=0)        
    parser.add_argument('--video_feature_name', type=str, default='i3d')
    parser.add_argument('--video_features_path', type=str, default='')
    parser.add_argument('--B', type=int, default=32, help='batch size per device')
    args = parser.parse_args()

    print("!!!!Arguments:", args)

    # Loading models and other essential stuff
    cfg, model, train_dataset = load_cap_model(args.pretrained_cap_model_path, args.dataset, args.device_id)

    cfg.val_1_meta_path = args.val_1_meta_path
    cfg.B = args.B
    cfg.modality = args.modality
    cfg.reference_paths = [args.reference_path]
    #cfg.log_path = "./inference/"

    print(cfg.device, cfg.device_ids)

    val_1_dataset = ActivityNetCaptionsDataset(cfg, 'val_1', get_full_feat=False)
    print("Val 1 dataset length:", len(val_1_dataset))
    print("Val 1 dataset phase:", val_1_dataset.phase)
    val_1_loader = DataLoader(val_1_dataset, collate_fn=val_1_dataset.dont_collate)

    val_1_metrics = inference_validation(
        cfg, model, val_1_loader, greedy_decoder, 0
    )

    print("Validation 1 Results:")
    experiment_values = {}
    for k, v in val_1_metrics.items():
        for kk, vv in v.items():
            value = vv*100
            value = round(value, 4)
            #print(f"{k} {kk}: {value}")
            if k == 'Average across tIoUs':
                experiment_values[kk] = value
    print("Experiment Values:", experiment_values)
    
    test_file = cfg.val_1_meta_path.split('/')[-1].split('_')[0]
    save_filename = f'inference_results_{test_file}.json'
    expirement_name = cfg.log_path.split('/')[-1]    
    submission_path = os.path.join('./inference', expirement_name, save_filename)

    with open(submission_path, 'w') as outf:
        json.dump(experiment_values, outf)
    
        



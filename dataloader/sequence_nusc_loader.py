"""
dataloader of NuScenes dataset
Obtain the observation information(detection) of each frame iteratively
--------ATTENTION: Detector files must be in chronological order-------
"""

import pdb
import numpy as np
from utils.io import load_file
from data.script.NUSC_CONSTANT import *
from pre_processing import dictdet2array, arraydet2box, blend_nms
from nuscenes.nuscenes import NuScenes
from data.script.NUSC_CONSTANT import NUSCENES_TO_POLYMOT_CATEGORY

class nuScenes_sequence_loader:
    def __init__(self, config, dataroot, dataset_version='v1.0-mini',  sequence_index=0):
        """
        :param config: dict, hyperparameter setting
        """

        def get_scene_annotations_nuscenes(scene_index, nusc):
            my_scene = nusc.scene[scene_index]
            first_sample_token = my_scene['first_sample_token']

            current_sample_token = first_sample_token
            result_dict = {}
            scene_sample_tokens = []

            while True:
                scene_sample_tokens.append(current_sample_token)
                current_sample = nusc.get('sample', current_sample_token)
                annotations = current_sample['anns']

                sample_annotations = []
                for annotation_token in annotations:
                    annotation_metadata = nusc.get('sample_annotation', annotation_token)
                    if annotation_metadata['category_name'] not in NUSCENES_TO_POLYMOT_CATEGORY:
                        continue
                    # annotation_metadata["tracking_id"] = 1
                    annotation_metadata["detection_score"] = 1.0
                    annotation_metadata["velocity"] = [0.0, 0.0]
                    annotation_metadata["detection_name"] = NUSCENES_TO_POLYMOT_CATEGORY[annotation_metadata["category_name"]]
                    sample_annotations.append(annotation_metadata)
                result_dict[current_sample_token] = sample_annotations
                
                next_sample_token = current_sample['next']
                if next_sample_token == '':
                    break
                current_sample_token = next_sample_token

            return result_dict, scene_sample_tokens
        
        nusc = NuScenes(version=dataset_version, dataroot=dataroot + dataset_version, verbose=True)
        print("Using NuScenes dataset version: ", dataset_version)

        if sequence_index < 0 or sequence_index >= len(nusc.scene):
            raise ValueError(f"sequence_index {sequence_index} is out of range")

        self.result_dict, self.sequence_samples_token = get_scene_annotations_nuscenes(sequence_index, nusc)
        del nusc
        
        self.config, self.data_info = config, {}
        self.SF_thre, self.NMS_thre = config['preprocessing']['SF_thre'], config['preprocessing']['NMS_thre']
        self.NMS_type, self.NMS_metric = config['preprocessing']['NMS_type'], config['preprocessing']['NMS_metric']
        self.seq_id = self.frame_id = 0

    def __getitem__(self, item) -> dict:
        """
        data_info(dict): {
            'is_first_frame': bool
            'timestamp': int
            'sample_token': str
            'seq_id': int
            'frame_id': int
            'has_velo': bool
            'np_dets': np.array, [det_num, 14](x, y, z, w, l, h, vx, vy, ry(orientation, 1x4), det_score, class_label)
            'np_dets_bottom_corners': np.array, [det_num, 4, 2]
            'box_dets': np.array[NuscBox], [det_num]
            'no_dets': bool, corner case,
            'det_num': int,
        }
        """
        # curr_token = self.all_sample_token[item]
        curr_token = self.sequence_samples_token[item]
        ori_dets = self.result_dict[curr_token]

        # assign seq and frame id
        # if curr_token in self.seq_first_token:
        if item == 0:
            self.seq_id += 1
            self.frame_id = 1
        else: self.frame_id += 1

        # all categories are blended together and sorted by detection score
        list_dets, np_dets = dictdet2array(ori_dets, 'translation', 'size', 'velocity', 'rotation',
                                           'detection_score', 'detection_name')

        # Score Filter based on category-specific thresholds
        np_dets = np.array([det for det in list_dets if det[-2] > self.SF_thre[det[-1]]])

        # NMS, "blend" ref to blend all categories together during NMS
        if len(np_dets) != 0:
            box_dets, np_dets_bottom_corners = arraydet2box(np_dets)
            assert len(np_dets) == len(box_dets) == len(np_dets_bottom_corners)
            tmp_infos = {'np_dets': np_dets, 'np_dets_bottom_corners': np_dets_bottom_corners}
            keep = globals()[self.NMS_type](box_infos=tmp_infos, metrics=self.NMS_metric, thre=self.NMS_thre)
            keep_num = len(keep)
        # corner case, no det left
        else: keep = keep_num = 0

        print(f"\n Total {len(list_dets) - keep_num} bboxes are filtered; "
              f"{len(list_dets) - len(np_dets)} during SF, "
              f"{len(np_dets) - keep_num} during NMS, "
              f"Still {keep_num} bboxes left. "
              f"seq id {self.seq_id}, frame id {self.frame_id}, "
              f"Total frame id {item + 1}.")

        # Available information for the current frame
        data_info = {
            'is_first_frame': item == 0,
            'timestamp': item,
            'sample_token': curr_token,
            'seq_id': self.seq_id,
            'frame_id': self.frame_id,
            # 'has_velo': self.config['basic']['has_velo'],
            'has_velo': False,
            'np_dets': np_dets[keep] if keep_num != 0 else np.zeros(0),
            'np_dets_bottom_corners': np_dets_bottom_corners[keep] if keep_num != 0 else np.zeros(0),
            'box_dets': box_dets[keep] if keep_num != 0 else np.zeros(0),
            'no_dets': keep_num == 0,
            'det_num': keep_num,
        }
        return data_info

    def __len__(self) -> int:
        return len(self.sequence_samples_token)

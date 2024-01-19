from tqdm import tqdm
import os
import cv2
from typing import Tuple, List
from nuscenes.nuscenes import NuScenes
import numpy as np

from nuscenes_utils import render_sample_data_custom


def scene_samples_render(scene_sample_tokens: List[str], result_dict: dict, nusc: NuScenes, images_folder: str):
    """
    Render all the sample data in a scene

    Args:
        scene_sample_tokens: list of sample tokens in a scene
        result_dict: result dictionary
        nusc: NuScenes object
        images_folder: folder to save the images

    Returns:
        None
    """
    for sample_token in tqdm(scene_sample_tokens):
        annotations_list = result_dict[sample_token]
        sample_data_token = nusc.get('sample', sample_token)['data']['CAM_FRONT']
        render_sample_data_custom(sample_data_token, annotations_list, nusc, 
                                out_path = os.path.join(images_folder, sample_token + '.png'), verbose = False)

    img_array = []
    for filename in tqdm(scene_sample_tokens):
        img = cv2.imread(os.path.join(images_folder, filename + '.png'))
        height, width, _ = img.shape
        img_size = (width, height)
        img_array.append(img)

    return img_size, img_array 


def generate_video(img_array: List[np.array], video_path: str, fps: int = 2, size: Tuple[int, int] = (1600, 900)):
    '''
    generate video as mp4 from a list of images

    Args:
        img_array: list of images
        video_path: path to save the video

    Returns:
        None
    '''
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in tqdm(range(len(img_array))):
        out.write(img_array[i])
    out.release()

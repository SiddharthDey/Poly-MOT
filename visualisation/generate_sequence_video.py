from nuscenes.nuscenes import NuScenes
import json
import os
import argparse
import yaml

from vis_utils import scene_samples_render, generate_video


def get_scene_annotations_nuscenes(scene_index):
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
            annotation_metadata["tracking_id"] = 1
            sample_annotations.append(annotation_metadata)
        result_dict[current_sample_token] = sample_annotations
        
        next_sample_token = current_sample['next']
        if next_sample_token == '':
            break
        current_sample_token = next_sample_token

    return result_dict, scene_sample_tokens

def get_scene_annotations_detector(detector_json_file):
    with open(detector_json_file) as f:
        results = json.load(f)

    scene_sample_tokens = list(results["results"].keys())

    return results["results"], scene_sample_tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='vis_config.yaml')
    args = parser.parse_args()
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.Loader)

    output_folder = os.path.join(config["results_root"], config["dataloader_type"], config["nuSences_version"], \
                                            str(config["scene_index"]))
    data_path = os.path.join(config["data_root"], config["nuSences_version"])
    if config["use_result_json"] == True:
        detector_json_file = os.path.join(output_folder, "results.json")

    nusc = NuScenes(version=config["nuSences_version"], dataroot=data_path, verbose=True)
    print("Using NuScenes dataset version: ", config["nuSences_version"])

    if config["use_result_json"] == False:
        result_dict, scene_sample_tokens = get_scene_annotations_nuscenes(config["scene_index"])
    else:
        result_dict, scene_sample_tokens = get_scene_annotations_detector(detector_json_file)
    
    if config["use_result_json"] == False:
        images_folder = os.path.join(config["results_root"], "no_tracking", config["nuSences_version"], str(config["scene_index"]), 'images')
        video_folder = os.path.join(config["results_root"], "no_tracking", config["nuSences_version"], str(config["scene_index"]), 'videos')
    else:
        images_folder = os.path.join(output_folder, 'images')
        video_folder = os.path.join(output_folder, 'videos')

    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    # render images with bounding boxes for all samples
    img_size, img_array = scene_samples_render(scene_sample_tokens, result_dict, nusc, images_folder)

    # write video in mp4
    video_path = os.path.join(video_folder, str(config["scene_index"]) + '.mp4')
    generate_video(img_array, video_path, config["FPS"], img_size)
    

    


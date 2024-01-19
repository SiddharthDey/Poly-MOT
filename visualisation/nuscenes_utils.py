from typing import Tuple, List
from pyquaternion import Quaternion
from matplotlib.axes import Axes
from typing import Tuple, List
from PIL import Image
import matplotlib.pyplot as plt

from nuscenes.utils.geometry_utils import box_in_image, BoxVisibility, view_points
import numpy as np
from nuscenes.utils.data_classes import Box
from nuscenes.nuscenes import NuScenes
from vis_constants import POLYMOT_TO_NUSCENES_CATEGORY


class NuscBox_custom(Box):
    def __init__(self, center: List[float], size: List[float], rotation: List[float], label: int = np.nan,
                 score: float = np.nan, velocity: Tuple = (np.nan, np.nan, np.nan), name: str = None,
                 token: str = None, tracking_id: int = None):
        super().__init__(center, size, rotation, label, score, velocity, name, token)
        
        self.tracking_id = tracking_id

def get_box(record) -> Box:
    """
    Instantiates a Box class from a sample annotation record.
    :param sample_annotation_token: Unique sample_annotation identifier.
    """
    if "tracking_name" in record.keys():
        category = record['tracking_name']
    else:
        category = record['category_name']

    return NuscBox_custom(record['translation'], record['size'], Quaternion(record['rotation']),
                name=category, tracking_id=record['tracking_id'])

def get_boxes(sample_data_token: str, annotations_list: List, nusc: NuScenes) -> List[Box]:
    """
    Instantiates Boxes for all annotation for a particular sample_data record. If the sample_data is a
    keyframe, this returns the annotations for that sample. But if the sample_data is an intermediate
    sample_data, a linear interpolation is applied to estimate the location of the boxes at the time the
    sample_data was captured.
    :param sample_data_token: Unique sample_data identifier
    :param selected_anntokens: list of bounding boxes
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    curr_sample_record = nusc.get('sample', sd_record['sample_token'])

    if curr_sample_record['prev'] == "" or sd_record['is_key_frame']:
        # If no previous annotations available, or if sample_data is keyframe just return the current ones.
        boxes = list(map(get_box, annotations_list))

    # NOTE: else part is not used in our case, kept for future use cases
    else:
        prev_sample_record = nusc.get('sample', curr_sample_record['prev'])

        curr_ann_recs = [nusc.get('sample_annotation', token) for token in curr_sample_record['anns']]
        prev_ann_recs = [nusc.get('sample_annotation', token) for token in prev_sample_record['anns']]

        # Maps instance tokens to prev_ann records
        prev_inst_map = {entry['instance_token']: entry for entry in prev_ann_recs}

        t0 = prev_sample_record['timestamp']
        t1 = curr_sample_record['timestamp']
        t = sd_record['timestamp']

        # There are rare situations where the timestamps in the DB are off so ensure that t0 < t < t1.
        t = max(t0, min(t1, t))

        boxes = []
        for curr_ann_rec in curr_ann_recs:

            if curr_ann_rec['instance_token'] in prev_inst_map:
                # If the annotated instance existed in the previous frame, interpolate center & orientation.
                prev_ann_rec = prev_inst_map[curr_ann_rec['instance_token']]

                # Interpolate center.
                center = [np.interp(t, [t0, t1], [c0, c1]) for c0, c1 in zip(prev_ann_rec['translation'],
                                                                                curr_ann_rec['translation'])]

                # Interpolate orientation.
                rotation = Quaternion.slerp(q0=Quaternion(prev_ann_rec['rotation']),
                                            q1=Quaternion(curr_ann_rec['rotation']),
                                            amount=(t - t0) / (t1 - t0))

                box = Box(center, curr_ann_rec['size'], rotation, name=curr_ann_rec['category_name'],
                            token=curr_ann_rec['token'])
            else:
                # If not, simply grab the current annotation.
                box = nusc.get_box(curr_ann_rec['token'])

            boxes.append(box)
            
    return boxes

def get_sample_data(sample_data_token: str,
                    annotations_list: List[str],
                    nusc: NuScenes,
                    box_vis_level: BoxVisibility = BoxVisibility.ANY,
                    use_flat_vehicle_coordinates: bool = False) -> \
            Tuple[str, List[Box], np.array]:
        """
        Returns the data path as well as all annotations related to that sample_data.
        Note that the boxes are transformed into the current sensor's coordinate frame.
        :param sample_data_token: Sample_data token.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param selected_anntokens: If provided only return the selected annotation.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                             aligned to z-plane in the world.
        :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
        """

        # Retrieve sensor & pose records
        sd_record = nusc.get('sample_data', sample_data_token)
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        sensor_record = nusc.get('sensor', cs_record['sensor_token'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        data_path = nusc.get_sample_data_path(sample_data_token)

        if sensor_record['modality'] == 'camera':
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            imsize = (sd_record['width'], sd_record['height'])
        else:
            cam_intrinsic = None
            imsize = None

        # Retrieve all sample annotations and map to sensor coordinate system.
        boxes = get_boxes(sample_data_token, annotations_list, nusc)

        # Make list of Box objects including coord system transforms.
        box_list = []
        for box in boxes:
            if use_flat_vehicle_coordinates:
                # Move box to ego vehicle coord system parallel to world z plane.
                yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
            else:
                # Move box to ego vehicle coord system.
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(pose_record['rotation']).inverse)

                #  Move box to sensor coord system.
                box.translate(-np.array(cs_record['translation']))
                box.rotate(Quaternion(cs_record['rotation']).inverse)

            if sensor_record['modality'] == 'camera' and not \
                    box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
                continue

            box_list.append(box)

        return data_path, box_list, cam_intrinsic

def render_sample_data_custom(sample_data_token: str,
                       annotations_list: List,
                       nusc: NuScenes,
                       with_anns: bool = True,
                       box_vis_level: BoxVisibility = BoxVisibility.ANY,
                       ax: Axes = None,
                       out_path: str = None,
                       verbose: bool = True) -> None:

    # Get sensor modality.
    sd_record = nusc.get('sample_data', sample_data_token)
    sensor_modality = sd_record['sensor_modality']

    
    if sensor_modality == 'camera':
        # Load boxes and image.
        data_path, boxes, camera_intrinsic = get_sample_data(sample_data_token, annotations_list, nusc,
                                                                            box_vis_level=box_vis_level)
        data = Image.open(data_path)

        # Init axes.
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 16))

        # Show image.
        ax.imshow(data)

        # Show boxes.
        if with_anns:
            for box in boxes:
                c = np.array(nusc.explorer.get_color(POLYMOT_TO_NUSCENES_CATEGORY[box.name])) / 255.0
                # c = np.array((255, 158, 0))/255.0
                box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))

                # add tracking id on top left corner of bounding box
                if box.tracking_id is not None:
                    corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
                    top_left = corners.T[0]
                    label = 'ID: %d' % int(box.tracking_id)
                    if top_left[0] < 0 or top_left[1] < 0 or top_left[0] > data.size[0] or top_left[1] > data.size[1]:
                        continue
                    ax.text(top_left[0], top_left[1]-15, label, fontsize=12, color=c)

        # Limit visible range.
        ax.set_xlim(0, data.size[0])
        ax.set_ylim(data.size[1], 0)

    else:
        raise ValueError("Error: Unknown sensor modality!")

    ax.axis('off')
    ax.set_title(sd_record['channel'])
    ax.set_aspect('equal')

    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
    
    if verbose == False:
        plt.close()

    if verbose == True:
        plt.show()
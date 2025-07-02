import os
import numpy as np
from PIL import Image, ImageDraw
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

# CONFIG
NUSCENES_DIR = '/data_pool/data/nuscenes'       # CHANGE THIS
SCENE_NAME = 'scene-0061'                # CHANGE THIS TO TARGET SCENE
OUTPUT_DIR = '/home/q681861/Documents/test_scripts/output_dynamic_masks' # Where to save results

# Dynamic categories in nuScenes
DYNAMIC_CLASSES = [
    'vehicle.car', 'vehicle.truck', 'vehicle.bus', 'vehicle.motorcycle',
    'vehicle.bicycle', 'vehicle.trailer', 'vehicle.construction',
    'human.pedestrian.adult', 'human.pedestrian.child',
    'human.pedestrian.stroller', 'human.pedestrian.wheelchair'
]

# Initialize NuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot=NUSCENES_DIR, verbose=True)

# Get scene
scene = next(s for s in nusc.scene if s['name'] == SCENE_NAME)
scene_token = scene['token']
sample_token = scene['first_sample_token']

# Loop through all frames in the scene
while sample_token:
    sample = nusc.get('sample', sample_token)
    cam_token = sample['data']['CAM_FRONT']
    cam_data = nusc.get('sample_data', cam_token)
    image_path = os.path.join(NUSCENES_DIR, cam_data['filename'])
    calibrated_sensor = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    cam_translation = np.array(calibrated_sensor['translation'])
    cam_rotation = Quaternion(calibrated_sensor['rotation'])
    intrinsic = np.array(calibrated_sensor['camera_intrinsic'])

    # Load image
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    mask = Image.new("L", (width, height), 0)
    draw_mask = ImageDraw.Draw(mask)

    # Debug overlay
    image_debug = image.copy()
    draw_debug = ImageDraw.Draw(image_debug)

    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        if ann['category_name'] not in DYNAMIC_CLASSES:
            continue

        box = nusc.get_box(ann_token)
        box.translate(-cam_translation)
        box.rotate(cam_rotation.inverse)

        corners_3d = box.corners()
        corners_2d = view_points(corners_3d, intrinsic, normalize=True)[:2, :]

        x_coords = corners_2d[0]
        y_coords = corners_2d[1]

        if np.any(np.isnan(x_coords)) or np.any(np.isnan(y_coords)):
            continue

        xmin = int(np.clip(np.min(x_coords), 0, width - 1))
        xmax = int(np.clip(np.max(x_coords), 0, width - 1))
        ymin = int(np.clip(np.min(y_coords), 0, height - 1))
        ymax = int(np.clip(np.max(y_coords), 0, height - 1))

        if xmin >= xmax or ymin >= ymax:
            continue

        draw_mask.rectangle([xmin, ymin, xmax, ymax], fill=255)
        draw_debug.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)

    # Prepare output paths
    scene_output_dir = os.path.join(OUTPUT_DIR, SCENE_NAME)
    os.makedirs(scene_output_dir, exist_ok=True)
    timestamp = cam_data['timestamp']
    base_name = f"{timestamp}_CAM_FRONT"

    image.save(os.path.join(scene_output_dir, f"{base_name}_image.jpg"))
    mask.save(os.path.join(scene_output_dir, f"{base_name}_mask.png"))
    image_debug.save(os.path.join(scene_output_dir, f"{base_name}_debug.jpg"))

    print(f"Saved frame: {base_name}")

    sample_token = sample['next']

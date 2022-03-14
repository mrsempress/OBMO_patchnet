import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOOL_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.dirname(TOOL_DIR)
sys.path.append(ROOT_DIR)
import argparse
import tqdm
import pickle
import numpy as np
from PIL import Image
from lib.datasets.kitti_dataset import KittiDataset
from numba import jit

data_root = os.path.join(ROOT_DIR, 'data/KITTI/object')
parser = argparse.ArgumentParser()
parser.add_argument('--gen_train', action='store_true', help='Generate train split frustum data [from GT]')
parser.add_argument('--gen_val', action='store_true', help='Generate val split frustum data [from GT]')
parser.add_argument('--gen_val_rgb_detection', action='store_true', help='Generate frustum val split frustum data [from 2d detections]')
parser.add_argument('--gen_test_rgb_detection', action='store_true', help='Generate frustum test split frustum data [from 2d detections]')
parser.add_argument('--car_only', action='store_true', help='Only generate frustum data from car instances')
parser.add_argument('--data', default='mono', help='mono or stereo')
parser.add_argument('--vis_test', action='store_true')
parser.add_argument('--quality_score', default='linear', help='linear or 2D IoU or 3D IoU')
args = parser.parse_args()


@jit
def calc_iou(box1, box2):
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    xx1 = np.maximum(box1[0], box2[0])
    yy1 = np.maximum(box1[1], box2[1])
    xx2 = np.minimum(box1[2], box2[2])
    yy2 = np.minimum(box1[3], box2[3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    ovr = inter / (area1 + area2 - inter)

    return ovr


def random_shift_box2d(box2d, shift_ratio=0.1):
    ''' Randomly shift box center, randomly scale width and height
    '''
    r = shift_ratio
    xmin,ymin,xmax,ymax = box2d
    h = ymax-ymin
    w = xmax-xmin
    cx = (xmin+xmax)/2.0
    cy = (ymin+ymax)/2.0
    cx2 = cx + w*r*(np.random.random()*2-1)
    cy2 = cy + h*r*(np.random.random()*2-1)
    h2 = h*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    w2 = w*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    return np.array([cx2-w2/2.0, cy2-h2/2.0, cx2+w2/2.0, cy2+h2/2.0])


def extract_patch_data(split, output_filename, whitelist=['Car'], quality_score='linear'):
    ''' Extract depth patches and corresponding annotations
        defined generated from 2D bounding boxes
        (as that in 3d box label files)

    Input:
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        whitelist: a list of strings, object types we are interested in.
        quality_score: string, the quality score you choose
    Output:
        None (will write a .pickle file to the disk)
    '''
    data_dir = os.path.join(ROOT_DIR, 'data')
    dataset = KittiDataset(root_dir=data_dir, split=split)

    patch_xyz_list = []
    patch_rgb_list = []
    type_list = []
    heading_list = []
    box3d_center_list = []
    box3d_size_list = []
    frustum_angle_list = []
    quality_score_list = []

    progress_bar = tqdm.tqdm(total=len(dataset.idx_list), leave=True, desc='%s split patch data gen' % split)
    for data_idx in dataset.idx_list: # image idx
        data_idx = int(data_idx)
        calib = dataset.get_calib(data_idx)
        objects = dataset.get_label(data_idx)

        # compute x,y,z for each pixel in depth map
        depth = dataset.get_depth(data_idx)
        image = dataset.get_image(data_idx)
        assert depth.size == image.size
        width, height = depth.size
        depth = np.array(depth).astype(np.float32) / 256
        uvdepth = np.zeros((height, width, 3), dtype=np.float32)
        for v in range(height):
            for u in range(width):
                uvdepth[v, u, 0] = u
                uvdepth[v, u, 1] = v
        uvdepth[:, :, 2] = depth
        uvdepth = uvdepth.reshape(-1, 3)
        xyz = calib.img_to_rect(uvdepth[:, 0], uvdepth[:, 1], uvdepth[:, 2])  # rect coord sys
        xyz = xyz.reshape(height, width, 3)  # record xyz, data type: float32
        # uvdepth = uvdepth.reshape(height, width, 3)
        rgb = np.array(image)

        for object in objects:
            if object.cls_type not in whitelist:
                continue

            # get 2d box from ground truth
            box2d = object.box2d
            xmin, ymin, xmax, ymax = box2d

            # filter
            if object.level > 3: continue

            centerx, centery = (xmin + xmax) / 2, (ymin + ymax) / 2
            h = ymax - ymin
            w = xmax - xmin
            centerx3 = xyz[(int)(centery), (int)(centerx), 0]
            centery3 = xyz[(int)(centery), (int)(centerx), 1] 
            centerz3 = xyz[(int)(centery), (int)(centerx), 2]
            # compute the X-Z ratio and Y-Z ratio
            x_z_ratio = centerx3 / centerz3
            y_z_ratio = centery3 / centerz3

            if split != 'train':
                epsilon = [0.0]
            else:
                epsilon = [-0.08, -0.04, 0.0, +0.04, +0.08]
            
            for e in epsilon:
                # perturb the depth
                deltaz = e * centerz3
                centerz3_p = centerz3 + deltaz
                centerx3_p = centerz3_p * x_z_ratio
                centery3_p = centerz3_p * y_z_ratio

                box2d_center, centerz_img = calib.rect_to_img(np.array([centerx3_p, centery3_p, centerz3_p]).reshape(1, 3))
                centerx_p, centery_p = box2d_center[0, 0], box2d_center[0, 1]
                xmin_p = centerx_p - w / 2
                xmax_p = centerx_p + w / 2
                ymin_p = centery_p - h / 2
                ymax_p = centery_p + h / 2

                xmin_p, ymin_p = max(xmin_p, 0), max(ymin_p, 0)   # check range
                xmax_p, ymax_p = min(xmax_p, width), min(ymax_p, height)  # check range

                patch_xyz = xyz[int(ymin_p):int(ymax_p), int(xmin_p):int(xmax_p), :]
                patch_rgb = rgb[int(ymin):int(ymax), int(xmin):int(xmax), :]

                # Get frustum angle (according to center pixel in 2D BOX)
                uvdepth = np.zeros((1, 3))
                uvdepth[0, 0:2] = box2d_center[0]
                uvdepth[0, 2] = 20  # some random depth
                box2d_center_rect = calib.img_to_rect(uvdepth[:, 0], uvdepth[:, 1], uvdepth[:, 2])
                frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2], box2d_center_rect[0, 0])

                patch_xyz_list.append(patch_xyz)
                patch_rgb_list.append(patch_rgb)
                type_list.append(object.cls_type)
                heading_list.append(object.ry)
                box3d_center_list.append((object.pos - [0.0, object.h/2, 0.0]).astype(np.float32))
                box3d_size = np.array([object.l, object.w, object.h]).astype(np.float32)
                box3d_size_list.append(box3d_size)
                frustum_angle_list.append(frustum_angle) #
                
                if quality_score == 'linear':
                    quality_score_list.append((1 - abs(deltaz) / 4) * object.score)
                elif quality_score == '3DIoU':
                    obser = abs(object.ry - np.arctan2(centerx3_p, centerz3_p))
                    if obser > np.pi/2:
                        obser = abs(obser - np.pi)
                    overlap = (object.w - deltaz*np.sin(obser)) * (object.l - deltaz*np.cos(obser))
                    quality_score_list.append(overlap / (2 * object.w * object.l - overlap) * object.score)
                elif quality_score == '2DIoU':
                    IoU = calc_iou(np.array([xmin, ymin, xmax, ymax]), np.array([xmin_p, ymin_p, xmax_p, ymax_p]))
                    quality_score_list.append(IoU * object.score)
                
        progress_bar.update()
    progress_bar.close()

    with open(output_filename,'wb') as fp:
        pickle.dump(patch_xyz_list, fp)
        pickle.dump(patch_rgb_list, fp)
        pickle.dump(type_list, fp)
        pickle.dump(heading_list, fp)
        pickle.dump(box3d_center_list, fp)
        pickle.dump(box3d_size_list, fp)
        pickle.dump(frustum_angle_list, fp)
        pickle.dump(quality_score_list, fp)


def read_det_file(det_filename):
    ''' Parse lines in 2D detection output files '''
    det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
    id_list = []
    type_list = []
    prob_list = []
    box2d_list = []
    for line in open(det_filename, 'r'):
        t = line.rstrip().split(' ')
        id_list.append(int(os.path.basename(t[0]).rstrip('.png')))
        type_list.append(det_id2str[int(t[1])])
        prob_list.append(float(t[2]))
        box2d_list.append(np.array([float(t[i]) for i in range(3, 7)]))
    return id_list, type_list, box2d_list, prob_list


def extract_patch_data_rgb_detection(det_filename, split, output_filename,
                                       whitelist=['Car'],
                                       img_height_threshold=25):
    ''' Extract point clouds in frustums extruded from 2D detection boxes.
        Update: Lidar points and 3d boxes are in *rect camera* coord system
            (as that in 3d box label files)

    Input:
        det_filename: string, each line is
            img_path typeid confidence xmin ymin xmax ymax
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        whitelist: a list of strings, object types we are interested in.
        img_height_threshold: int, neglect image with height lower than that.
        lidar_point_threshold: int, neglect frustum with too few points.
    Output:
        None (will write a .pickle file to the disk)
    '''
    data_dir = os.path.join(ROOT_DIR, 'data')
    dataset = KittiDataset(root_dir=data_dir, split=split)
    det_id_list, det_type_list, det_box2d_list, det_prob_list = read_det_file(det_filename)
    cache_id = -1
    cache = None

    id_list = []
    type_list = []
    box2d_list = []
    prob_list = []
    patch_xyz_list = []
    patch_rgb_list = []
    frustum_angle_list = []

    progress_bar = tqdm.tqdm(total=len(det_id_list), leave=True, desc='%s split patch data gen (from 2d detections)' % split)
    for det_idx in range(len(det_id_list)):
        data_idx = det_id_list[det_idx]
        if cache_id != data_idx:
            calib = dataset.get_calib(data_idx)

            # compute x,y,z for each pixel in depth map
            depth = dataset.get_depth(data_idx)
            image = dataset.get_image(data_idx)
            assert depth.size == image.size
            width, height = depth.size
            depth = np.array(depth).astype(np.float32) / 256
            uvdepth = np.zeros((height, width, 3), dtype=np.float32)
            for v in range(height):
                for u in range(width):
                    uvdepth[v, u, 0] = u
                    uvdepth[v, u, 1] = v
            uvdepth[:, :, 2] = depth
            uvdepth = uvdepth.reshape(-1, 3)
            xyz = calib.img_to_rect(uvdepth[:, 0], uvdepth[:, 1], uvdepth[:, 2])  # rect coord sys
            xyz = xyz.reshape(height, width, 3)  # record xyz, data type: float32
            rgb = np.array(image)

            cache = [xyz, rgb]
            cache_id = data_idx
        else:
            xyz, rgb = cache   # xyz map for whole image

        if det_type_list[det_idx] not in whitelist:
            progress_bar.update()
            continue

        # 2D BOX: Get pts rect backprojected
        xmin, ymin, xmax, ymax = det_box2d_list[det_idx]

        # Get frustum angle (according to center pixel in 2D BOX)
        box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
        uvdepth = np.zeros((1, 3))
        uvdepth[0, 0:2] = box2d_center
        uvdepth[0, 2] = 20  # some random depth
        box2d_center_rect = calib.img_to_rect(uvdepth[:, 0], uvdepth[:, 1], uvdepth[:, 2])
        frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2], box2d_center_rect[0, 0])

        # Pass objects that are too small
        if ymax - ymin < img_height_threshold:
            progress_bar.update()
            continue

        height, width, _ = xyz.shape
        xmin, ymin = max(xmin, 0), max(ymin, 0)  # check range
        xmax, ymax = min(xmax, width), min(ymax, height)  # check range
        patch_xyz = xyz[int(ymin):int(ymax), int(xmin):int(xmax), :]
        patch_rgb = rgb[int(ymin):int(ymax), int(xmin):int(xmax), :]

        id_list.append(data_idx)
        box2d_list.append(det_box2d_list[det_idx])
        patch_xyz_list.append(patch_xyz)
        patch_rgb_list.append(patch_rgb)
        type_list.append(det_type_list[det_idx])
        frustum_angle_list.append(frustum_angle)
        prob_list.append(det_prob_list[det_idx])

        progress_bar.update()
    progress_bar.close()

    with open(output_filename, 'wb') as fp:
        pickle.dump(id_list, fp)
        pickle.dump(box2d_list, fp)
        pickle.dump(patch_xyz_list, fp)
        pickle.dump(patch_rgb_list, fp)
        pickle.dump(type_list, fp)
        pickle.dump(frustum_angle_list, fp)
        pickle.dump(prob_list, fp)


def vis_demo():
    test_id = 4
    with open('../../data/KITTI/pickle_files/obmo_carpedcyc_val.pickle', 'rb') as fp:
        patch_xyz_list = pickle.load(fp)
        patch_rgb_list = pickle.load(fp)
        type_list = pickle.load(fp)
        heading_list = pickle.load(fp)
        box3d_center_list = pickle.load(fp)
        box3d_size_list = pickle.load(fp)

        print(patch_xyz_list[test_id].shape)
        print(patch_rgb_list[test_id].shape)
        print(type_list[test_id])
        print(heading_list[test_id])
        print(box3d_center_list[test_id])
        print(box3d_size_list[test_id])

        # show patch rgb by image
        img = Image.fromarray(patch_rgb_list[test_id])
        img.show()

        # show patch xyz by pesudo_lidar
        import mayavi.mlab as mlab
        fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500))
        pesudo_lidar = patch_xyz_list[test_id].reshape(-1, 3)
        color = pesudo_lidar[:, 2]
        mlab.points3d(pesudo_lidar[:, 0], pesudo_lidar[:, 1], pesudo_lidar[:, 2], color, color=None,
                      mode='point', colormap='gnuplot', scale_factor=1, figure=fig)

        # draw origin
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)

        # draw axis
        axes = np.array([
            [2., 0., 0., 0.],
            [0., 2., 0., 0.],
            [0., 0., 2., 0.],
        ], dtype=np.float64)
        mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig)

        # draw fov
        fov = np.array([  # 45 degree
            [20., 20., 0., 0.],
            [20., -20., 0., 0.],
        ], dtype=np.float64)

        input()





if __name__ == '__main__':
    # setting
    whitelist = ['Car'] if args.car_only else ['Car', 'Pedestrian', 'Cyclist']
    output_prefix = 'obmo_caronly_' if args.car_only else 'obmo_carpedcyc_'

    train_filename = output_prefix + 'train.pickle'
    val_filename = output_prefix + 'val.pickle'
    val_detections_filename = output_prefix + 'val_rgb_detection.pickle'
    rgb_detections = os.path.join(ROOT_DIR, 'data/KITTI/2d_detections/fpointnet/rgb_detection_val.txt')
    test_detections_filename = output_prefix + 'test_rgb_detection.pickle'
    test_rgb_detections = os.path.join(ROOT_DIR, 'data/KITTI/2d_detections/fpointnet/rgb_detection_test.txt')

    if args.gen_train:
        extract_patch_data(split = 'train',
                           output_filename = train_filename,
                           whitelist = whitelist)
    if args.gen_val:
        extract_patch_data(split = 'val',
                           output_filename = val_filename,
                           whitelist = whitelist)

    if args.gen_val_rgb_detection:
        extract_patch_data_rgb_detection(det_filename = rgb_detections,
                                           split = 'val',
                                           output_filename = val_detections_filename,
                                           whitelist = whitelist)
    if args.gen_test_rgb_detection:
        extract_patch_data_rgb_detection(det_filename = test_rgb_detections,
                                           split = 'test',
                                           output_filename = test_detections_filename,
                                           whitelist = whitelist)

    if args.vis_test:
        vis_demo()





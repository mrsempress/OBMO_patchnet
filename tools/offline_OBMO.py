import os
import numpy as np
import math
from tqdm import tqdm
from numba import jit
import argparse


def project_3d(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=False):
    """
    Projects a 3D box into 2D vertices

    Args:
        p2 (nparray): projection matrix of size 4x3
        x3d: x-coordinate of center of object
        y3d: y-coordinate of center of object
        z3d: z-cordinate of center of object
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: rotation w.r.t y-axis
    """

    # compute rotational matrix around yaw axis
    R = np.array([[+math.cos(ry3d), 0, +math.sin(ry3d)],
                  [0, 1, 0],
                  [-math.sin(ry3d), 0, +math.cos(ry3d)]])

    # 3D bounding box corners
    x_corners = np.array([0, l3d, l3d, l3d, l3d, 0, 0, 0])
    y_corners = np.array([0, 0, h3d, h3d, 0, 0, h3d, h3d])
    z_corners = np.array([0, 0, 0, w3d, w3d, w3d, w3d, 0])

    x_corners += -l3d / 2
    y_corners += -h3d / 2
    z_corners += -w3d / 2

    # bounding box in object co-ordinate
    corners_3d = np.array([x_corners, y_corners, z_corners])

    # rotate
    corners_3d = R.dot(corners_3d)

    # translate
    corners_3d += np.array([x3d, y3d, z3d]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))
    corners_2D = p2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]

    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]

    verts3d = (corners_2D[:, bb3d_lines_verts_idx][:2]).astype(float).T

    if return_3d:
        return verts3d, corners_3d
    else:
        return verts3d

@jit
def calc_iou(box1, box2):
    # area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    # area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    xx1 = np.maximum(box1[0], box2[0])
    yy1 = np.maximum(box1[1], box2[1])
    xx2 = np.minimum(box1[2], box2[2])
    yy2 = np.minimum(box1[3], box2[3])

    # w = np.maximum(0.0, xx2 - xx1 + 1)
    # h = np.maximum(0.0, yy2 - yy1 + 1)
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    ovr = inter / (area1 + area2 - inter)

    return ovr

def compute_iou_proj_with_2d(obj, P2):
    # obj: trucation(0), occlusion(1), alpha(2), bbox2d(3-6), h(7), w(8), l(9), pos(10-12), ry(13), score(14)
    bbox2d = obj[3:7]

    corner_proj = project_3d(P2, obj[10], obj[11] - obj[7] / 2, obj[12], obj[8], obj[7], obj[9], obj[13])
    bbox_proj = np.array([np.min(corner_proj[:, 0]), np.min(corner_proj[:, 1]), np.max(corner_proj[:, 0]), np.max(corner_proj[:, 1])])
    
    return calc_iou(bbox2d, bbox_proj)

def OBMO_offline(pred_path, calib_path, epsilons=[0.04, 0.02, 0.01], change_score=False):
    all_files = sorted(os.listdir(os.path.join(pred_path, 'data')))
    save = os.path.join(pred_path, 'data_OBMO_offline')
    if not os.path.exists(save):
        os.makedirs(save)
    
    for file_name in tqdm(all_files):
        with open(os.path.join(calib_path, file_name), encoding='utf-8') as f:
            text = f.readlines()
            P2 = np.array(text[2].split(' ')[1:], dtype=np.float32).reshape(3, 4)

        pred_list = np.loadtxt(os.path.join(pred_path, 'data', file_name), dtype=str).reshape(-1, 16)

        new_pred_list = []
        for p in pred_list:
            # obj: cls, trucation(0), occlusion(1), alpha(2), bbox2d(3-6), h(7), w(8), l(9), pos(10-12), ry(13), score(14)
            _epsilons = np.multiply(p[13].astype(np.float32), epsilons).tolist()
            max_iou = compute_iou_proj_with_2d(p[1:].astype(np.float32), P2)
            final_p = p[1:].copy().astype(np.float32)
            
            x_z_ratio = p[11].astype(np.float32) / p[13].astype(np.float32)
            y_z_ratio = p[12].astype(np.float32) / p[13].astype(np.float32)

            for e in _epsilons:
                new_p = p[1:].copy().astype(np.float32)
                
                # OBMO
                new_p[12] = p[13].astype(np.float32) * (1 + e)
                new_p[10] = new_p[12].astype(np.float32) * x_z_ratio
                new_p[11] = new_p[12].astype(np.float32) * y_z_ratio

                iou = compute_iou_proj_with_2d(new_p, P2)
                if iou > max_iou:
                    iou = max_iou
                    final_p = new_p
            
            if change_score:
                center_y = final_p[11].astype(np.float32) - final_p[7].astype(np.float32) / 2
                dis = np.sqrt(final_p[12].astype(np.float32) ** 2 + center_y ** 2 + final_p[10].astype(np.float32) ** 2)
                final_p[-1] = max_iou / np.exp(dis / 80) * final_p[-1].astype(np.float32)
            
            final_p = final_p.tolist()
            final_p.insert(0, p[0])
            new_pred_list.append(final_p)
        
        np.savetxt(os.path.join(save, file_name), new_pred_list, fmt='%s')
            
def configs():
    parser = argparse.ArgumentParser(description='Implementation of OBMO offline.')
    parser.add_argument('pred', help='The root of predictions.')
    parser.add_argument('calib', help='The root of calibrations.')
    parser.add_argument('--change_score', default=True, help='Whether employ the 2D-3D confidence mechanism from OCM3d.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = configs()

    epsilons = [0.04, 0.02, 0.01]
    neg_epsilons = np.multiply(-1, epsilons).tolist()
    epsilons.extend(neg_epsilons)

    OBMO_offline(args.pred, args.calib, epsilons, change_score=args.change_score)

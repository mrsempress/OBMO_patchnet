import os
import numpy as np
from scipy.spatial import ConvexHull
from lib.utils.kitti.kitti_utils import get_cls_type_mean_size, class2type, type2class, rotate_pc_along_y
import lib.utils.kitti.calibration as calibration

# ---------------------------------------------------
# Helper functions: hybird loss [size and heading]
# ---------------------------------------------------

NUM_HEADING_BIN = 12

def size2class(size, type_name):
    ''' Convert 3D bounding box size to template class and residuals.
    Input:
        size: numpy array of shape (3,) for (l,w,h)
        type_name:
        type_id:
    Output:
        size_class: int scalar
        size_residual: numpy array of shape (3,)
    '''
    size_class = type2class[type_name]
    size_residual = size - get_cls_type_mean_size(type_name)
    return size_class, size_residual


def class2size(pred_cls, residual):
    ''' Inverse function to size2class '''
    mean_size = get_cls_type_mean_size(class2type[pred_cls])
    return mean_size + residual


def angle2class(angle, num_class):
    ''' Convert continuous angle to discrete class and residual.
    Input:
        angle: rad scalar, from 0-2pi (or -pi~pi), class center at
            0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        num_class: int scalar, number of classes N
    Output:
        class_id, int, among 0,1,...,N-1
        residual_angle: float, a number such that
            class*(2pi/N) + residual_angle = angle
    '''
    angle = angle % (2 * np.pi)
    assert (angle>=0 and angle<=2*np.pi)
    angle_per_class = 2*np.pi/float(num_class)
    shifted_angle = (angle + angle_per_class/2) % (2*np.pi)
    class_id = int(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle


def class2angle(pred_cls, residual, num_class, to_label_format=True):
    ''' Inverse function to angle2class.
    If to_label_format, adjust angle to the range as in labels.
    '''
    angle_per_class = 2*np.pi/float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle>np.pi:
        angle = angle - 2*np.pi
    return angle


# ---------------------------------------------------
# Helper functions: box IoU computation
# ---------------------------------------------------

def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.
    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**
    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """
    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return (outputList)


def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c


def compute_box3d_iou(center_pred,
                      heading_logits, heading_residuals,
                      size_logits, size_residuals,
                      center_label,
                      heading_class_label, heading_residual_label,
                      size_class_label, size_residual_label):
    ''' Compute 3D bounding box IoU from network output and labels.
    All inputs are numpy arrays.
    Inputs:
        center_pred: (B,3)
        heading_logits: (B,NUM_HEADING_BIN)
        heading_residuals: (B,NUM_HEADING_BIN)
        size_logits: (B,NUM_SIZE_CLUSTER)
        size_residuals: (B,NUM_SIZE_CLUSTER,3)
        center_label: (B,3)
        heading_class_label: (B,)
        heading_residual_label: (B,)
        size_class_label: (B,)
        size_residual_label: (B,3)
    Output:
        iou2ds: (B,) birdeye view oriented 2d box ious
        iou3ds: (B,) 3d box ious
    '''
    batch_size = heading_logits.shape[0]
    heading_class = np.argmax(heading_logits, 1) # B
    heading_residual = np.array([heading_residuals[i,heading_class[i]] \
        for i in range(batch_size)]) # B,
    size_class = np.argmax(size_logits, 1) # B
    size_residual = np.vstack([size_residuals[i,size_class[i],:] \
        for i in range(batch_size)])

    iou2d_list = []
    iou3d_list = []
    for i in range(batch_size):
        heading_angle = class2angle(heading_class[i],
            heading_residual[i], NUM_HEADING_BIN)
        box_size = class2size(size_class[i], size_residual[i])
        corners_3d = get_3d_box(box_size, heading_angle, center_pred[i])

        heading_angle_label = class2angle(heading_class_label[i],
            heading_residual_label[i], NUM_HEADING_BIN)
        box_size_label = class2size(size_class_label[i], size_residual_label[i])
        corners_3d_label = get_3d_box(box_size_label,
            heading_angle_label, center_label[i])

        iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label)
        iou3d_list.append(iou_3d)
        iou2d_list.append(iou_2d)
    return np.array(iou2d_list, dtype=np.float32), \
        np.array(iou3d_list, dtype=np.float32)


def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.
    Input:
        box_size: tuple of (l,w,h)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    R = roty(heading_angle)
    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d


def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)]
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])
    inter_vol = inter_area * max(0.0, ymax-ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d


# ---------------------------------------------------
# Helper functions: evaluation [format tansfer, files filling]
# ---------------------------------------------------

def write_detection_results(result_dir, id_list, type_list, box2d_list, center_list, \
                            heading_cls_list, heading_res_list, \
                            size_cls_list, size_res_list, \
                            rot_angle_list, score_list, \
                            _2Dto3D=False, split='val'):
    ''' Write frustum pointnets results to KITTI format label files. '''
    if result_dir is None: return
    results = {}  # map from idx to list of strings, each string is a line (without \n)

    for i in range(len(center_list)):
        idx = id_list[i]
        output_str = type_list[i] + " -1 -1 -10 "
        box2d = box2d_list[i]
        output_str += "%f %f %f %f " % (box2d[0],box2d[1],box2d[2],box2d[3])
        h,w,l,tx,ty,tz,ry = from_prediction_to_label_format(center_list[i],
            heading_cls_list[i], heading_res_list[i],
            size_cls_list[i], size_res_list[i], rot_angle_list[i])
        score = score_list[i]

        if _2Dto3D:
            """Ocm3d: Object-centric monocular 3d object detection."""
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
            root_dir = os.path.join(PROJECT_DIR, 'data')
            calib_dir = os.path.join(root_dir, 'KITTI', 'object', 'testing' if split=='test' else 'training', 'calib')
            calib_file = os.path.join(calib_dir, '%06d.txt' % idx)
            assert os.path.exists(calib_file)
            calib = calibration.Calibration(calib_file)

            corner3d_proj = project_3d(calib.P2, tx, ty-h/2, tz, h, w, l, ry)
            box3d_proj = np.array([np.min(corner3d_proj[:, 0]), np.min(corner3d_proj[:, 1]),
                                    np.max(corner3d_proj[:, 0]), np.max(corner3d_proj[:, 1])])
            IoU = calc_iou(box2d, box3d_proj)
            dis = np.sqrt(tx ** 2 + (ty - h/2) ** 2 + tz ** 2)
            score = score * IoU / np.exp(dis / 80)

        output_str += "%f %f %f %f %f %f %f %f" % (h,w,l,tx,ty,tz,ry,score)
        if idx not in results: results[idx] = []
        results[idx].append(output_str)

    # Write TXT files
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%06d.txt' % (idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line + '\n')
        fout.close()



def from_prediction_to_label_format(center, angle_class, angle_res,\
                                    size_class, size_res, rot_angle):
    ''' Convert predicted box parameters to label format. '''
    l,w,h = class2size(size_class, size_res)
    ry = class2angle(angle_class, angle_res, NUM_HEADING_BIN) + rot_angle
    tx,ty,tz = rotate_pc_along_y(np.expand_dims(center,0),-rot_angle).squeeze()
    ty += h/2.0
    return h,w,l,tx,ty,tz,ry


def fill_files(output_dir, to_fill_filename_list):
    ''' Create empty files if not exist for the filelist. '''
    for filename in to_fill_filename_list:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            fout = open(filepath, 'w')
            fout.close()


from numba import jit
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
    import math
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

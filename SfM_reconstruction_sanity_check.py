import cv2
import yaml
import transformations
import numpy as np
from plyfile import PlyData
from pathlib import Path

if __name__ == '__main__':
    prefix_seq = Path("/home/xliu89/RemoteData/Sinus Project Data/xingtong/EndoscopicVideoData/bag_4/_start_000872_end_001187_stride_25_segment_10")
    #Path("/home/xliu89/RemoteData/Sinus Project Data/xingtong/EndoscopicVideoData/bag_5/_start_000727_end_001668_stride_25_segment_18")
    #Path("/home/xliu89/RemoteData/Sinus Project Data/xingtong/EndoscopicVideoData/bag_5/_start_000727_end_001668_stride_25_segment_09")
    #Path("/home/xliu89/RemoteData/Sinus Project Data/xingtong/EndoscopicVideoData/bag_4/_start_002869_end_002937_stride_25_segment_01")
    #Path("/home/xliu89/RemoteData/Sinus Project Data/xingtong/EndoscopicVideoData/bag_4/_start_002315_end_002504_stride_25_segment_01")
    #Path("/home/xliu89/RemoteData/Sinus Project Data/xingtong/EndoscopicVideoData/bag_4/_start_000872_end_001187_stride_25_segment_10")
    #Path("/home/xliu89/RemoteData/Sinus Project Data/xingtong/EndoscopicVideoData/bag_3/_start_000786_end_002094_stride_25_segment_39")
    #Path("/home/xliu89/RemoteData/Sinus Project Data/xingtong/EndoscopicVideoData/bag_3/_start_000786_end_002094_stride_25_segment_11")
    #Path("/home/xliu89/RemoteData/Sinus Project Data/xingtong/EndoscopicVideoData/bag_2/_start_001559_end_002189_stride_25_segment_00")
    #Path("/home/xliu89/RemoteData/Sinus Project Data/xingtong/EndoscopicVideoData/bag_2/_start_002450_end_002701_stride_25_segment_01")
    #Path("/home/xliu89/RemoteData/Sinus Project Data/xingtong/EndoscopicVideoData/bag_1/_start_004259_end_004629_stride_25_segment_08")
    #Path("/home/xliu89/RemoteData/Sinus Project Data/xingtong/EndoscopicVideoData/bag_1/_start_002603_end_002984_stride_25_segment_00")

    # Read selected indexes
    selected_indexes = []
    with open(str(prefix_seq / 'selected_indexes')) as fp:
        for line in fp:
            selected_indexes.append(int(line))

    # Read sparse point cloud from SfM
    lists_3D_points = []
    plydata = PlyData.read(str(prefix_seq / 'structure.ply'))
    for i in range(plydata['vertex'].count):
        temp = list(plydata['vertex'][i])
        temp = temp[:3]
        temp.append(1.0)
        lists_3D_points.append(temp)

    # Read camera poses from SfM
    stream = open(str(prefix_seq / "motion.yaml"), 'r')
    doc = yaml.load(stream)
    keys, values = doc.items()
    poses = values[1]

    # Read indexes of visible views
    visible_view_indexes = []
    with open(str(prefix_seq / 'visible_view_indexes')) as fp:
        for line in fp:
            visible_view_indexes.append(int(line))

    # Read view indexes per point
    view_indexes_per_point = np.zeros((plydata['vertex'].count, len(visible_view_indexes)))
    point_count = -1
    with open(str(prefix_seq / 'view_indexes_per_point')) as fp:
        for line in fp:
            if int(line) == -1:
                point_count = point_count + 1
            else:
                view_indexes_per_point[point_count][visible_view_indexes.index(int(line))] = 1

    # Read camera intrinsics used by SfM
    camera_intrinsics = []
    param_count = 0
    temp_camera_intrincis = np.zeros((3, 4))
    with open(str(prefix_seq / 'camera_intrinsics_per_view')) as fp:
        for line in fp:
            if param_count == 0:
                temp_camera_intrincis[0][0] = float(line)
                temp_camera_intrincis[1][1] = float(line)
                param_count = 1
            elif param_count == 1:
                temp_camera_intrincis[0][2] = float(line)
                param_count = 2
            elif param_count == 2:
                temp_camera_intrincis[1][2] = float(line)
                temp_camera_intrincis[2][2] = 1.0
                camera_intrinsics.append(temp_camera_intrincis)
                temp_camera_intrincis = np.zeros((3, 4))
                param_count = 0

    # Generating projection and extrinsic matrices
    projection_matrices = []
    extrinsic_matrices = []
    projection_matrix = np.zeros((3, 4))
    for i in range(len(visible_view_indexes)):
        rigid_transform = transformations.quaternion_matrix([poses["poses[" + str(i) + "]"]['orientation']['w'], poses["poses[" + str(i) + "]"]['orientation']['x'],
                                                             poses["poses[" + str(i) + "]"]['orientation']['y'], poses["poses[" + str(i) + "]"]['orientation']['z']])
        rigid_transform[0][3] = poses["poses[" + str(i) + "]"]['position']['x']
        rigid_transform[1][3] = poses["poses[" + str(i) + "]"]['position']['y']
        rigid_transform[2][3] = poses["poses[" + str(i) + "]"]['position']['z']

        transform = np.asmatrix(rigid_transform)
        transform = np.linalg.inv(transform)
        extrinsic_matrices.append(transform)

        projection_matrix = np.dot(camera_intrinsics[0], transform)
        projection_matrices.append(projection_matrix)

    # Read mask image
    img_mask = cv2.imread(str(prefix_seq / 'undistorted_mask.bmp'), cv2.IMREAD_GRAYSCALE)

    # Drawing 2D overlay of sparse point cloud onto every image plane
    for i in range(len(visible_view_indexes)):
        img = cv2.imread(str(prefix_seq / (("%08d") % (visible_view_indexes[i]) + '.jpg')))
        height, width = img.shape[:2]

        projection_matrix = projection_matrices[i]
        extrinsic_matrix = extrinsic_matrices[i]

        for j in range(len(lists_3D_points)):
            if view_indexes_per_point[j][i] > 0.5:
                point_3D_position = np.asarray(lists_3D_points[j])
                point_3D_position_camera = np.asarray(extrinsic_matrix).dot(point_3D_position)
                point_3D_position_camera = point_3D_position_camera / point_3D_position_camera[3]

                point_projected_undistorted = np.asarray(projection_matrix).dot(point_3D_position)
                point_projected_undistorted = point_projected_undistorted / point_projected_undistorted[2]

                round_u = int(round(point_projected_undistorted[0]))
                round_v = int(round(point_projected_undistorted[1]))
                # We will treat this point as valid if it is projected into the range of mask
                if 0 <= round_u < width and 0 <= round_v < height:
                    if img_mask[round_v][round_u] == 255:
                        cv2.circle(img, (round_u, round_v), 3, (0, 255, 0), -1)
            else:
                point_3D_position = np.asarray(lists_3D_points[j])
                point_3D_position_camera = np.asarray(extrinsic_matrix).dot(point_3D_position)
                point_3D_position_camera = point_3D_position_camera / point_3D_position_camera[3]

                point_projected_undistorted = np.asarray(projection_matrix).dot(point_3D_position)
                point_projected_undistorted = point_projected_undistorted / point_projected_undistorted[2]

                round_u = int(round(point_projected_undistorted[0]))
                round_v = int(round(point_projected_undistorted[1]))
                # We will treat this point as valid if it is projected into the range of mask
                if 0 <= round_u < width and 0 <= round_v < height:
                    if img_mask[round_v][round_u] == 255:
                        # pass
                        cv2.circle(img, (round_u, round_v), 3, (0, 255, 0), -1)

        cv2.imshow("projected spatial points", img)
        cv2.waitKey()

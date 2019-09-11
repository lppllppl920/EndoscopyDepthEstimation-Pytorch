'''
Author: Xingtong Liu, Ayushi Sinha, Masaru Ishii, Gregory D. Hager, Austin Reiter, Russell H. Taylor, and Mathias Unberath

Copyright (C) 2019 Johns Hopkins University - All Rights Reserved
You may use, distribute and modify this code under the
terms of the GNU GENERAL PUBLIC LICENSE Version 3 license for non-commercial usage.

You should have received a copy of the GNU GENERAL PUBLIC LICENSE Version 3 license with
this file. If not, please write to: xliu89@jh.edu or rht@jhu.edu or unberath@jhu.edu
'''

import tqdm
import cv2
import numpy as np
from pathlib import Path
import torchsummary
import torch
import random
from tensorboardX import SummaryWriter
import argparse
import datetime
import scipy.io
# Local
import models
import losses
import utils
import dataset
import geonet_models
import sfmlearner_models

if __name__ == '__main__':
    cv2.destroyAllWindows()
    parser = argparse.ArgumentParser(
        description='Self-supervised Depth Estimation on Monocular Endoscopy Dataset -- Evaluation with ours',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--adjacent_range', nargs='+', type=int, help='interval range for a pair of video frames')
    parser.add_argument('--id_range', nargs='+', type=int, help='id range for the training and testing dataset')
    parser.add_argument('--input_downsampling', type=float, default=4.0,
                        help='image downsampling rate to speed up training and reduce overfitting')
    parser.add_argument('--torchsummary_input_size', nargs='+', type=int, default=None,
                        help='input size for torchsummary (analysis purpose only)')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for training and testing')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for input data loader')
    parser.add_argument('--network_downsampling', type=int, default=64, help='network downsampling of model')
    parser.add_argument('--inlier_percentage', type=float, default=0.99,
                        help='percentage of inliers of SfM point clouds (for pruning some outliers)')
    parser.add_argument('--validation_interval', type=int, default=1, help='epoch interval for validation')
    parser.add_argument('--zero_division_epsilon', type=float, default=1.0e-8, help='epsilon to prevent zero division')
    parser.add_argument('--display_interval', type=int, default=10, help='iteration interval for image display')
    parser.add_argument('--testing_patient_id', nargs='+', type=int, help='id of the testing patient')
    parser.add_argument('--load_intermediate_data', action='store_true', help='whether to load intermediate data')
    parser.add_argument('--use_hsv_colorspace', action='store_true',
                        help='convert RGB to hsv colorspace')
    parser.add_argument('--training_result_root', type=str, help='root of the training input and ouput')
    parser.add_argument('--training_data_root', type=str, help='path to the training data')
    parser.add_argument('--architecture_summary', action='store_true', help='display the network architecture')
    parser.add_argument('--trained_model_path', type=str, default=None,
                        help='path to the trained student model')
    parser.add_argument('--trained_geonet_model_path', type=str, default=None,
                        help='path to the trained geonet model')
    parser.add_argument('--trained_sfmlearner_model_path', type=str, default=None,
                        help='path to the trained sfmlearner model')
    parser.add_argument('--selected_frame_index_list', nargs='+', type=int, required=False, default=None,
                        help='selected frame index list)')

    args = parser.parse_args()

    # Fix randomness for reproducibility
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(10085)
    np.random.seed(10085)
    random.seed(10085)

    # Hyper-parameters
    adjacent_range = args.adjacent_range
    input_downsampling = args.input_downsampling
    if args.torchsummary_input_size is not None and len(args.torchsummary_input_size) == 2:
        height, width = args.torchsummary_input_size
    else:
        height = 256
        width = 320
    batch_size = args.batch_size
    num_workers = args.num_workers
    network_downsampling = args.network_downsampling
    inlier_percentage = args.inlier_percentage
    validation_each = args.validation_interval
    depth_scaling_epsilon = args.zero_division_epsilon
    depth_warping_epsilon = args.zero_division_epsilon
    wsl_epsilon = args.zero_division_epsilon
    display_each = args.display_interval
    testing_patient_id = args.testing_patient_id
    load_intermediate_data = args.load_intermediate_data
    is_hsv = args.use_hsv_colorspace
    training_result_root = args.training_result_root
    display_architecture = args.architecture_summary
    trained_model_path = args.trained_model_path
    trained_geonet_model_path = args.trained_geonet_model_path
    trained_sfmlearner_model_path = args.trained_sfmlearner_model_path
    training_data_root = Path(args.training_data_root)
    id_range = args.id_range
    selected_frame_index_list = args.selected_frame_index_list

    currentDT = datetime.datetime.now()

    log_root = Path(training_result_root) / "depth_estimation_evaluation_run_{}_{}_{}_{}_test_id_{}".format(
        currentDT.month,
        currentDT.day,
        currentDT.hour,
        currentDT.minute,
        testing_patient_id)
    if not log_root.exists():
        log_root.mkdir()
    writer = SummaryWriter(logdir=str(log_root))
    print("Tensorboard visualization at {}".format(str(log_root)))

    # Get color image filenames
    train_filenames, val_filenames, test_filenames = utils.get_color_file_names_by_bag(training_data_root,
                                                                                       validation_patient_id=[],
                                                                                       testing_patient_id=testing_patient_id,
                                                                                       id_range=id_range)
    folder_list = utils.get_parent_folder_names(training_data_root, id_range=id_range)

    # Build validation dataset
    validation_dataset = dataset.SfMDataset(image_file_names=test_filenames,
                                            folder_list=folder_list,
                                            adjacent_range=adjacent_range,
                                            transform=None,
                                            downsampling=input_downsampling,
                                            network_downsampling=network_downsampling,
                                            inlier_percentage=inlier_percentage,
                                            use_store_data=True,
                                            store_data_root=training_data_root,
                                            phase="validation", is_hsv=is_hsv,
                                            num_pre_workers=num_workers, visible_interval=30, rgb_mode="rgb")

    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False,
                                                    num_workers=batch_size)

    depth_estimation_model_ours = models.FCDenseNet57(n_classes=1)
    # Initialize the depth estimation network with Kaiming He initialization
    depth_estimation_model_ours = utils.init_net(depth_estimation_model_ours, type="kaiming", mode="fan_in",
                                                 activation_mode="relu",
                                                 distribution="normal")
    # Multi-GPU running
    depth_estimation_model_ours = torch.nn.DataParallel(depth_estimation_model_ours)
    # Summary network architecture
    if display_architecture:
        torchsummary.summary(depth_estimation_model_ours, input_size=(3, height, width))
    # Load previous student model, lr scheduler, and so on
    if Path(trained_model_path).exists():
        print("Loading {:s} ...".format(trained_model_path))
        state = torch.load(trained_model_path)
        step = state['step']
        epoch = state['epoch']
        depth_estimation_model_ours.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {}'.format(epoch, step))
    else:
        print("No trained student model detected")
        raise OSError

    depth_estimation_model_geonet = geonet_models.DispNet(training=False)
    # Initialize the depth estimation network with Kaiming He initialization
    utils.init_net(depth_estimation_model_geonet, type="kaiming", mode="fan_in", activation_mode="relu",
                   distribution="normal")
    # Summary network architecture
    if display_architecture:
        torchsummary.summary(depth_estimation_model_geonet, input_size=(3, height, width))
    # Multi-GPU running
    depth_estimation_model_geonet = torch.nn.DataParallel(depth_estimation_model_geonet)
    # Load previous student model
    state = {}
    if Path(trained_geonet_model_path).exists():
        print("Loading {:s} ...".format(trained_geonet_model_path))
        state = torch.load(trained_geonet_model_path)
        depth_estimation_model_geonet.load_state_dict(state['model'])
    else:
        print("Trained model could not be found")
        raise OSError

    depth_estimation_model_sfmlearner = sfmlearner_models.DispNetS(training=False)
    # Initialize the depth estimation network with Kaiming He initialization
    utils.init_net(depth_estimation_model_sfmlearner, type="kaiming", mode="fan_in", activation_mode="relu",
                   distribution="normal")
    # Summary network architecture
    if display_architecture:
        torchsummary.summary(depth_estimation_model_sfmlearner, input_size=(3, height, width))
    # Load previous student model
    state = {}
    if Path(trained_sfmlearner_model_path).exists():
        print("Loading {:s} ...".format(trained_sfmlearner_model_path))
        state = torch.load(trained_sfmlearner_model_path)
        depth_estimation_model_sfmlearner.load_state_dict(state['state_dict'])
    else:
        print("Trained model could not be found")
        raise OSError
    # Multi-GPU running
    depth_estimation_model = torch.nn.DataParallel(depth_estimation_model_sfmlearner)

    # Custom layers
    depth_scaling_layer = models.DepthScalingLayer(epsilon=depth_scaling_epsilon)
    # Loss functions
    abs_rel_error = losses.AbsRelError()
    threshold = losses.Threshold()

    depth_estimation_model_ours.eval()
    mean_loss = 0.0
    mean_depth_consistency_loss = 0.0
    mean_sparse_flow_loss = 0.0
    tq = tqdm.tqdm(total=len(validation_loader) * batch_size, dynamic_ncols=True, ncols=40)
    tq.set_description('Validation Epoch {}'.format(epoch))

    abs_rel_error_list = []
    sigma_1_list = []
    sigma_2_list = []
    sigma_3_list = []

    with torch.no_grad():
        for batch, (
                colors_1, colors_2, sparse_depths_1, sparse_depths_2, sparse_depth_masks_1,
                sparse_depth_masks_2, sparse_flows_1,
                sparse_flows_2, sparse_flow_masks_1, sparse_flow_masks_2, boundaries, rotations_1_wrt_2,
                rotations_2_wrt_1, translations_1_wrt_2, translations_2_wrt_1, intrinsics,
                folders, filenames) in enumerate(validation_loader):

            colors_1 = colors_1.cuda()
            colors_2 = colors_2.cuda()
            sparse_depths_1 = sparse_depths_1.cuda()
            sparse_depths_2 = sparse_depths_2.cuda()
            sparse_depth_masks_1 = sparse_depth_masks_1.cuda()
            sparse_depth_masks_2 = sparse_depth_masks_2.cuda()
            boundaries = boundaries.cuda()

            colors_1 = boundaries * colors_1

            # Predicted depth from student model
            predicted_depth_maps_1 = depth_estimation_model_ours(colors_1)
            scaled_depth_maps_1, normalized_scale_std_1 = depth_scaling_layer(
                [torch.abs(predicted_depth_maps_1), sparse_depths_1, sparse_depth_masks_1])
            scaled_depth_maps_1 = scaled_depth_maps_1 * boundaries

            array_colors_1 = np.uint8((0.5 * (colors_1.permute(0, 2, 3, 1) + 1.0)).data.cpu().numpy() * 255)

            bgr_colors_list_1 = []
            rgb_colors_list_1 = []
            for i in range(array_colors_1.shape[0]):
                rgb_colors_list_1.append(cv2.cvtColor(array_colors_1[i], cv2.COLOR_HSV2RGB_FULL))
                bgr_colors_list_1.append(cv2.cvtColor(array_colors_1[i], cv2.COLOR_HSV2BGR_FULL))

            bgr_colors = torch.from_numpy(np.asarray(bgr_colors_list_1).reshape(-1, height, width, 3)).cuda().permute(0,
                                                                                                                      3,
                                                                                                                      1,
                                                                                                                      2).float() / 127.5 - 1.0
            rgb_colors = torch.from_numpy(np.asarray(rgb_colors_list_1).reshape(-1, height, width, 3)).cuda().permute(0,
                                                                                                                      3,
                                                                                                                      1,
                                                                                                                      2).float() / 127.5 - 1.0
            bgr_colors = boundaries * bgr_colors
            rgb_colors = boundaries * rgb_colors

            # Predicted depth from student model
            disparities_1 = depth_estimation_model_geonet(rgb_colors)
            predicted_depth_maps_1_geonet = 1.0 / disparities_1
            scaled_depth_maps_1_geonet, normalized_scale_std_1 = depth_scaling_layer(
                [torch.abs(predicted_depth_maps_1_geonet), sparse_depths_1, sparse_depth_masks_1])
            scaled_depth_maps_1_geonet = scaled_depth_maps_1_geonet * boundaries

            # Predicted depth from student model
            disparities_1 = depth_estimation_model_sfmlearner(rgb_colors)
            predicted_depth_maps_1_sfmlearner = 1.0 / disparities_1
            scaled_depth_maps_1_sfmlearner, normalized_scale_std_1 = depth_scaling_layer(
                [torch.abs(predicted_depth_maps_1_sfmlearner), sparse_depths_1, sparse_depth_masks_1])
            scaled_depth_maps_1_sfmlearner = scaled_depth_maps_1_sfmlearner * boundaries

            for i in range(scaled_depth_maps_1.shape[0]):
                if int(filenames[i]) in selected_frame_index_list:
                    display_depth = scaled_depth_maps_1[i].view(height, width).data.cpu().numpy()
                    min_depth = 0.0
                    max_depth = np.max(display_depth)

                    display_depth = display_depth / max_depth

                    display_depth_geonet = scaled_depth_maps_1_geonet[i].view(height, width).data.cpu().numpy()
                    display_depth_geonet = np.clip(display_depth_geonet / max_depth, a_min=0.0, a_max=1.0)

                    display_depth_sfmlearner = scaled_depth_maps_1_sfmlearner[i].view(height, width).data.cpu().numpy()
                    display_depth_sfmlearner = np.clip(display_depth_sfmlearner / max_depth, a_min=0.0, a_max=1.0)

                    cv2.imwrite(str(log_root / (str(int(filenames[i])) + "_ours.png")),
                                cv2.applyColorMap(np.uint8(255 * display_depth), cv2.COLORMAP_JET))
                    cv2.imwrite(str(log_root / (str(int(filenames[i])) + "_geonet.png")),
                                cv2.applyColorMap(np.uint8(255 * display_depth_geonet), cv2.COLORMAP_JET))
                    cv2.imwrite(str(log_root / (str(int(filenames[i])) + "_sfmlearner.png")),
                                cv2.applyColorMap(np.uint8(255 * display_depth_sfmlearner), cv2.COLORMAP_JET))

            # error = abs_rel_error([scaled_depth_maps_1, sparse_depths_1, sparse_depth_masks_1])
            # threshold_error = threshold([scaled_depth_maps_1, sparse_depths_1, sparse_depth_masks_1])
            # sigma_1, sigma_2, sigma_3 = threshold_error
            #
            # error = error.view(-1).data.cpu().numpy()
            # sigma_1 = sigma_1.view(-1).data.cpu().numpy()
            # sigma_2 = sigma_2.view(-1).data.cpu().numpy()
            # sigma_3 = sigma_3.view(-1).data.cpu().numpy()
            #
            # for i in range(error.shape[0]):
            #     abs_rel_error_list.append(error[i])
            #     sigma_1_list.append(sigma_1[i])
            #     sigma_2_list.append(sigma_2[i])
            #     sigma_3_list.append(sigma_3[i])
            #
            # if batch == 0:
            #     mean_loss = np.mean(error)
            #     mean_sigma_1 = np.mean(sigma_1)
            #     mean_sigma_2 = np.mean(sigma_2)
            #     mean_sigma_3 = np.mean(sigma_3)
            # else:
            #     mean_loss = (mean_loss * batch + np.mean(error)) / (batch + 1.0)
            #     mean_sigma_1 = (mean_sigma_1 * batch + np.mean(sigma_1)) / (batch + 1.0)
            #     mean_sigma_2 = (mean_sigma_2 * batch + np.mean(sigma_2)) / (batch + 1.0)
            #     mean_sigma_3 = (mean_sigma_3 * batch + np.mean(sigma_3)) / (batch + 1.0)
            #
            # tq.set_postfix(abs_rel_error='avg: {:.5f} cur: {:.5f}'.format(
            #     mean_loss, np.mean(error)),
            #     sigma_1='avg: {:.5f} cur: {:.5f}'.format(
            #         mean_sigma_1, np.mean(sigma_1)),
            #     sigma_2='avg: {:.5f} cur: {:.5f}'.format(
            #         mean_sigma_2, np.mean(sigma_2)),
            #     sigma_3='avg: {:.5f} cur: {:.5f}'.format(
            #         mean_sigma_3, np.mean(sigma_3)))
            tq.update(batch_size)

    # evaluation_dict = {}
    # evaluation_dict["abs_rel_err_patient_" + str(testing_patient_id)] = abs_rel_error_list
    # evaluation_dict["sigma_1_patient_" + str(testing_patient_id)] = sigma_1_list
    # evaluation_dict["sigma_2_patient_" + str(testing_patient_id)] = sigma_2_list
    # evaluation_dict["sigma_3_patient_" + str(testing_patient_id)] = sigma_3_list
    # scipy.io.savemat(file_name=str(log_root / "ours_patient_{}.mat".format(testing_patient_id)), mdict=evaluation_dict)
    tq.close()

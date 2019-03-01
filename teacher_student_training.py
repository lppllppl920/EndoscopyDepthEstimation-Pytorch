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
import math
import torch
import random
from tensorboardX import SummaryWriter
import albumentations as albu
import argparse
import shutil
# Local
import models
import losses
import utils
import dataset
import scheduler

if __name__ == '__main__':
    cv2.destroyAllWindows()
    parser = argparse.ArgumentParser(
        description='Self-supervised Depth Estimation on Monocular Endoscopy Dataset -- Student Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--adjacent_range', nargs='+', type=int, help='interval range for a pair of video frames')
    parser.add_argument('--downsampling', type=float, default=4.0,
                        help='image downsampling rate to speed up training and reduce overfitting')
    parser.add_argument('--torchsummary_input_size', nargs='+', type=int, default=None,
                        help='input size for torchsummary (analysis purpose only)')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for training and testing')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for input data loader')
    parser.add_argument('--dcl_weight', type=float, default=1.0, help='weight for depth consistency loss')
    parser.add_argument('--sfl_weight', type=float, default=100.0, help='weight for sparse flow loss')
    parser.add_argument('--ssl_weight', type=float, default=0.3,
                        help='weight for scale standard deviation loss')
    parser.add_argument('--max_lr', type=float, default=1.0e-3, help='upper bound learning rate for cyclic lr')
    parser.add_argument('--min_lr', type=float, default=1.0e-4, help='lower bound learning rate for cyclic lr')
    parser.add_argument('--teacher_depth', type=int, default=7, help='depth of teacher model')
    parser.add_argument('--filter_base', type=int, default=3, help='filter base of teacher model')
    parser.add_argument('--inlier_percentage', type=float, default=0.995,
                        help='percentage of inliers of SfM point clouds (for pruning some outliers)')
    parser.add_argument('--validation_interval', type=int, default=1, help='epoch interval for validation')
    parser.add_argument('--zero_division_epsilon', type=float, default=1.0e-8, help='epsilon to prevent zero division')
    parser.add_argument('--display_interval', type=int, default=10, help='iteration interval for image display')
    parser.add_argument('--testing_patient_id', type=int, help='id of the testing patient')
    parser.add_argument('--load_intermediate_data', action='store_true', help='whether to load intermediate data')
    parser.add_argument('--student_learn_from_sfm', action='store_true', help='whether to learn directly from sfm')
    parser.add_argument('--use_view_indexes_per_point', action='store_true',
                        help='whether to use view indexes for reconstructing a particular point by SfM')
    parser.add_argument('--visualize_dataset_input', action='store_true',
                        help='whether to visualize input of data loader')
    parser.add_argument('--sfm_result_outlier_threshold', type=float, default=0.2,
                        help='outlier threshold related to sparse flow loss')
    parser.add_argument('--use_previous_student_model', action='store_true',
                        help='whether to use previous trained student model')
    parser.add_argument('--number_epoch', type=int, help='number of epochs in total')
    parser.add_argument('--use_hsv_colorspace', action='store_true',
                        help='convert RGB to hsv colorspace')
    parser.add_argument('--no_shuffle', action='store_false',
                        help='do not shuffle training data')
    parser.add_argument('--training_root', type=str, help='root of the training input and ouput')
    parser.add_argument('--architecture_summary', action='store_true', help='display the network architecture')
    parser.add_argument('--teacher_model_path', type=str, default=None, help='path to the trained teacher model')
    parser.add_argument('--student_model_path', type=str, default=None, help='path to the trained student model')
    parser.add_argument('--training_data_path', type=str, help='path to the training data')

    args = parser.parse_args()

    # Fix randomness for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(10085)
    np.random.seed(10085)
    random.seed(10085)

    device = torch.device("cuda")

    # Hyper-parameters
    adjacent_range = args.adjacent_range
    downsampling = args.downsampling
    if args.torchsummary_input_size is not None and len(args.torchsummary_input_size) == 2:
        height, width = args.torchsummary_input_size
    else:
        height = 256
        width = 320
    batch_size = args.batch_size
    num_workers = args.num_workers
    depth_consistency_weight = args.dcl_weight
    sparse_flow_weight = args.sfl_weight
    scale_std_loss_weight = args.ssl_weight
    max_lr = args.max_lr
    min_lr = args.min_lr
    teacher_depth = args.teacher_depth
    filter_base = args.filter_base
    inlier_percentage = args.inlier_percentage
    validation_each = args.validation_interval
    depth_scaling_epsilon = args.zero_division_epsilon
    depth_warping_epsilon = args.zero_division_epsilon
    wsl_epsilon = args.zero_division_epsilon
    display_each = args.display_interval
    which_bag = args.testing_patient_id
    load_intermediate_data = args.load_intermediate_data
    student_learn_from_sfm = args.student_learn_from_sfm
    outlier_threshold = args.sfm_result_outlier_threshold
    use_view_indexes_per_point = args.use_view_indexes_per_point
    visualize = args.visualize_dataset_input
    use_previous_student_model = args.use_previous_student_model
    n_epochs = args.number_epoch
    is_hsv = args.use_hsv_colorspace
    shuffle = args.no_shuffle
    training_root = args.training_root
    display_architecture = args.architecture_summary
    teacher_model_path = args.teacher_model_path
    best_student_model_path = args.student_model_path
    training_data_root = Path(args.training_data_path)

    depth_estimation_model_teacher = []
    failure_sequences = []

    if student_learn_from_sfm:
        enable_failure_detection = True
        enable_validation = True
        use_validation_loss = False
    else:
        enable_failure_detection = False
        enable_validation = False
        use_validation_loss = False

    if student_learn_from_sfm:
        training_transforms = albu.Compose([
            # Color augmentation
            albu.OneOf([
                albu.Compose([
                    albu.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                    albu.RandomGamma(gamma_limit=(50, 180), p=0.5),
                    albu.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=0, val_shift_limit=0, p=0.5)]),
                albu.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=30, p=0.5)
            ]),
            # Image quality augmentation
            albu.OneOf([
                albu.Blur(p=0.5),
                albu.MedianBlur(p=0.5),
                albu.MotionBlur(p=0.5),
                albu.JpegCompression(quality_lower=20, quality_upper=100, p=0.5)
            ]),
            # Noise augmentation
            albu.OneOf([
                albu.GaussNoise(var_limit=(10, 50), p=0.5),
                albu.IAAAdditiveGaussianNoise(loc=0, scale=(0.01 * 255, 0.05 * 255))
            ]),
        ], p=1.)
    else:
        training_transforms = albu.Compose(
            [albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)], p=1.)

    validation_transforms = albu.Compose([
        albu.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0, p=1.)], p=1.)

    root = Path(training_root) / 'test_down_{down}_depth_{depth}_base_{base}_inliner_{inlier}_hsv_{hsv}_bag_{bag}'.format(
        bag=which_bag,
        down=downsampling,
        depth=teacher_depth,
        base=filter_base,
        inlier=inlier_percentage,
        hsv=is_hsv)

    writer = SummaryWriter(log_dir=str(root / "runs"))
    precompute_root = root / "precompute"
    try:
        precompute_root.mkdir(mode=0o777, parents=True)
    except OSError:
        pass
    # Directories for storing models and results
    model_root = root / "models"
    try:
        model_root.mkdir(mode=0o777, parents=True)
    except OSError:
        pass
    results_root = root / "results"
    try:
        results_root.mkdir(mode=0o777, parents=True)
    except OSError:
        pass

    # If the path to teacher model is not specified, use default path instead
    if teacher_model_path is None:
        teacher_model_path = str(model_root / "best_teacher_model.pt")

    # Get color image filenames
    train_filenames, val_filenames, test_filenames = utils.get_color_file_names_by_bag(training_data_root,
                                                                                       which_bag=which_bag,
                                                                                       split_ratio=(0.5, 0.5))
    training_folder_list, val_folder_list = utils.get_parent_folder_names(training_data_root, which_bag=which_bag)

    # Build training and validation dataset
    train_dataset = dataset.SfMDataset(image_file_names=train_filenames,
                                       folder_list=training_folder_list + val_folder_list,
                                       adjacent_range=adjacent_range, to_augment=True, transform=training_transforms,
                                       downsampling=downsampling,
                                       net_depth=teacher_depth, inlier_percentage=inlier_percentage,
                                       use_store_data=load_intermediate_data,
                                       store_data_root=precompute_root,
                                       use_view_indexes_per_point=use_view_indexes_per_point, visualize=visualize,
                                       phase="train", is_hsv=is_hsv)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle,
                                               num_workers=num_workers)

    validation_dataset = dataset.SfMDataset(image_file_names=val_filenames,
                                            folder_list=training_folder_list + val_folder_list,
                                            adjacent_range=adjacent_range, to_augment=True,
                                            transform=validation_transforms,
                                            downsampling=downsampling,
                                            net_depth=teacher_depth, inlier_percentage=inlier_percentage,
                                            use_store_data=True,
                                            store_data_root=precompute_root,
                                            use_view_indexes_per_point=use_view_indexes_per_point, visualize=visualize,
                                            phase="validation", is_hsv=is_hsv)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False,
                                                    num_workers=batch_size)

    # Load trained teacher model
    if not student_learn_from_sfm:
        # Depth estimation architecture
        depth_estimation_model_teacher = models.UNet(in_channels=3, out_channels=1, depth=teacher_depth, wf=filter_base,
                                                     padding=True, up_mode='upsample')
        # Initialize the depth estimation network with Kaiming He initialization
        utils.init_net(depth_estimation_model_teacher, type="kaiming", mode="fan_in", activation_mode="relu",
                       distribution="normal")
        # Multi-GPU running
        depth_estimation_model_teacher = torch.nn.DataParallel(depth_estimation_model_teacher)
        # Define teacher network weight path
        # Load previous teacher model
        if Path(teacher_model_path).exists():
            state = torch.load(teacher_model_path)
            depth_estimation_model_teacher.load_state_dict(state['model'])
            for param in depth_estimation_model_teacher.parameters():
                param.requires_grad = False
        else:
            raise OSError

        # Summary network architecture
        if display_architecture:
            torchsummary.summary(depth_estimation_model_teacher, input_size=(3, height, width))

    depth_estimation_model_student = models.FCDenseNet57(n_classes=1)
    # Initialize the depth estimation network with Kaiming He initialization
    utils.init_net(depth_estimation_model_student, type="kaiming", mode="fan_in", activation_mode="relu",
                   distribution="normal")
    # Multi-GPU running
    depth_estimation_model_student = torch.nn.DataParallel(depth_estimation_model_student)
    # Summary network architecture
    if display_architecture:
        torchsummary.summary(depth_estimation_model_student, input_size=(3, height, width))
    # Optimizer
    optimizer = torch.optim.SGD(depth_estimation_model_student.parameters(), lr=max_lr, momentum=0.9)
    lr_scheduler = scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr)

    # Custom layers
    depth_scaling_layer = models.DepthScalingLayer(epsilon=depth_scaling_epsilon)
    depth_warping_layer = models.DepthWarpingLayer(epsilon=depth_warping_epsilon)
    flow_from_depth_layer = models.FlowfromDepthLayer()
    # Loss functions
    sparse_masked_l1_loss = losses.SparseMaskedL1Loss()
    sparse_masked_l1_loss_detector = losses.SparseMaskedL1LossDisplay()
    scale_invariant_loss = losses.ScaleInvariantLoss()
    normalized_weighted_masked_l2_loss = losses.NormalizedWeightedMaskedL2Loss()

    if best_student_model_path is None:
        best_student_model_path = str(model_root / "best_student_model.pt")
    # Load previous student model, lr scheduler, failure SfM sequences, and so on
    best_validation_losses = [1000.0]
    if use_previous_student_model:
        if Path(best_student_model_path).exists():
            print("Loading {:s} ...".format(best_student_model_path))
            state = torch.load(best_student_model_path)
            step = state['step']
            epoch = state['epoch']
            depth_estimation_model_student.load_state_dict(state['model'])
            if 'validation' in state:
                if use_validation_loss:
                    best_validation_losses = state['validation']
                    print("previous best outlier robust validation losses: {:.5f}".format(
                        np.mean(best_validation_losses)))
            if 'failure' in state:
                failure_sequences = state['failure']
                print("failure sequences: ", failure_sequences)
            print('Restored model, epoch {}, step {}'.format(epoch, step))
        else:
            print("No previous student model detected")
            raise OSError
    else:
        epoch = 0
        step = 0

    for epoch in range(epoch, n_epochs + 1):
        epoch_failure_sequences = {}
        # Set the seed correlated to epoch for reproducibility
        torch.manual_seed(10086 + epoch)
        np.random.seed(10086 + epoch)
        random.seed(10086 + epoch)
        if not student_learn_from_sfm:
            depth_estimation_model_teacher.eval()

        depth_estimation_model_student.train()

        # Update progress bar
        tq = tqdm.tqdm(total=len(train_loader) * num_workers, dynamic_ncols=True, ncols=40)
        # Variable initialization
        losses = []
        depth_consistency_losses = []
        sparse_flow_losses = []
        scale_std_losses = []
        mean_loss = 0.0
        depth_consistency_loss = torch.tensor(0.0).float().cuda()
        sparse_flow_loss = torch.tensor(0.0).float().cuda()
        scale_std_loss = torch.tensor(0.0).float().cuda()
        failure_detection_loss_1 = torch.tensor(0.0).float().cuda()
        failure_detection_loss_2 = torch.tensor(0.0).float().cuda()
        Nan_folder_list = []
        # Set the student model to train mode
        for param in depth_estimation_model_student.parameters():
            param.requires_grad = True

        try:
            for batch, (
                    colors_1, colors_2, sparse_depths_1, sparse_depths_2, sparse_depth_masks_1, sparse_depth_masks_2,
                    flows_1, flows_2, flow_masks_1, flow_masks_2, sparse_depths_1_visible,
                    sparse_depths_2_visible,
                    sparse_depth_masks_1_visible, sparse_depth_masks_2_visible, flows_1_visible,
                    flows_2_visible, flow_masks_1_visible, flow_masks_2_visible, boundaries, rotations,
                    rotations_inverse, translations, translations_inverse, intrinsic_matrices, folders) in \
                    enumerate(train_loader):

                # Update learning rate
                lr_scheduler.batch_step(batch_iteration=step)
                tq.set_description('Epoch {}, lr {}'.format(epoch, lr_scheduler.get_lr()))

                colors_1, colors_2, \
                sparse_depths_1, sparse_depths_2, sparse_depth_masks_1, sparse_depth_masks_2, flows_1, flows_2, flow_masks_1, \
                flow_masks_2, \
                sparse_depths_1_visible, sparse_depths_2_visible, sparse_depth_masks_1_visible, sparse_depth_masks_2_visible, flows_1_visible, flows_2_visible, flow_masks_1_visible, \
                flow_masks_2_visible, \
                boundaries, rotations, rotations_inverse, translations, translations_inverse, intrinsic_matrices = \
                    colors_1.to(device), colors_2.to(device), \
                    sparse_depths_1.to(device), sparse_depths_2.to(device), \
                    sparse_depth_masks_1.to(device), sparse_depth_masks_2.to(device), flows_1.to(
                        device), flows_2.to(
                        device), flow_masks_1.to(device), flow_masks_2.to(device), \
                    sparse_depths_1_visible.to(device), sparse_depths_2_visible.to(
                        device), sparse_depth_masks_1_visible.to(
                        device), sparse_depth_masks_2_visible.to(device), flows_1_visible.to(
                        device), flows_2_visible.to(device), flow_masks_1_visible.to(device), \
                    flow_masks_2_visible.to(device), \
                    boundaries.to(device), rotations.to(device), \
                    rotations_inverse.to(device), translations.to(device), translations_inverse.to(
                        device), intrinsic_matrices.to(device)

                if enable_failure_detection:
                    for i, folder in enumerate(folders):
                        if folder in failure_sequences:
                            flows_1[i] = flows_1_visible[i]
                            flows_2[i] = flows_2_visible[i]
                            flow_masks_1[i] = flow_masks_1_visible[i]
                            flow_masks_2[i] = flow_masks_2_visible[i]

                # Binarize the boundaries
                boundaries = torch.where(boundaries >= torch.tensor(0.9).float().cuda(),
                                         torch.tensor(1.0).float().cuda(), torch.tensor(0.0).float().cuda())
                # Remove invalid regions of color images
                colors_1 = boundaries * colors_1
                colors_2 = boundaries * colors_2
                # We should let the student learn by themselves after the generated results from student
                # is similar with these from teacher
                if not student_learn_from_sfm:
                    # Learn from teacher
                    loss, scaled_depth_maps_1, scaled_depth_maps_2, goal_scaled_depth_maps_1, goal_scaled_depth_maps_2 = utils.learn_from_teacher(
                        boundaries, colors_1,
                        colors_2,
                        depth_estimation_model_teacher,
                        depth_estimation_model_student,
                        scale_invariant_loss)
                    # Handle nan cases
                    if math.isnan(loss.item()):
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.zero_grad()
                        continue
                    else:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        losses.append(loss.item())
                        mean_loss = np.mean(losses)

                    step += 1
                    tq.update(batch_size)
                    tq.set_postfix(loss='average: {:.5f}, current: {:.5f}'.format(mean_loss, loss.item()))
                    # TensorboardX
                    writer.add_scalars('Training', {'overall': mean_loss}, step)

                    # Display depth and color at tensorboardX
                    if batch % display_each == 0:
                        utils.display_training_output(1, step, writer, colors_1, scaled_depth_maps_1)
                        utils.display_training_output(2, step, writer, colors_2, scaled_depth_maps_2)
                        utils.display_depth_goal(1, step, writer, goal_scaled_depth_maps_1)
                        utils.display_depth_goal(2, step, writer, goal_scaled_depth_maps_2)

                else:
                    loss, scaled_depth_maps_1, scaled_depth_maps_2, epoch_failure_sequences, \
                    depth_consistency_loss, sparse_flow_loss, scale_std_loss, warped_depth_maps_2_to_1, warped_depth_maps_1_to_2, predicted_depth_maps_1, \
                    sparse_flow_losses_1, sparse_flow_losses_2 = \
                        utils.learn_from_sfm(colors_1, colors_2, sparse_depths_1, sparse_depths_2, sparse_depth_masks_1,
                                             sparse_depth_masks_2,
                                             depth_estimation_model_student, depth_scaling_layer, sparse_flow_weight,
                                             flow_from_depth_layer, boundaries,
                                             translations, rotations, intrinsic_matrices, translations_inverse,
                                             rotations_inverse,
                                             flow_masks_1, flow_masks_2, flows_1, flows_2,
                                             enable_failure_detection,
                                             sparse_masked_l1_loss, depth_consistency_weight, depth_warping_layer,
                                             normalized_weighted_masked_l2_loss,
                                             batch, epoch, outlier_threshold, sparse_masked_l1_loss_detector,
                                             epoch_failure_sequences,
                                             folders, batch_size, visualize=visualize,
                                             scale_std_loss_weight=scale_std_loss_weight)

                    if math.isnan(loss.item()):
                        losses_array = sparse_flow_losses_1.data.cpu().numpy()
                        indexes = np.where(np.isnan(losses_array))
                        for index in indexes[0]:
                            Nan_folder_list.append(folders[index])
                        optimizer.zero_grad()
                        loss.backward()
                        # Prevent one sample from having too much impact on the training
                        torch.nn.utils.clip_grad_norm_(depth_estimation_model_student.parameters(), 10.0)
                        optimizer.zero_grad()
                        optimizer.step()
                        continue
                    else:
                        optimizer.zero_grad()
                        loss.backward()
                        # Prevent one sample from having too much impact on the training
                        torch.nn.utils.clip_grad_norm_(depth_estimation_model_student.parameters(), 10.0)
                        optimizer.step()
                        losses.append(loss.item())
                        mean_loss = np.mean(losses)
                        depth_consistency_losses.append(depth_consistency_weight * depth_consistency_loss.item())
                        sparse_flow_losses.append(sparse_flow_weight * sparse_flow_loss.item())
                        scale_std_losses.append(scale_std_loss_weight * scale_std_loss.item())

                    step += 1
                    tq.update(batch_size)
                    tq.set_postfix(loss='average: {:.5f} current: {:.5f}'.format(mean_loss, loss.item()),
                                   loss_depth_consistency='average: {:.5f} current: {:.5f}'.format(
                                       np.mean(depth_consistency_losses),
                                       depth_consistency_weight * depth_consistency_loss.item()),
                                   loss_sparse_flow='average: {:.5f} current: {:.5f}'.format(
                                       np.mean(sparse_flow_losses),
                                       sparse_flow_weight * sparse_flow_loss.item()),
                                   loss_scale_std='average: {:.5f} current: {:.5f}'.format(np.mean(scale_std_losses),
                                                                                           scale_std_loss_weight * scale_std_loss.item()))
                    writer.add_scalars('Training', {'overall': mean_loss,
                                                    'depth consistency': np.mean(depth_consistency_losses),
                                                    'sparse opt': np.mean(sparse_flow_losses)}, step)

                    # Display depth and color at TensorboardX
                    if batch % display_each == 0:
                        utils.display_network_weights(depth_estimation_model_student, writer, step)
                        utils.display_training_output(1, step, writer, colors_1, scaled_depth_maps_1 * boundaries)
                        utils.display_training_output(2, step, writer, colors_2, scaled_depth_maps_2 * boundaries)
            tq.close()
            if enable_failure_detection:
                failure_sequences = epoch_failure_sequences
                print("failure sequences: ", failure_sequences)
                print("number of failure sequences: {:d}".format(len(failure_sequences)))

            if enable_validation:
                # Save student model
                if epoch % validation_each == 0:
                    # Validation
                    # Update progress bar
                    tq = tqdm.tqdm(total=(len(validation_loader) * batch_size), dynamic_ncols=True, ncols=40)
                    tq.set_description('Epoch {} validation'.format(epoch))
                    validation_loss, validation_losses = utils.network_validation(writer, validation_loader, batch_size,
                                                                                  epoch,
                                                                                  depth_estimation_model_student,
                                                                                  device,
                                                                                  depth_scaling_layer,
                                                                                  sparse_flow_weight,
                                                                                  flow_from_depth_layer,
                                                                                  sparse_masked_l1_loss,
                                                                                  depth_consistency_weight,
                                                                                  normalized_weighted_masked_l2_loss,
                                                                                  is_hsv, depth_warping_layer,
                                                                                  results_root, tq, which_bag)

                    best_validation_losses = utils.save_student_model(model_root, depth_estimation_model_student,
                                                                      optimizer, epoch,
                                                                      step, failure_sequences, best_student_model_path,
                                                                      validation_losses,
                                                                      best_validation_losses, use_validation_loss)
                    tq.close()
            else:
                utils.save_student_model(model_root, depth_estimation_model_student, optimizer, epoch,
                                         step, failure_sequences, best_student_model_path, [0.],
                                         [1.], False)
            writer.export_scalars_to_json(str(results_root / ("all_scalars_" + str(epoch) + ".json")))
        except KeyboardInterrupt:
            writer.close()
            tq.close()
            torch.cuda.empty_cache()

    writer.close()

import torch
import argparse
from pathlib import Path
import torchsummary
import numpy as np
import random
import tqdm
import datetime
from tensorboardX import SummaryWriter
import math

from . import utils
from . import models
from . import dataset
from . import losses
from . import transforms

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GeoNet -- Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--adjacent_range', nargs='+', type=int, help='interval range for a pair of video frames')
    parser.add_argument('--id_range', nargs='+', type=int, help='id range for the training and testing dataset')
    parser.add_argument('--input_downsampling', type=float, default=4.0,
                        help='image downsampling rate to speed up training and reduce overfitting')
    parser.add_argument('--torchsummary_input_size', nargs='+', type=int, required=True,
                        help='input size for torchsummary (analysis purpose only)')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for training and testing')
    parser.add_argument('--num_worker', type=int, default=4, help='number of workers for input data loader')
    parser.add_argument('--num_source', type=int, default=5, help='number of source images for one target')
    parser.add_argument('--lr', type=float, default=2.0e-4, help='learning rate for adam optimizer')
    parser.add_argument('--beta_1', type=float, default=0.9, help='beta 1 of adam optimizer')
    parser.add_argument('--beta_2', type=float, default=0.999, help='beta 2 of adam optimizer')
    parser.add_argument('--number_epoch', type=int, default=1000, help='number of epochs in total')
    parser.add_argument('--num_iter', type=int, default=1000, help='number of iterations per epoch')
    parser.add_argument('--network_downsampling', type=int, default=64, help='network downsampling of model')
    parser.add_argument('--inlier_percentage', type=float, default=0.99,
                        help='percentage of inliers of SfM point clouds (for pruning some outliers)')
    parser.add_argument('--validation_interval', type=int, default=1, help='epoch interval for validation')
    parser.add_argument('--display_interval', type=int, default=50, help='iteration interval for image display')
    parser.add_argument('--testing_patient_id', type=int, required=True, help='id of the testing patient')
    parser.add_argument('--load_intermediate_data', action='store_true', help='whether to load intermediate data')
    parser.add_argument('--load_trained_model', action='store_true',
                        help='whether or not to load trained model')
    parser.add_argument('--training_result_root', type=str, required=True, help='root of the training input and ouput')
    parser.add_argument('--architecture_summary', action='store_true', help='display the network architecture')
    parser.add_argument('--trained_disp_model_path', type=str, default=None,
                        help='path to the trained disparity model')
    parser.add_argument('--trained_pose_model_path', type=str, default=None,
                        help='path to the trained pose model')
    parser.add_argument('--training_data_root', type=str, required=True, help='path to the training data')
    parser.add_argument('--cc_loss_weight', type=float, default=1.0, help='color consistency loss weight')
    parser.add_argument('--ds_loss_weight', type=float, default=0.5, help='depth smoothness loss weight')
    parser.add_argument('--gc_loss_weight', type=float, default=0.2, help='geometric consistency loss weight')

    args = parser.parse_args()

    adjacent_range = args.adjacent_range
    id_range = args.id_range
    input_downsampling = args.input_downsampling
    if len(args.torchsummary_input_size) == 2:
        height, width = args.torchsummary_input_size
    else:
        raise IOError

    num_iter = args.num_iter
    number_epoch = args.number_epoch
    batch_size = args.batch_size
    num_worker = args.num_worker
    num_source = args.num_source
    lr = args.lr
    beta_1 = args.beta_1
    beta_2 = args.beta_2
    network_downsampling = args.network_downsampling
    inlier_percentage = args.inlier_percentage
    validation_interval = args.validation_interval
    display_interval = args.display_interval
    testing_patient_id = args.testing_patient_id
    load_intermediate_data = args.load_intermediate_data
    load_trained_model = args.load_trained_model
    training_result_root = args.training_result_root
    architecture_summary = args.architecture_summary
    training_data_root = Path(args.training_data_root)

    cc_loss_weight = args.cc_loss_weight
    ds_loss_weight = args.ds_loss_weight
    gc_loss_weight = args.gc_loss_weight

    currentDT = datetime.datetime.now()

    trained_disp_model_path = None
    trained_pose_model_path = None
    if args.trained_disp_model_path is not None:
        trained_disp_model_path = Path(args.trained_disp_model_path)
    elif load_trained_model:
        raise IOError

    if args.trained_pose_model_path is not None:
        trained_pose_model_path = Path(args.trained_pose_model_path)
    elif load_trained_model:
        raise IOError

    log_root = Path(training_result_root) / "depth_estimation_run_{}_{}_{}_{}_bag_{}".format(currentDT.month,
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
                                                                                       testing_patient_id=testing_patient_id,
                                                                                       id_range=id_range,
                                                                                       split_ratio=(0.5, 0.5))
    training_folder_list, val_folder_list = utils.get_parent_folder_names(training_data_root,
                                                                          testing_patient_id=testing_patient_id,
                                                                          id_range=id_range)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomScaleCrop(),
        transforms.RandomColor(prob=0.7)
    ])

    train_dataset = dataset.EndoscopyFolder(image_file_names=train_filenames,
                                            folder_list=training_folder_list + val_folder_list,
                                            adjacent_range=adjacent_range,
                                            transform=train_transform, downsampling=input_downsampling,
                                            network_downsampling=network_downsampling,
                                            use_store_data=load_intermediate_data,
                                            store_data_root=training_data_root,
                                            phase="train", sequence_length=num_source,
                                            pre_processing_workers=num_worker)

    validation_dataset = dataset.EndoscopyFolder(image_file_names=val_filenames,
                                                 folder_list=training_folder_list + val_folder_list,
                                                 adjacent_range=adjacent_range,
                                                 transform=None, downsampling=input_downsampling,
                                                 network_downsampling=network_downsampling,
                                                 use_store_data=True,
                                                 store_data_root=training_data_root,
                                                 phase="train", sequence_length=num_source,
                                                 pre_processing_workers=num_worker)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_worker)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False,
                                                    num_workers=batch_size)

    disp_model = models.DispNet()
    # Initialize the depth estimation network with Kaiming He initialization
    disp_model = utils.init_net(disp_model, type="kaiming", mode="fan_in", activation_mode="relu",
                                distribution="normal")
    # Multi-GPU running
    disp_model = torch.nn.DataParallel(disp_model)
    # Summary network architecture
    if architecture_summary:
        torchsummary.summary(disp_model, input_size=(3, height, width))

    pose_model = models.PoseNet(num_source=num_source)
    # Initialize the depth estimation network with Kaiming He initialization
    pose_model = utils.init_net(pose_model, type="kaiming", mode="fan_in", activation_mode="relu",
                                distribution="normal")
    # Multi-GPU running
    pose_model = torch.nn.DataParallel(pose_model)
    # Summary network architecture
    if architecture_summary:
        torchsummary.summary(pose_model, input_size=(3 * (num_source + 1), height, width))

    optimizer = torch.optim.Adam(params=list(disp_model.parameters()) + list(pose_model.parameters()), lr=lr,
                                 betas=(beta_1, beta_2))

    # Load previous student model, lr scheduler, and so on
    if load_trained_model:
        if trained_disp_model_path.exists():
            print("Loading {:s} ...".format(str(trained_disp_model_path)))
            state = torch.load(str(trained_disp_model_path))
            step = state['step']
            epoch = state['epoch']
            disp_model.load_state_dict(state['model'])
            print('Restored model, epoch {}, step {}'.format(epoch, step))
        else:
            print("No trained model detected")
            raise OSError

        if trained_pose_model_path.exists():
            print("Loading {:s} ...".format(str(trained_pose_model_path)))
            state = torch.load(str(trained_pose_model_path))
            step = state['step']
            epoch = state['epoch']
            pose_model.load_state_dict(state['model'])
            print('Restored model, epoch {}, step {}'.format(epoch, step))
        else:
            print("No trained model detected")
            raise OSError
    else:
        epoch = 0
        step = 0

    for epoch in range(epoch, number_epoch + 1):
        # Set the seed correlated to epoch for reproducibility
        torch.manual_seed(10086 + epoch)
        np.random.seed(10086 + epoch)
        random.seed(10086 + epoch)
        pose_model.train()
        disp_model.train()

        # Update progress bar
        tq = tqdm.tqdm(total=len(train_loader) * batch_size, dynamic_ncols=True, ncols=40)

        # set the net into the training mode
        disp_model.train()
        pose_model.train()

        mean_loss = 0.0
        mean_cc_loss = 0.0
        mean_ds_loss = 0.0
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        for batch, (batch_tgt_imgs, batch_multi_src_ref_imgs, batch_masks, batch_intrinsics,
                    batch_intrinsics_inv) in enumerate(train_loader):

            if batch >= num_iter:
                break

            with torch.no_grad():
                batch_tgt_imgs = batch_tgt_imgs.cuda()
                batch_multi_src_ref_imgs = batch_multi_src_ref_imgs.cuda()  # B x num_src x 3 x H x W
                batch_intrinsics = batch_intrinsics.cuda()
                batch_intrinsics_inv = batch_intrinsics_inv.cuda()
                batch_masks = batch_masks.cuda()
                batch_size, num_src, _, height, width = batch_multi_src_ref_imgs.shape

            multi_scale_batch_disparities = disp_model(batch_tgt_imgs)
            multi_scale_batch_depths = [1 / disp for disp in multi_scale_batch_disparities]

            # output B x num_source x 6
            batch_multi_src_poses = pose_model(batch_tgt_imgs,
                                               batch_multi_src_ref_imgs.view(batch_size, num_source * 3, height, width))

            cc_loss = cc_loss_weight * losses.photometric_reconstruction_loss(batch_tgt_imgs, batch_multi_src_ref_imgs,
                                                                              batch_masks,
                                                                              batch_intrinsics, batch_intrinsics_inv,
                                                                              multi_scale_batch_depths,
                                                                              batch_multi_src_poses, 0.85,
                                                                              args.rotation_mode, args.padding_mode)

            ds_loss = ds_loss_weight * losses.depth_smoothness_loss(batch_tgt_imgs, multi_scale_batch_depths,
                                                                    batch_masks)

            loss = cc_loss + ds_loss

            if math.isnan(loss.item()) or math.isinf(loss.item()):
                optimizer.zero_grad()
                loss.backward()
                optimizer.zero_grad()
                optimizer.step()
                continue
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch == 0:
                    mean_loss = loss.item()
                    mean_cc_loss = cc_loss.item()
                    mean_ds_loss = ds_loss.item()
                else:
                    mean_loss = (mean_loss * batch + loss.item()) / (batch + 1.0)
                    mean_cc_loss = (mean_cc_loss * batch +
                                    cc_loss.item()) / (batch + 1.0)
                    mean_ds_loss = (mean_ds_loss * batch + ds_loss.item()) / (batch + 1.0)

            step += 1
            tq.update(batch_size)
            tq.set_postfix(loss='avg: {:.5f} cur: {:.5f}'.format(mean_loss, loss.item()),
                           cc_loss='avg: {:.5f} cur: {:.5f}'.format(
                               mean_cc_loss,
                               cc_loss.item()),
                           ds_loss='avg: {:.5f} cur: {:.5f}'.format(
                               mean_ds_loss,
                               ds_loss.item()))
            writer.add_scalars('Training', {'overall': mean_loss,
                                            'color_consistency': mean_cc_loss,
                                            'depth_smoothness': mean_ds_loss}, step)

            # Display depth and color at TensorboardX
            if batch % display_interval == 0:
                utils.display_output(phase="Training", idx=1, step=step, writer=writer,
                                     colors_1=batch_tgt_imgs,
                                     scaled_depth_maps_1=multi_scale_batch_depths[0], is_hsv=False)

        tq.close()

        # Save student model
        if epoch % validation_interval != 0:
            continue

        tq = tqdm.tqdm(total=len(validation_loader) * batch_size, dynamic_ncols=True, ncols=40)
        with torch.no_grad():
            disp_model.eval()
            pose_model.eval()
            tq.set_description('Validation Epoch {}'.format(epoch))
            for batch, (batch_tgt_imgs, batch_multi_src_ref_imgs, batch_masks, batch_intrinsics,
                        batch_intrinsics_inv) in enumerate(validation_loader):
                batch_tgt_imgs = batch_tgt_imgs.cuda()
                batch_multi_src_ref_imgs = batch_multi_src_ref_imgs.cuda()  # B x num_src x 3 x H x W
                batch_intrinsics = batch_intrinsics.cuda()
                batch_intrinsics_inv = batch_intrinsics_inv.cuda()
                batch_masks = batch_masks.cuda()
                batch_size, num_src, _, height, width = batch_multi_src_ref_imgs.shape

                multi_scale_batch_disparities = disp_model(batch_tgt_imgs)
                multi_scale_batch_depths = [1 / disp for disp in multi_scale_batch_disparities]

                # output B x num_source x 6
                batch_multi_src_poses = pose_model(batch_tgt_imgs,
                                                   batch_multi_src_ref_imgs.view(batch_size, num_source * 3, height,
                                                                                 width))

                cc_loss = cc_loss_weight * losses.photometric_reconstruction_loss(batch_tgt_imgs,
                                                                                  batch_multi_src_ref_imgs,
                                                                                  batch_masks,
                                                                                  batch_intrinsics,
                                                                                  batch_intrinsics_inv,
                                                                                  multi_scale_batch_depths,
                                                                                  batch_multi_src_poses, 0.85,
                                                                                  args.rotation_mode, args.padding_mode)

                ds_loss = ds_loss_weight * losses.depth_smoothness_loss(batch_tgt_imgs, multi_scale_batch_depths,
                                                                        batch_masks)

                loss = cc_loss + ds_loss

                if math.isnan(loss.item()) or math.isinf(loss.item()):
                    continue

                if batch == 0:
                    mean_loss = loss.item()
                    mean_cc_loss = cc_loss.item()
                    mean_ds_loss = ds_loss.item()
                else:
                    mean_loss = (mean_loss * batch + loss.item()) / (batch + 1.0)
                    mean_cc_loss = (mean_cc_loss * batch +
                                    cc_loss.item()) / (batch + 1.0)
                    mean_ds_loss = (mean_ds_loss * batch + ds_loss.item()) / (batch + 1.0)

                step += 1
                tq.update(batch_size)
                tq.set_postfix(loss='avg: {:.5f} cur: {:.5f}'.format(mean_loss, loss.item()),
                               cc_loss='avg: {:.5f} cur: {:.5f}'.format(
                                   mean_cc_loss,
                                   cc_loss.item()),
                               ds_loss='avg: {:.5f} cur: {:.5f}'.format(
                                   mean_ds_loss,
                                   ds_loss.item()))
                writer.add_scalars('Validation', {'overall': mean_loss,
                                                  'color_consistency': mean_cc_loss,
                                                  'depth_smoothness': mean_ds_loss}, step)

                # Display depth and color at TensorboardX
                if batch % display_interval == 0:
                    utils.display_output(phase="Validation", idx=1, step=step, writer=writer,
                                         colors_1=batch_tgt_imgs,
                                         scaled_depth_maps_1=multi_scale_batch_depths[0], is_hsv=False)
        tq.close()

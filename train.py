import tqdm
import cv2
import numpy as np
from pathlib import Path
import torchsummary
import math
import torch
import random

import matplotlib
matplotlib.use('Agg', warn=False, force=True)
from matplotlib import pyplot as plt

# Local
import transforms
import models
import losses
import utils
import dataset

if __name__ == '__main__':

    # Fix randomness for reproducibility
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)
    np.random.seed(10086)
    random.seed(10086)
    torch.backends.cudnn.deterministic = True

    cv2.destroyAllWindows()
    device = torch.device("cuda")

    # Data generating related
    split_ratio = [0.9, 0.05, 0.05]
    epsilon = 1.0e-5
    adjacent_range = [2, 7]
    # Image downsampling factor
    downsampling = 4.0
    width = 512
    height = 512
    batch_size = 8
    num_workers = 8
    # TODO: Color augmentation, Gaussian noise
    # Training related
    depth_consistency_weight = 0.0
    sparse_opt_weight = 100.0
    lr = 2.0e-4
    n_epochs = 1000
    net_depth = 7
    filter_base = 2
    periodicity = 2.0 * np.pi
    weight_decay = 0.0
    inlier_percentage = 0.99
    rotate_augment = False
    use_store_data = False
    use_view_indexes_per_point = False
    visualize = False
    shuffle = True
    is_hsv = True

    augments = transforms.DualCompose([
        transforms.Normalize(normalize_mask=False)])

    root = Path(
        "/home/xliu89/PycharmProjects/DepthEstimation/Training") / 'down_{down}_batch_{batch}_depth_{depth}_base_{base}_range_{range1}_{range2}_sparse_opt_{opt_weight}_' \
                                                                   'inliner_{inlier}_hsv_{hsv}_view_indexes_{use}'.format(
        batch=batch_size,
        down=downsampling,
        depth=net_depth,
        base=filter_base,
        range1=adjacent_range[0],
        range2=adjacent_range[1],
        opt_weight=sparse_opt_weight,
        inlier=inlier_percentage,
        hsv=is_hsv,
        use=use_view_indexes_per_point)
    data_root = Path("/home/xliu89/RemoteData/Sinus Project Data/xingtong/EndoscopicVideoData")

    checkpoint_root = root / "data"
    try:
        checkpoint_root.mkdir(mode=0o777, parents=True)
    except OSError:
        pass
    # Get color image filenames
    train_filenames, val_filenames, test_filenames = utils.get_color_file_names(data_root,
                                                                                split_ratio=[0.9, 0.05, 0.05])
    folder_list = utils.get_parent_folder_names(data_root)
    train_dataset = dataset.SfMDataset(image_file_names=train_filenames, folder_list=folder_list,
                                       split_ratio=split_ratio,
                                       adjacent_range=adjacent_range, to_augment=True, transform=augments,
                                       downsampling=downsampling,
                                       net_depth=net_depth, inlier_percentage=inlier_percentage,
                                       use_store_data=use_store_data,
                                       store_data_root=checkpoint_root,
                                       use_view_indexes_per_point=use_view_indexes_per_point, visualize=visualize,
                                       phase="train", is_hsv=is_hsv, use_random_seed=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle,
                                               num_workers=num_workers, worker_init_fn=utils.init_fn)
    # Depth estimation architecture
    depth_estimation_model = models.UNet(in_channels=3, out_channels=1, depth=net_depth, wf=filter_base, padding=True,
                                         batch_norm=False, up_mode='upsample')
    # Initialize the depth estimation network with Kaiming He initialization
    utils.init_net(depth_estimation_model, type="kaiming", mode="fan_in", activation_mode="relu", distribution="normal")
    torchsummary.summary(depth_estimation_model, input_size=(3, height, width))
    # Weight decay helps to avoid gradient explosion
    optimizer = torch.optim.Adam(depth_estimation_model.parameters(), lr=lr)  # weight_decay=weight_decay
    # Custom layers
    depth_scaling_layer = models.DepthScalingLayer(epsilon=1.0e-8)
    depth_warping_layer = models.DepthWarpingLayer()
    opt_flow_layer = models.OpticalFlowfromDepthLayer()
    # Loss functions
    weighted_scale_invariant_loss = losses.WeightedScaleInvariantLoss(epsilon=1.0e-8)
    masked_log_l1_loss = losses.MaskedLogL1Loss()
    sparse_masked_l1_loss = losses.SparseMaskedL1Loss()
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
    # Define network weight path
    model_path = model_root / \
                 'model_down_{down}_batch_{batch}_depth_{depth}_base_{base}_range_{range1}_{range2}_sparse_opt_{opt_weight}_' \
                 'inliner_{inlier}_hsv_{hsv}_view_indexes_{use}.pt'.format(
                     batch=batch_size,
                     down=downsampling,
                     depth=net_depth,
                     base=filter_base,
                     range1=adjacent_range[0],
                     range2=adjacent_range[1],
                     opt_weight=sparse_opt_weight,
                     inlier=inlier_percentage,
                     hsv=is_hsv,
                     use=use_view_indexes_per_point)
    # Load previous model if exists
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        optimizer.load_state_dict(state['optimizer'])
        depth_estimation_model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0

    dataset_length = len(train_loader)
    report_each = 10
    log = root.joinpath(
        'train_down_{down}_batch_{batch}_depth_{depth}_base_{base}_range_{range1}_{range2}_sparse_opt_{opt_weight}_' \
        'inliner_{inlier}_hsv_{hsv}_view_indexes_{use}.log'.format(
            batch=batch_size,
            down=downsampling,
            depth=net_depth,
            base=filter_base,
            range1=adjacent_range[0],
            range2=adjacent_range[1],
            opt_weight=sparse_opt_weight,
            inlier=inlier_percentage,
            hsv=is_hsv,
            use=use_view_indexes_per_point)).open('at', encoding='utf8')
    valid_each = 4
    valid_losses = []

    depth_consistency_loss = torch.tensor(0.0).float().cuda()
    sparse_opt_loss = torch.tensor(0.0).float().cuda()
    for epoch in range(epoch, n_epochs + 1):
        depth_estimation_model.train()
        tq = tqdm.tqdm(total=(len(train_loader) * batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        depth_consistency_losses = []
        sparse_opt_losses = []

        try:
            mean_loss = 0.0
            for i, (colors_1, colors_2, sparse_depths_1, sparse_depths_2, sparse_masks_1, sparse_masks_2, opt_flows_1,
                    opt_flows_2,
                    opt_flow_masks_1, opt_flow_masks_2, boundaries, rotations,
                    rotations_inverse, translations, translations_inverse, intrinsic_matrices) in enumerate(
                train_loader):

                colors_1, colors_2, sparse_depths_1, sparse_depths_2, sparse_masks_1, sparse_masks_2, opt_flows_1, opt_flows_2, opt_flow_masks_1, \
                opt_flow_masks_2, boundaries, rotations, \
                rotations_inverse, translations, translations_inverse, intrinsic_matrices = \
                    colors_1.to(device), colors_2.to(device), sparse_depths_1.to(device), sparse_depths_2.to(device), \
                    sparse_masks_1.to(device), sparse_masks_2.to(device), opt_flows_1.to(device), opt_flows_2.to(
                        device), opt_flow_masks_1.to(device), \
                    opt_flow_masks_2.to(device), boundaries.to(device), rotations.to(device), \
                    rotations_inverse.to(device), translations.to(device), translations_inverse.to(
                        device), intrinsic_matrices.to(device)

                boundaries = torch.where(boundaries >= torch.tensor(0.9).float().cuda(),
                                         torch.tensor(1.0).float().cuda(), torch.tensor(0.0).float().cuda())

                colors_1 = boundaries * colors_1
                colors_2 = boundaries * colors_2

                if rotate_augment:
                    rotation_angles = torch.tensor(
                        np.random.uniform(low=-periodicity, high=periodicity, size=colors_1.shape[0])).float().cuda()
                    source_coord_h_flat, source_coord_w_flat = models.images_rotation_coordinates_calculate(
                        thetas=rotation_angles, image_h=colors_1.shape[2], image_w=colors_1.shape[3])
                    rotated_colors_1 = models.images_warping(colors_1, source_coord_w_flat,
                                                             source_coord_h_flat, padding_mode="zeros")
                    rotated_colors_2 = models.images_warping(colors_2, source_coord_w_flat,
                                                             source_coord_h_flat, padding_mode="zeros")
                    predicted_depth_maps_1 = depth_estimation_model(rotated_colors_1)
                    predicted_depth_maps_2 = depth_estimation_model(rotated_colors_2)

                    inverse_source_coord_h_flat, inverse_source_coord_w_flat = models.images_rotation_coordinates_calculate(
                        thetas=-rotation_angles, image_h=colors_1.shape[2], image_w=colors_1.shape[3])

                    rotated_boundaries = models.images_warping(boundaries, source_coord_w_flat, source_coord_h_flat,
                                                               padding_mode="zeros")
                    reverse_boundaries = models.images_warping(rotated_boundaries, inverse_source_coord_w_flat,
                                                               inverse_source_coord_h_flat,
                                                               padding_mode="zeros")

                    boundaries = reverse_boundaries * boundaries
                    boundaries = torch.where(boundaries >= torch.tensor(0.9).float().cuda(),
                                             torch.tensor(1.0).float().cuda(), torch.tensor(0.0).float().cuda())

                    original_predicted_depth_maps_1 = models.images_warping(predicted_depth_maps_1,
                                                                            inverse_source_coord_w_flat,
                                                                            inverse_source_coord_h_flat,
                                                                            padding_mode="border")
                    original_predicted_depth_maps_2 = models.images_warping(predicted_depth_maps_2,
                                                                            inverse_source_coord_w_flat,
                                                                            inverse_source_coord_h_flat,
                                                                            padding_mode="border")

                    absolute_depth_maps_1 = boundaries * torch.abs(original_predicted_depth_maps_1)
                    absolute_depth_maps_2 = boundaries * torch.abs(original_predicted_depth_maps_2)
                    sparse_depths_1 = boundaries * sparse_depths_1
                    sparse_depths_2 = boundaries * sparse_depths_2
                    sparse_masks_1 = boundaries * sparse_masks_1
                    sparse_masks_2 = boundaries * sparse_masks_2
                    # Scale the dense depth maps to agree with the global scale of the SfM results
                    scaled_depth_maps_1 = depth_scaling_layer(
                        [absolute_depth_maps_1, sparse_depths_1, sparse_masks_1])
                    scaled_depth_maps_2 = depth_scaling_layer(
                        [absolute_depth_maps_2, sparse_depths_2, sparse_masks_2])

                else:
                    predicted_depth_maps_1 = depth_estimation_model(colors_1)
                    predicted_depth_maps_2 = depth_estimation_model(colors_2)
                    absolute_depth_maps_1 = torch.abs(predicted_depth_maps_1) * boundaries
                    absolute_depth_maps_2 = torch.abs(predicted_depth_maps_2) * boundaries
                    sparse_depths_1 = sparse_depths_1 * boundaries
                    sparse_depths_2 = sparse_depths_2 * boundaries
                    sparse_masks_1 = sparse_masks_1 * boundaries
                    sparse_masks_2 = sparse_masks_2 * boundaries
                    # Scale the dense depth maps to agree with the global scale of the SfM results
                    scaled_depth_maps_1 = depth_scaling_layer(
                        [absolute_depth_maps_1, sparse_depths_1, sparse_masks_1])
                    scaled_depth_maps_2 = depth_scaling_layer(
                        [absolute_depth_maps_2, sparse_depths_2, sparse_masks_2])

                if sparse_opt_weight > 0.0:
                    # Sparse optical flow loss
                    # Optical flow maps calculated using predicted dense depth maps and camera poses
                    # should agree with the sparse optical flows of feature points from SfM
                    opt_flows_from_depth_1 = opt_flow_layer([scaled_depth_maps_1, boundaries, translations, rotations,
                                                             intrinsic_matrices])
                    opt_flows_from_depth_2 = opt_flow_layer(
                        [scaled_depth_maps_2, boundaries, translations_inverse, rotations_inverse,
                         intrinsic_matrices])
                    opt_flow_masks_1 = opt_flow_masks_1 * boundaries
                    opt_flow_masks_2 = opt_flow_masks_2 * boundaries
                    opt_flows_1 = opt_flows_1 * boundaries
                    opt_flows_2 = opt_flows_2 * boundaries
                    opt_flows_from_depth_1 = opt_flows_from_depth_1 * boundaries
                    opt_flows_from_depth_2 = opt_flows_from_depth_2 * boundaries
                    sparse_opt_loss = 0.5 * sparse_masked_l1_loss(
                        [opt_flows_1, opt_flows_from_depth_1, opt_flow_masks_1, boundaries]) + \
                                      0.5 * sparse_masked_l1_loss(
                        [opt_flows_2, opt_flows_from_depth_2, opt_flow_masks_2, boundaries])
                    # utils.visualize_color_image("original color 1", colors_1, rebias=True, is_hsv=is_hsv)
                    # utils.draw_hsv(opt_flows_1, "sparse_flow_")
                    # utils.draw_hsv(opt_flows_from_depth_1, "flow_from_depth_")
                    # cv2.waitKey()

                if depth_consistency_weight > 0.0:
                    # Depth consistency loss
                    warped_depth_maps_2_to_1, intersect_masks_1 = depth_warping_layer(
                        [scaled_depth_maps_1, scaled_depth_maps_2, boundaries, translations, rotations,
                         intrinsic_matrices])
                    warped_depth_maps_1_to_2, intersect_masks_2 = depth_warping_layer(
                        [scaled_depth_maps_2, scaled_depth_maps_1, boundaries, translations_inverse, rotations_inverse,
                         intrinsic_matrices])

                    depth_consistency_loss = 0.5 * masked_log_l1_loss(
                        [scaled_depth_maps_1, warped_depth_maps_2_to_1, intersect_masks_1]) + \
                                             0.5 * masked_log_l1_loss(
                        [scaled_depth_maps_2, warped_depth_maps_1_to_2, intersect_masks_2])

                loss = depth_consistency_weight * depth_consistency_loss + sparse_opt_weight * sparse_opt_loss
                depth_consistency_losses.append(depth_consistency_weight * depth_consistency_loss.item())
                sparse_opt_losses.append(sparse_opt_weight * sparse_opt_loss.item())

                # This is to prevent the network getting affected by unexpected adversarial samples
                if math.isnan(loss.item()):
                    print("NAN for sample batch " + str(i))
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

                tq.set_postfix(loss='{:.5f}'.format(mean_loss),
                               loss_depth_consistency='{:.5f}'.format(np.mean(depth_consistency_losses)),
                               loss_sparse_opt='{:.5f}'.format(np.mean(sparse_opt_losses)))

                if i == 1:
                    depth_estimation_model.eval()
                    color_inputs_cpu = colors_1.data.cpu().numpy()
                    pred_depths_cpu = scaled_depth_maps_1.data.cpu().numpy()
                    color_imgs = []
                    pred_depth_imgs = []

                    for j in range(batch_size):
                        color_img = color_inputs_cpu[j]
                        pred_depth_img = pred_depths_cpu[j]

                        color_img = np.moveaxis(color_img, source=[0, 1, 2], destination=[2, 0, 1])
                        color_img = np.uint8(255 * (color_img * 0.5 + 0.5))
                        if is_hsv:
                            color_img = cv2.cvtColor(color_img, cv2.COLOR_HSV2BGR_FULL)

                        pred_depth_img = np.moveaxis(pred_depth_img, source=[0, 1, 2], destination=[2, 0, 1])
                        color_img = cv2.resize(color_img, dsize=(300, 300))
                        pred_depth_img = cv2.resize(pred_depth_img, dsize=(300, 300))
                        color_imgs.append(color_img)

                        if j == 0:
                            histr = cv2.calcHist([pred_depth_img], [0], None, histSize=[100], ranges=[0, 40])
                            plt.plot(histr, color='b')
                            plt.xlim([0, 40])
                            plt.savefig(str(results_root / 'generated_depth_hist_{epoch}.png'.format(epoch=epoch)))
                            plt.clf()
                        display_depth_img = utils.display_depth_map(pred_depth_img)
                        pred_depth_imgs.append(display_depth_img)

                    final_color = color_imgs[0]
                    final_pred_depth = pred_depth_imgs[0]
                    for j in range(batch_size - 1):
                        final_color = cv2.hconcat((final_color, color_imgs[j + 1]))
                        final_pred_depth = cv2.hconcat((final_pred_depth, pred_depth_imgs[j + 1]))

                    final = cv2.vconcat((final_color, final_pred_depth))
                    cv2.imwrite(str(results_root / 'generated_mask_{epoch}.png'.format(epoch=epoch)),
                                final)
                    depth_estimation_model.train()
                utils.write_event(log, step, loss=mean_loss, loss_depth_consistency=np.mean(depth_consistency_losses),
                                  loss_sparse_opt=np.mean(sparse_opt_losses))
            utils.write_event(log, step, loss=mean_loss, loss_depth_consistency=np.mean(depth_consistency_losses),
                              loss_sparse_opt=np.mean(sparse_opt_losses))
            tq.close()
            model_path_epoch = model_root / 'model_down_{down}_batch_{batch}_depth_{depth}_base_{base}_range_{range1}_' \
                                            '{range2}_sparse_opt_{opt_weight}_' \
                                            'inliner_{inlier}_hsv_{hsv}_view_indexes_{use}_' \
                                            'epoch_{epoch}.pt'.format(batch=batch_size,
                                                                      down=downsampling,
                                                                      depth=net_depth,
                                                                      base=filter_base,
                                                                      range1=adjacent_range[0],
                                                                      range2=adjacent_range[1],
                                                                      opt_weight=sparse_opt_weight,
                                                                      inlier=inlier_percentage,
                                                                      hsv=is_hsv,
                                                                      use=use_view_indexes_per_point,
                                                                      epoch=epoch)
            utils.save_model(model=depth_estimation_model, optimizer=optimizer, epoch=epoch + 1, step=step,
                             model_path=model_path_epoch)
            utils.save_model(model=depth_estimation_model, optimizer=optimizer, epoch=epoch + 1, step=step,
                             model_path=model_path)
        except KeyboardInterrupt:
            tq.close()

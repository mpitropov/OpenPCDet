import glob
import os

import torch
import tqdm
from torch.nn.utils import clip_grad_norm_
import numpy as np

def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        if batch['batch_size'] % 3 != 0:
            print("batch has not indivisble by 3. Skip this batch size", batch['batch_size'])
            break

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        # MIMO voxel combining

        # Store voxel information
        voxels = []
        voxel_coords = []
        voxel_num_points = []
        # Keep track of the batch ID when it switches
        curr_batch_id = 0
        new_batch_id = 0
        # Create a dictionary with key= (x_coord,y_coord) value= new array index
        voxel_coords_dict = {}
        for index, value in enumerate(batch['voxel_coords']):
            batch_id = value[0]
            z_coord = value[1] # This is always 0
            x_coord = value[2]
            y_coord = value[3]
            # Calculate the current head ID
            head_id = batch_id % 3
            # Detect if we have switched to a new batch
            if curr_batch_id != batch_id:
                curr_batch_id += 1
                # Detect if we have switched to new group of three point clouds
                if curr_batch_id % 3 == 0:
                    new_batch_id += 1

            xy_key = (x_coord, y_coord)
            if xy_key not in voxel_coords_dict:
                # Add the new index of where to find this voxel
                voxel_coords_dict[xy_key] = len(voxels)
                # Select only the valid points using number of points in voxel
                # Also add a new column for head ID
                num_points = batch['voxel_num_points'][index]
                new_column = np.full(num_points, head_id)
                new_voxel = np.hstack((batch['voxels'][index][:num_points],
                    np.atleast_2d(new_column).T))
                voxels.append(new_voxel)
                # Switch the batch number for the voxel coordinate
                new_voxel_coords = batch['voxel_coords'][index]
                new_voxel_coords[0] = new_batch_id
                voxel_coords.append(new_voxel_coords)
                # Keep the same number of points
                voxel_num_points.append(num_points)
            else:
                # Time to combine voxels!
                voxel_index = voxel_coords_dict[xy_key]
                # Select only the valid points using number of points in voxel
                # Also add a new column for head ID
                # except this time we have to concat with the prev points!
                num_points = batch['voxel_num_points'][index]
                new_column = np.full(num_points, head_id)
                new_voxel = np.hstack((batch['voxels'][index][:num_points],
                    np.atleast_2d(new_column).T))
                voxels[voxel_index] = np.concatenate((voxels[voxel_index],
                    new_voxel), axis=0)
                # We already have the coords added but we have to update the point count
                voxel_num_points[voxel_index] += num_points

        # Update the voxels in the batch dict
        new_max_points_in_voxel = np.max(voxel_num_points)
        # Create new numpy voxels with new max number of points per voxel!
        new_voxels = np.zeros((len(voxels), new_max_points_in_voxel, 5))
        for index, value in enumerate(voxels):
            num_points_in_voxel = len(value)
            new_voxels[index][:num_points_in_voxel] = value
        batch['voxels'] = new_voxels
        # Convert the other voxel information to np arrays
        batch['voxel_coords'] = np.array(voxel_coords)
        batch['voxel_num_points'] = np.array(voxel_num_points)


        loss, tb_dict, disp_dict = model_func(model, batch)

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)

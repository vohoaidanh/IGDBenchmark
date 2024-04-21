import os
import sys
import time
import comet_ml
import torch
import torch.nn
import argparse
from PIL import Image
from tensorboardX import SummaryWriter

from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions


"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    val_opt = get_val_opt()
    
    ################################################################
    # Create commet logs
    comet_ml.init(api_key='MS89D8M6skI3vIQQvamYwDgEc')
    comet_train_params = {
        'CropSize': opt.CropSize,
        'batch_size':opt.batch_size,
        'detect_method':opt.detect_method,
        'earlystop_epoch':opt.earlystop_epoch,
        'epoch_count':opt.epoch_count,
        'fix_backbone':opt.fix_backbone,
        'last_epoch':opt.last_epoch,
        'loadSize':opt.loadSize,
        'loss_freq':opt.loss_freq,
        'lr':opt.lr,
        'mode':opt.mode,
        'name':opt.name,
        'niter':opt.niter,
        'optim':opt.optim,
        'save_epoch_freq':opt.save_epoch_freq,
        'save_latest_freq':opt.save_latest_freq,
        'train_split':opt.train_split,
        'val_split':opt.val_split,
        'weight_decay':opt.weight_decay
        }
    
    experiment = comet_ml.Experiment(
        project_name="ai-generated-image-detection"
    )
    
    experiment.log_parameter('Train params', comet_train_params)
    ####################################################################

    data_loader = create_dataloader(opt)
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    model = Trainer(opt)
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)

            if model.total_steps % opt.save_latest_freq == 0:
                print('saving the latest model %s (epoch %d, model.total_steps %d)' %
                      (opt.name, epoch, model.total_steps))
                model.save_networks('latest')

            # print("Iter time: %d sec" % (time.time()-iter_data_time))
            # iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        # Validation
        model.eval()
        acc, ap, val_conf_mat  = validate(model.model, val_opt)[:3]
        
        TP = val_conf_mat[1, 1]
        TN = val_conf_mat[0, 0]
        FP = val_conf_mat[0, 1]
        FN = val_conf_mat[1, 0]
        
        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)
        
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))
        
        experiment.log_metric('val/epoch_acc', acc, epoch=epoch)
        experiment.log_metric('val/TPR', TPR, epoch=epoch)
        experiment.log_metric('val/TNR', TNR, epoch=epoch)
        
        file_name = "val_epoch_{}_{}.json".format(epoch, comet_train_params['name'])
        experiment.log_confusion_matrix(matrix = val_conf_mat, file_name=file_name, epoch=epoch)

        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                print("Early stopping.")
                experiment.end()
                break
        model.train()

    experiment.end()

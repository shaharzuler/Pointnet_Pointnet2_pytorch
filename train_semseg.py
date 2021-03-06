"""
Author: Benny
Date: Nov 2019
"""
import argparse
import datetime
import importlib
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import provider
from Pointnet_Pointnet2_pytorch.data_utils.PartCustomDataset.PartCsutomDataset import PartCustomDataset
from utils.LoggingUtils import TensorBoardHandler
from utils.VisualizationUtils import VisualizationUtils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ["1", "2", "3"]
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=1000, type=int, help='Epoch to run [default: 128]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--normal', action='store_true', default=True, help='Whether to use normal information [default: True]')
    parser.add_argument('--save_image_every', default=5, help='epoch frequency for saving images')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    TB_handler, checkpoints_dir, experiment_dir, images_dir, log_dir = create_dirs(args)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data/custom_partseg_data/'

    NUM_CLASSES = 2
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    print("start loading training data ...")
    TRAIN_DATASET = PartCustomDataset(root=root, npoints=NUM_POINT, split='train', normal_channel=args.normal, is_train=True, minimal_preprocess=True)
    print("start loading test data ...")
    TEST_DATASET = PartCustomDataset(root=root, npoints=NUM_POINT, split='val', normal_channel=args.normal, is_train=False, minimal_preprocess=False)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True,
                                                  num_workers=4)  # , pin_memory=True, drop_last=True, worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)  # , pin_memory=True, drop_last=True)
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    classifier, criterion = load_model_architecture(NUM_CLASSES, args, experiment_dir)
    classifier, start_epoch = load_model_weights(classifier, experiment_dir, log_string)
    optimizer = set_optimizer(args, classifier)

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0
    visualization_data = {}
    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        update_lr(LEARNING_RATE_CLIP, args, epoch, log_string, optimizer)
        momentum = update_momentum(MOMENTUM_DECCAY, MOMENTUM_DECCAY_STEP, MOMENTUM_ORIGINAL, epoch)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        loss_sum, total_correct, total_seen = reset_train_metrics()
        for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            classifier, loss_sum, total_correct, total_seen = batch_train_routine(NUM_CLASSES, NUM_POINT, classifier, criterion, data, loss_sum, optimizer, total_correct,
                                                                                  total_seen, weights)
        post_train_epoch_logging(TB_handler, checkpoints_dir, classifier, epoch, log_string, logger, loss_sum, num_batches, optimizer, total_correct, total_seen)

        '''Evaluate on chopped scenes'''
        if num_batches > 0:
            with torch.no_grad():
                num_batches = len(testDataLoader)
                labelweights, loss_sum, total_correct, total_correct_class, total_iou_deno_class, total_seen, total_seen_class = reset_eval_metrics(NUM_CLASSES, global_epoch,
                                                                                                                                                    log_string)
                for i, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                    classifier, labelweights, loss_sum, total_correct, total_seen = batch_evaluation_routine(BATCH_SIZE, NUM_CLASSES, NUM_POINT, classifier, criterion, data,
                                                                                                             labelweights, loss_sum, total_correct, total_correct_class,
                                                                                                             total_iou_deno_class, total_seen, total_seen_class, visualization_data,
                                                                                                             weights)
                post_evaluation_epoch_logging(NUM_CLASSES, TB_handler, args, best_iou, checkpoints_dir, classifier, epoch, images_dir, labelweights, log_string, logger, loss_sum,
                                              num_batches, optimizer, total_correct, total_correct_class, total_iou_deno_class, total_seen, total_seen_class, visualization_data)
        global_epoch += 1


def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum


def set_optimizer(args, classifier):
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)
    return optimizer


def load_model_weights(classifier, experiment_dir, log_string):
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)
    return classifier, start_epoch


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


def load_model_architecture(NUM_CLASSES, args, experiment_dir):
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet_util.py', str(experiment_dir))
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion = MODEL.get_loss().cuda()
    return classifier, criterion


def create_dirs(args):
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    TB_handler = TensorBoardHandler(experiment_dir)
    images_dir = experiment_dir.joinpath('images/')
    images_dir.mkdir(exist_ok=True)
    return TB_handler, checkpoints_dir, experiment_dir, images_dir, log_dir


def reset_train_metrics():
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    return loss_sum, total_correct, total_seen


def update_momentum(MOMENTUM_DECCAY, MOMENTUM_DECCAY_STEP, MOMENTUM_ORIGINAL, epoch):
    momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
    if momentum < 0.01:
        momentum = 0.01
    print('BN momentum updated to: %f' % momentum)
    return momentum


def update_lr(LEARNING_RATE_CLIP, args, epoch, log_string, optimizer):
    lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
    log_string('Learning rate:%f' % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def post_evaluation_epoch_logging(NUM_CLASSES, TB_handler, args, best_iou, checkpoints_dir, classifier, epoch, images_dir, labelweights, log_string, logger, loss_sum, num_batches,
                                  optimizer, total_correct, total_correct_class, total_iou_deno_class, total_seen, total_seen_class, visualization_data):
    if epoch % args.save_image_every == 0:
        VisualizationUtils().save_point_cloud_image(os.path.join(images_dir, "e_" + str(epoch) + ".png"),
                                                    visualization_data["points"],
                                                    visualization_data["seg_pred"],
                                                    visualization_data["target"])
    labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
    mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    TB_handler.write_loss("val", loss_sum / float(num_batches), epoch)
    log_string('eval point avg class IoU: %f' % (mIoU))
    TB_handler.write_mIoU("val", mIoU, epoch)
    log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
    TB_handler.write_acc("val", total_correct / float(total_seen), epoch)
    log_string('eval point avg class acc: %f' % (
        np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
    iou_per_class_str = '------- IoU --------\n'
    for l in range(NUM_CLASSES):
        iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
            seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
            total_correct_class[l] / float(total_iou_deno_class[l]))
    log_string(iou_per_class_str)
    log_string('Eval mean loss: %f' % (loss_sum / num_batches))
    log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))
    if mIoU >= best_iou:
        best_iou = mIoU
        logger.info('Save model...')
        savepath = str(checkpoints_dir) + '/best_model' + str(epoch) + '.pth'
        log_string('Saving at %s' % savepath)
        state = {
            'epoch': epoch,
            'class_avg_iou': mIoU,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, savepath)
        log_string('Saving model....')
    log_string('Best mIoU: %f' % best_iou)


def reset_eval_metrics(NUM_CLASSES, global_epoch, log_string):
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    labelweights = np.zeros(NUM_CLASSES)
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
    log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
    return labelweights, loss_sum, total_correct, total_correct_class, total_iou_deno_class, total_seen, total_seen_class


def batch_evaluation_routine(BATCH_SIZE, NUM_CLASSES, NUM_POINT, classifier, criterion, data, labelweights, loss_sum, total_correct, total_correct_class, total_iou_deno_class,
                             total_seen, total_seen_class, visualization_data, weights):
    points, target = data
    visualization_data["points"] = points.cpu().numpy()[0, :, :]
    visualization_data["target"] = target.cpu().numpy()[0, :]
    points = points.data.numpy()
    points = torch.Tensor(points)
    points, target = points.float().cuda(), target.long().cuda()
    points = points.transpose(2, 1)
    classifier = classifier.eval()
    seg_pred, trans_feat = classifier(points)
    pred_val = seg_pred.contiguous().cpu().data.numpy()
    visualization_data["seg_pred"] = np.argmax(seg_pred.contiguous()[0, :, :].view(-1, NUM_CLASSES).cpu().numpy(), axis=1)
    seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
    batch_label = target.cpu().data.numpy()
    target = target.view(-1, 1)[:, 0]
    loss = criterion(seg_pred, target, trans_feat, weights)
    loss_sum += loss
    pred_val = np.argmax(pred_val, 2)
    correct = np.sum((pred_val == batch_label))
    total_correct += correct
    total_seen += (BATCH_SIZE * NUM_POINT)
    tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
    labelweights += tmp
    for l in range(NUM_CLASSES):
        total_seen_class[l] += np.sum((batch_label == l))
        total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
        total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))
    return classifier, labelweights, loss_sum, total_correct, total_seen


def post_train_epoch_logging(TB_handler, checkpoints_dir, classifier, epoch, log_string, logger, loss_sum, num_batches, optimizer, total_correct, total_seen):
    log_string('Training mean loss: %f' % (loss_sum / float(num_batches)))
    TB_handler.write_loss("train", loss_sum / float(num_batches), epoch)
    log_string('Training accuracy: %f' % (total_correct / float(total_seen)))
    TB_handler.write_acc("train", total_correct / float(total_seen), epoch)
    if epoch % 5 == 0:
        logger.info('Save model...')
        savepath = str(checkpoints_dir) + '/model.pth'
        log_string('Saving at %s' % savepath)
        state = {
            'epoch': epoch,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, savepath)
        log_string('Saving model....')


def batch_train_routine(NUM_CLASSES, NUM_POINT, classifier, criterion, data, loss_sum, optimizer, total_correct, total_seen, weights):
    points, target = data
    B, N, D = points.shape  # fixed issue of calc metrics of batch size smaller than planned
    points = points.data.numpy()
    points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
    points = torch.Tensor(points)
    points, target = points.float().cuda(), target.long().cuda()
    points = points.transpose(2, 1)
    optimizer.zero_grad()
    classifier = classifier.train()
    seg_pred, trans_feat = classifier(points)
    seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
    batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
    target = target.view(-1, 1)[:, 0]
    loss = criterion(seg_pred, target, trans_feat, weights)
    loss.backward()
    optimizer.step()
    pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
    correct = np.sum(pred_choice == batch_label)
    total_correct += correct
    total_seen += (B * NUM_POINT)
    loss_sum += loss
    return classifier, loss_sum, total_correct, total_seen


if __name__ == '__main__':
    args = parse_args()
    main(args)

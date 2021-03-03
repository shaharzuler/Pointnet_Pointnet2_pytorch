import argparse
import os
from utils.VisualizationUtils import VisualizationUtils
from PostProcess.PostProcess import KMeansPostProcessor

import torch
import logging
from pathlib import Path
import sys
import importlib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ["1", "2"]
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--npoints', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--log_dir', type=str, default='2021-03-02_01-18', help='Experiment root')
    parser.add_argument('--point_cloud_path', type=str,
                        default="data/test_data/SCHUNK-39318568 PGN-plus-P 125-1-V, 000.txt",
                        help='point cloud for inference')
    parser.add_argument('--use_gpu', type=bool, default='0', help='0 to run inference on cpu')
    return parser.parse_args()


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def inference(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)
    prediction_dir = os.path.join(experiment_dir, "predictions")
    Path(prediction_dir).mkdir(exist_ok=True)
    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = 2

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES)
    if args.use_gpu == "1":
        classifier = classifier.cuda()
    device = 'gpu' if args.use_gpu == "1" else 'cpu'
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    with torch.no_grad():

        log_string('---- EVALUATION WHOLE SCENE----')
        point_set = np.loadtxt(args.point_cloud_path).astype(np.float32)
        original_shape = point_set.shape[0]
        if point_set.shape[1] == 7:  # case argb format
            point_set = np.delete(point_set, 3, axis=1)  # remove a column

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        point_set[:, 3:] /= 255

        downsample_ind = np.random.choice(point_set.shape[0], args.npoints, replace=True)
        point_set = point_set[downsample_ind, :]

        point_set = point_set.transpose()
        point_set = np.expand_dims(point_set, axis=0)
        tensor_point_cloud = torch.tensor(point_set)
        if args.use_gpu == "1":
            tensor_point_cloud = tensor_point_cloud.cuda()
        seg_pred, _ = classifier(tensor_point_cloud)

        pred_labels = np.argmax(seg_pred.contiguous().cpu().data[0, :, :].numpy(), 1)
        point_set = point_set[0, :, :].transpose()
        VisualizationUtils().save_point_cloud_image(
            os.path.join(experiment_dir + '/images', "e_" + args.point_cloud_path.split("/")[-1].replace(".txt", "") + "_inference" + ".png"),
            point_set,
            pred_labels,
            None)

        return downsample_ind, point_set, pred_labels, original_shape, experiment_dir, prediction_dir


def kmeans(point_set, pred_labels):
    pred_labels_3_classes = KMeansPostProcessor().cluster_mobile_links(point_set, pred_labels)
    return pred_labels_3_classes


def upsample(downsample_ind, pred_labels_3_classes, original_shape, prediction_dir):
    pred_labels_3_classes += 1
    results = np.zeros([original_shape])
    results[downsample_ind] = pred_labels_3_classes
    np.savetxt(os.path.join(prediction_dir, "results_" + args.point_cloud_path.split("/")[-1]), results.astype(np.int32), fmt='%i')

    return results


if __name__ == '__main__':
    args = parse_args()
    downsample_ind, point_set, pred_labels, original_shape, experiment_dir, prediction_dir = inference(args)
    pred_labels_3_classes = kmeans(point_set, pred_labels)
    VisualizationUtils().save_point_cloud_image(os.path.join(experiment_dir + '/images', "e3_" + args.point_cloud_path.split("/")[-1].replace(".txt", "") + "_inference" + ".png"),
                                                point_set,
                                                pred_labels_3_classes,
                                                None)
    results = upsample(downsample_ind, pred_labels_3_classes, original_shape, prediction_dir)

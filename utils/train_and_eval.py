import time
import os
from tqdm import tqdm
from pprint import PrettyPrinter

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from PIL import Image, ImageDraw

from utils.utils import AverageMeter, calculate_mAP
from models.backbones import MobileNetV2, MobileNetV1
from models.SSD import SSD, MultiBoxLoss

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Keep these hyperparams static
print_freq = 25
grad_clip = 1 # TODO: arbitrary... mess with this
momentum = 0.9
cudnn.benchmark = True


def train_and_eval(config, train_loader, test_loader):

    device = config["device"]
    backbone = config["backbone"]
    imsize = config["input_dims"]
    classes = config["classes"]  # to pass into eval, for use in calculate_map
    n_classes = len(config["classes"]) + 1  # +1 to account for the background
    lr = config["learning_rate"]
    weight_decay = config["weight_decay"]
    epochs = config["epochs"]
    save_results = config["save_results"]
    save_results_path = config["save_results_path"]
    detection_threshold = config["detection_threshold"]

    if save_results and not os.path.exists(save_results_path):
        os.mkdir(os.path.join(save_results_path))

    model = SSD(config["backbone_model"], device, n_classes=n_classes).to(device)  
        
    # optimizer = torch.optim.SGD(params=get_params_list(model, lr),
    #                             lr=lr,
    #                             momentum=momentum,
    #                             weight_decay=weight_decay)
    optimizer = torch.optim.Adam(params=get_params_list(model, lr), lr=lr)

    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy, device=device).to(device)

    # Epochs
    for epoch in range(epochs):

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              device=device)

        evaluate(test_loader, model, classes, device, save_results,
                 save_results_path, epoch, detection_threshold)


def train(train_loader, model, criterion, optimizer, epoch, device):
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels, _, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        optimizer.step()

        losses.update(loss.detach().item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


def get_params_list(model, learning_rate):
    biases = list()
    not_biases = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)

    return [{
        'params': biases,
        'lr': 2 * learning_rate
    }, {
        'params': not_biases
    }]


def evaluate(test_loader, model, classes, device, save_results,
             save_results_path, epoch, detection_threshold):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list(
    )  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():
        # Batches
        k = 0  # For inaming images
        for i, (images, boxes, labels, difficulties,
                fnames) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(
                predicted_locs,
                predicted_scores,
                min_score=0.001,
                max_overlap=0.45,
                top_k=100)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

            if save_results and epoch % 5 == 0:

                for fname, batch_boxes, batch_scores in list(
                        zip(fnames, det_boxes_batch, det_scores_batch)):

                    im = Image.open(fname)
                    w = im.width
                    h = im.height

                    for non_scaled_box, score in list(
                            zip(batch_boxes, batch_scores)):
                        if device != 'cpu':
                            score = float(score.cpu().numpy())
                            non_scaled_box = non_scaled_box.cpu().numpy()
                        else:
                            non_scaled_box = non_scaled_box.numpy()
                            score = float(score.numpy())

                        if score > detection_threshold:
                            scaled_box = [
                                non_scaled_box[0] * w, non_scaled_box[1] * h,
                                non_scaled_box[2] * w, non_scaled_box[3] * h
                            ]

                            imdraw = ImageDraw.Draw(im)
                            imdraw.rectangle(scaled_box,
                                             fill=None,
                                             outline=None,
                                             width=1)

                    pth = os.path.join(save_results_path, "epoch" + str(epoch))
                    if not os.path.exists(pth):
                        os.mkdir(pth)
                    im.save(f"{pth}/{k}.jpg")
                    k += 1

        # Calculate mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes,
                                 true_labels, true_difficulties, classes,
                                 device)

    # Print AP for each class
    pp.pprint(APs)
    del det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, predicted_locs, predicted_scores
    print('\nMean Average Precision (mAP): %.3f' % mAP)

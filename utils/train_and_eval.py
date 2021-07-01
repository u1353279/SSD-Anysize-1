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

    for epoch in range(epochs):

        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              device=device)

        evaluate(test_loader, model, classes, device, save_results,
                     save_results_path, epoch, detection_threshold)


def train(train_loader, model, criterion, optimizer, epoch, device):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    for i, (images, boxes, labels, _, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        predicted_locs, predicted_scores = model(images)

        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        optimizer.step()

        losses.update(loss.detach().item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

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

    return [{'params': biases, 'lr': 2 * learning_rate}, 
            {'params': not_biases}]


def evaluate(test_loader, model, classes, device, save_results,
             save_results_path, epoch, detection_threshold):

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  
        
    with torch.no_grad():
        # Batches
        k = 0  # For naming images
        for i, (images, boxes, labels, difficulties, fnames) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, 
            # top_k=200 for fair comparision with the paper's results and other repos
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(
                predicted_locs,
                predicted_scores,
                min_score=0.01,
                max_overlap=0.45,
                top_k=100)

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
                for fname, batch_boxes, batch_scores, batch_true_boxes in list(
                        zip(fnames, det_boxes_batch, det_scores_batch, boxes)):

                    im = Image.open(fname)

                    outline_color = (255,0,0) 
                    if im.mode == "L":
                        outline_color = (255)

                    w = im.width
                    h = im.height
                    
                    imdraw = ImageDraw.Draw(im)
                    for non_scaled_box, score in list(zip(batch_boxes, batch_scores)):
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

                            imdraw.rectangle(scaled_box,
                                             fill=None,
                                             outline=outline_color,
                                             width=3)
                            
                    for non_scaled_box in batch_true_boxes:
                        
                        scaled_box = [
                                non_scaled_box[0] * w, non_scaled_box[1] * h,
                                non_scaled_box[2] * w, non_scaled_box[3] * h
                            ]

                        imdraw.rectangle(scaled_box,
                                             fill=None,
                                             outline=outline_color,
                                             width=3)
                        
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

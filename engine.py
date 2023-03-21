import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

# -----------------------------------------------------------------------------------------------
#def intersection_boxes(annotated_box, predicted_box):
#    x = max(annotated_box[0], predicted_box[0])
#    xw = min((annotated_box[0] + annotated_box[2]), (predicted_box[0] + predicted_box[2]))
#    y = max(annotated_box[1], predicted_box[1])
#    yh = min((annotated_box[1] + annotated_box[3]), (predicted_box[1] + predicted_box[3]))
#    w = xw - x
#    h = yh - y
#
#    inter_area = w * h
#    ann_area = annotated_box[2] * annotated_box[3]
#    pre_area = predicted_box[2] * predicted_box[3]
#
#    inter_percent = inter_area / (ann_area + pre_area - inter_area)
#
#    return inter_percent


#def calculate_accuracy(outputs, writer, targets, step, mode, e_step):
#    
#    count = 0
#    num = 0
#    
#    for index, prediction in enumerate(outputs):       
#        annot_boxes = targets[index]['boxes'].tolist()
#        annot_labels = targets[index]['labels'].tolist()
#        predict_boxes = prediction['boxes'].tolist()
#        predict_labels = prediction['labels'].tolist()
#
#        num = num + len(predict_boxes)
#        
#        # Compare Each Annotated Box with Each Predicted Box
#        for ann_index, ann_box in enumerate(reversed(annot_boxes)):
#            ann_label = annot_labels[ann_index]
#            
#            for pre_index, pre_box in enumerate(reversed(predict_boxes)):
#                pre_label = predict_labels[pre_index]
#                intersection = intersection_boxes(ann_box, pre_box)
#
#                if intersection > 0.6 and ann_label == pre_label:
#                    count = count + 1
#                    #annot_boxes.remove(ann_box)
#                    #predict_boxes.remove(pre_box)
#
#    if num != 0:
#        accuracy = count / num
#    else:
#        accuracy = 0
#        
#    if mode == 'train':
#        #print(f'Plotting Training Accuracy: ({accuracy} , {step})')
#        writer.add_scalars('Training vs Validation Accuracy', {'Training': accuracy}, step)
#
#    else:
#        #print(f'Plotting Validation Accuracy: ({accuracy} , {e_step})')
#        writer.add_scalars('Training vs Validation Accuracy', {'Validation': accuracy}, e_step)


def calculate_loss(model, data_loader, device, scaler=None):
    model.train()

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        return losses_reduced
        
# -----------------------------------------------------------------------------------------------

def train_one_epoch(step, writer, model, optimizer, data_loader, device, epoch, print_freq, scaler=None):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # -----------------------------------------------------------------------------------------------
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            # -----------------------------------------------------------------------------------------------
            # loss_dict, outputs = model(images, targets)
            # losses = sum(loss for loss in loss_dict.values())
            # -----------------------------------------------------------------------------------------------

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        
# ------------------------------------------------------------------------------------------------------
        # Add the loss and accuracy to the writer for graphing
        
        #mode = 'train'
        #calculate_accuracy(outputs, writer, targets, step, mode, epoch)

        training_loss = losses_reduced
        writer.add_scalars('Training vs Validation Loss', {'Training': training_loss}, step)
        #print(f'Plotting Training Loss: ({training_loss},{step})')

        step = step + 1
# ------------------------------------------------------------------------------------------------------

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger




def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types




@torch.inference_mode()
def evaluate(model, data_loader, device, scaler=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()

# -----------------------------------------------------------------------------------------------
        outputs = model(images)
# -----------------------------------------------------------------------------------------------
#        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#        with torch.cuda.amp.autocast(enabled=scaler is not None):
#            loss_dict, outputs = model(images, targets)
#        
#        loss_dict_reduced = utils.reduce_dict(loss_dict)
#        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
# -----------------------------------------------------------------------------------------------

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    
    
# ------------------------------------------------------------------------------------------------------
    # Add the loss and accuracy to the writer for graphing
#    mode = 'eval'
#    e_step = ((epoch + 1) * dl_len) - 1
#    calculate_accuracy(outputs, writer, targets, epoch, mode, e_step)

#    validation_loss = losses_reduced
#    e_step = ((epoch + 1) * dl_len) - 1
#    writer.add_scalars('Training vs Validation Loss', {'Validation': validation_loss}, e_step)
# ------------------------------------------------------------------------------------------------------
        
    return coco_evaluator

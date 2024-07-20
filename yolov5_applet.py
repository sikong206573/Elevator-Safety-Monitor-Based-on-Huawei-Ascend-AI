import sys, os, time
import numpy as np
import cv2 as cv
import torch
import pickle
import zstd
from torchvision.ops import nms

from inferemote.atlas_remote import AtlasRemote
from inferemote.image_encoder import ImageEncoder
from inferemote.inference_job import InferenceJob
from inferemote.testing.data_source import ImageFactory


labels=['fall', 'stand','sit','motorbike','smoke']
MODEL_WIDTH = 640
MODEL_HEIGHT = 640
input_size=(MODEL_WIDTH,MODEL_HEIGHT)
OUTPUT_DIR = './__OUTPUTS__/'
conf_thres=0.3
iou_thres=0.45
red_color = (0, 0, 255)
blue_color = (255, 0, 0)

class Yolov5(AtlasRemote):

    def __init__(self, **kwargs):
        super().__init__(port=5602, **kwargs)
    
    def pre_process(self,image):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        target_size=(MODEL_WIDTH,MODEL_HEIGHT)
        old_size = image.shape[0:2]
        ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))
        new_size = tuple([int(i * ratio) for i in old_size])
        img_new = cv.resize(image, (new_size[1], new_size[0]))
        pad_w = target_size[1] - new_size[1]
        pad_h = target_size[0] - new_size[0]
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        resized_img = cv.copyMakeBorder(img_new, top, bottom, left, right, cv.BORDER_CONSTANT, None, (0, 0, 0))
        resized_img=resized_img.astype(np.float32)
        blob = ImageEncoder.image_to_bytes(resized_img)
        # new_image = resized_img / 255
        # new_image = new_image.transpose(2, 0, 1).copy()
        return blob
    
    def post_process(self, result):
        print("111")
        result = zstd.decompress(result)
        result = pickle.loads(result[0])
        pred = np.frombuffer(bytearray(result[0]), dtype=np.float32)

        return pred
    
    
    
def start(inference_func, file_path):
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    
    cap = cv.VideoCapture(file_path)
    output_file = os.path.join(OUTPUT_DIR, "out_" + os.path.basename(file_path))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv.VideoWriter(output_file, fourcc, 20.0, (frame_width, frame_height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        label_count=0
        
        result_list = inference_func(frame)
        pred = result_list.reshape(1, 25200, -1)
        pred = torch.tensor(pred)
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=1000)
        s = ""
        boxes = []
        annos = []
        names = []
        scores = []
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(input_size, det[:, :4], frame.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {labels[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    name = labels[c]
                    box = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    score = float(conf)

                    boxes.append(box)
                    names.append(name)
                    scores.append(score)
                    annos.append(c)
        out_img = frame.copy()
        if len(boxes) > 0:
            label_count += 1
            out_img = draw_box(out_img, boxes, names, scores)
            
            out.write(out_img)
    out.release()
    cap.release()

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def draw_box(image, boxes, names, scores, show_label=True):
    image_h, image_w, _ = image.shape

    for i, box in enumerate(boxes):
        box = np.array(box[:4], dtype=np.int32)  # xyxy

        line_width = int(3)
        txt_color = (255, 255, 255)
        box_color = (58, 56, 255)

        p1, p2 = (box[0], box[1]), (box[2], box[3])
        image = cv.rectangle(image, p1, p2, box_color, line_width)
        if show_label:
            tf = max(line_width - 1, 1)  # font thickness
            box_label = '%s: %.2f' % (names[i], scores[i])
            w, h = cv.getTextSize(box_label, 0, fontScale=line_width / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3

            image = cv.rectangle(image, p1, p2, box_color, -1, cv.LINE_AA)  # filled
            image = cv.putText(image, box_label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0,
                                line_width / 3, txt_color, thickness=tf, lineType=cv.LINE_AA)
            if names[i] in ["fall","smoke","motorbike"]:
                warning_text = "WARNING"
                text_size, _ = cv.getTextSize(warning_text, cv.FONT_HERSHEY_SIMPLEX, 3, 5)
                text_position = (10, int(text_size[1] + 10))
                cv.putText(image, warning_text, text_position, cv.FONT_HERSHEY_SIMPLEX, 3, red_color, 5, cv.LINE_AA)
                image = cv.rectangle(image, p1, p2, box_color, -1, cv.LINE_AA)  # filled
                image = cv.putText(image, box_label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0,
                                line_width / 3, txt_color, thickness=tf, lineType=cv.LINE_AA)
            else:
                image = cv.rectangle(image, p1, p2, blue_color, -1, cv.LINE_AA)  # filled
                image = cv.putText(image, box_label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0,
                                    line_width / 3, txt_color, thickness=tf, lineType=cv.LINE_AA)
    return image

if __name__ == '__main__':
    model = Yolov5()
    model.use_remote(sys.argv[1])
   
    start(model.inference_remote, sys.argv[2])
    
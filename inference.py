import cv2
import time
import random
import numpy as np
import onnxruntime as ort
from PIL import Image

cuda = False
weights = "runs/train/yolov7-tiny-mast/weights/best.onnx"
img = cv2.imread("image.jpg")

providers = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if cuda
    else ["CPUExecutionProvider"]
)
session = ort.InferenceSession(weights, providers=providers)


def letterbox(
    im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, r, (dw, dh)


names = ["mast"]
colors = {"mast": [255, 0, 0]}
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

image = img.copy()
# im = cv2.copyMakeBorder(
#     image,
#     (640 - 480) // 2,
#     (640 - 480) // 2,
#     0,
#     0,
#     cv2.BORDER_CONSTANT,
#     value=(114, 114, 114),
# )
image, ratio, dwdh = letterbox(image, auto=False)


image = image.transpose((2, 0, 1))
image = np.expand_dims(image, 0)
image = np.ascontiguousarray(image)

im = image.astype(np.float32)
im /= 255
im.shape

outname = [i.name for i in session.get_outputs()]
outname

inname = [i.name for i in session.get_inputs()]
inname

inp = {inname[0]: im}


outputs = session.run(outname, inp)[0]
print(outputs)

ori_images = [img.copy()]

for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
    image = ori_images[int(batch_id)]
    box = np.array([x0, y0, x1, y1])
    box -= np.array(dwdh * 2)
    box /= ratio
    box = box.round().astype(np.int32).tolist()
    cls_id = int(cls_id)
    score = round(float(score), 3)
    name = names[cls_id]
    color = colors[name]
    name += " " + str(score)
    cv2.rectangle(image, box[:2], box[2:], color, 2)
    cv2.putText(
        image,
        name,
        (box[0], box[1] - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        [225, 255, 255],
        thickness=2,
    )

Image.fromarray(ori_images[0]).save("thing.jpg")

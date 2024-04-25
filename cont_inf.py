import cv2
import numpy as np
import onnxruntime as ort

cuda = False
weights = "runs/train/yolov7-tiny-mast/weights/best.onnx"

providers = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if cuda
    else ["CPUExecutionProvider"]
)
session = ort.InferenceSession(weights, providers=providers)

names = ["mast"]


def infer(image):
    image = cv2.copyMakeBorder(
        image,
        (640 - 480) // 2,
        (640 - 480) // 2,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )[..., :3]

    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im /= 255

    outname = [i.name for i in session.get_outputs()]
    inname = [i.name for i in session.get_inputs()]
    inp = {inname[0]: im}

    outputs = session.run(outname, inp)[0]

    for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
        print(x0, y0, x1, y1, names[cls_id])

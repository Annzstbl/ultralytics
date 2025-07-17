import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import os

def find_last_conv_layer(model):
    """
    Traverse model to find the last nn.Conv2d layer for Grad-CAM.
    """
    conv_layers = [m for m in model.model.modules() if isinstance(m, torch.nn.Conv2d)]
    return conv_layers[-1]  # last convolutional layer

def generate_detection_heatmaps(
    weights_path: str,
    npy_path: str,
    output_dir: str,
    imgsz: int = 1280
):
    """
    Run YOLO11-OBB inference and generate a heatmap per detection using Grad-CAM.
    Saves one heatmap image per detected object.
    """
    # Prepare output
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = YOLO(weights_path)
    model.model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.model.to(device)

    # Load input and preprocess
    arr = np.load(npy_path)  # [H, W, C] for RGB or [H, W, C] for multi-band
    img = cv2.resize(arr, (imgsz, imgsz))
    img_norm = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm).permute(2,0,1).unsqueeze(0).to(device)

    # Run inference
    results = model(img_tensor, imgsz=imgsz)
    boxes = results[0].obb.xyxyxyxy.cpu().numpy()  # [N,8]
    classes = results[0].obb.cls.cpu().numpy().astype(int)
    scores = results[0].obb.conf.cpu().numpy()

    if len(boxes) == 0:
        print("No detections found.")
        return

    # Set up GradCAM
    target_layer = find_last_conv_layer(model)
    cam = GradCAM(model=model.model, target_layers=[target_layer], use_cuda=device.type == 'cuda')

    # For each detection, compute heatmap
    for i, (box, cls) in enumerate(zip(boxes, classes)):
        # Set target for class
        target = ClassifierOutputTarget(int(cls))
        # Generate cam
        grayscale_cam = cam(input_tensor=img_tensor, targets=[target])[0]  # [H, W]
        cam_image = show_cam_on_image(img_norm[..., ::-1], grayscale_cam, use_rgb=True)

        # Overlay bounding box
        pts = box.reshape(4,2).astype(int)
        overlay = cam_image.copy()
        cv2.polylines(overlay, [pts], isClosed=True, color=(0,255,0), thickness=2)
        heatmap_path = os.path.join(output_dir, f"detection_{i}_cls{cls}_score{scores[i]:.2f}.png")
        cv2.imwrite(heatmap_path, overlay[..., ::-1])  # RGB -> BGR for OpenCV
        print(f"Saved heatmap: {heatmap_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps per detection")
    parser.add_argument("--weights", default="/data/users/litianhao/hsmot_code/workdir/yolo11/yolov11l_8ch_CocoPretrain_CopyFirstConv_imgsize1280_4gpu/weights/last.pt", help="Path to YOLO weights (.pt)")
    parser.add_argument("input_npy", "/data/users/wangying01/lth/data/hsmot/npy/data24-1/000001.npy", help="Input .npy image path")
    parser.add_argument("output_dir", "/data3/litianhao/hsmot/paper", help="Directory to save heatmaps")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    args = parser.parse_args()

    generate_detection_heatmaps(
        args.weights,
        args.input_npy,
        args.output_dir,
        imgsz=args.imgsz
    )


# python gen_detection_heatmaps.py \
#   /path/to/yolov11l_8ch.pt \
#   data/test_video/frame_0001.npy \
#   outputs/heatmaps \
#   --imgsz 1280
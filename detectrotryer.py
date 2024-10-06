import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os

video_path = r'c:\Users\lando\Videos\raw\testpipe.mp4' 
output_dir = r'c:\Users\lando\Desktop\processedframes\testpipe'  

# Setup configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # Set threshold
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create predictor
predictor = DefaultPredictor(cfg)

# Function to process a frame
def process_frame(frame):
    outputs = predictor(frame)
    masks = outputs["instances"].pred_masks.cpu().numpy()
    
    if len(masks) > 0:
        areas = [mask.sum() for mask in masks]
        largest_mask = masks[areas.index(max(areas))]
        silhouette = (largest_mask * 255).astype('uint8')
        return silhouette
    return None

# Read video frames
cap = cv2.VideoCapture(video_path)
frame_count = 0
with ThreadPoolExecutor() as executor:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Submit the frame for processing
        future = executor.submit(process_frame, frame)
        
        # Get the result from the future (this will block until the frame is processed)
        silhouette = future.result()
        
        if silhouette is not None:
            # Display or save the silhouette
            print(frame_count)
            cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count}.png"), silhouette)

            
        frame_count += 1
cap.release()
cv2.destroyAllWindows()

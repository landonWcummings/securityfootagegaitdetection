import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import os

video_path = r'c:\Users\lando\Videos\raw\testpipe.mp4' 
output_dir = r'c:\Users\lando\Desktop\processedframes\testpipe'  

model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Load YOLOv5 model

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


video_name = os.path.basename(video_path).split(".")[0]  # Get the base name without extension
video_output_dir = os.path.join(output_dir, video_name)  # New folder path
if not os.path.exists(video_output_dir):
    os.makedirs(video_output_dir)  # Create the new folder

# Setup configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # Set threshold
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create predictor
predictor = DefaultPredictor(cfg)

confidence_min = 0.62
frame_count = 0
consecutive_detections = 0
image_tuples = []
frame_boxes = []
output_segments = 0
save_threshold = 25
pixel_buffer = 50
check_frequency = 3
cut_ends = 4

# Function to process a frame
def process_frame(frame,bbox):
    x, y, x2, y2 = bbox
    cropped_frame = frame[y:y2, x:x2]
    
    # Get model prediction for the cropped region
    outputs = predictor(cropped_frame)    
    masks = outputs["instances"].pred_masks.cpu().numpy()
    
    if len(masks) > 0:
        areas = [mask.sum() for mask in masks]
        largest_mask = masks[areas.index(max(areas))]
        silhouette = (largest_mask * 255).astype('uint8')
        
        
        canvas = np.zeros(frame.shape[:2], dtype='uint8')
        
        # Place the silhouette back on the original frame size in the position of the bbox
        canvas[y:y2, x:x2] = silhouette
        canvasfinal = cv2.resize(canvas, (64, 64), interpolation=cv2.INTER_AREA)
        _, final = cv2.threshold(canvasfinal, 127, 255, cv2.THRESH_BINARY)
        return final
    return None
def generate_silhouettes(base):
    i = 0
    for tframe, tname in image_tuples:
        frame_file = os.path.join(base, tname)
        boxes = frame_boxes[i//check_frequency][1]
        for box in boxes:
            x1, y1, x2, y2 = [int(coord.item()) for coord in box]
        height, width = frame.shape[:2]
        x1 = max(0, x1 - pixel_buffer)  # Ensure x1 is not less than 0
        y1 = max(0, y1 - pixel_buffer)  # Ensure y1 is not less than 0
        x2 = min(width, x2 + pixel_buffer)  # Ensure x2 does not exceed image width
        y2 = min(height, y2 + pixel_buffer)  
        box = (x1,y1,x2,y2)
        silhouette = process_frame(tframe,box)
        if silhouette is not None:
            cv2.imwrite(frame_file, silhouette)
        i += 1

def savebox(path):
    data = []
    for frame_num, boxes in frame_boxes:
        for box in boxes:
            x1, y1, x2, y2 = [coord.item() for coord in box]
            data.append([frame_num, x1, y1, x2, y2])


    # Create a DataFrame and save to CSV
    df = pd.DataFrame(data, columns=["Frame Number", "x1", "y1", "x2", "y2"])
    df.to_csv(path, index=False)
def reset():
    global image_tuples, consecutive_detections, frame_boxes
    image_tuples = []
    consecutive_detections = 0
    frame_boxes = []
# Read video frames
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#with ThreadPoolExecutor() as executor:
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    if frame_count % check_frequency ==0:
        results = model(frame)

        # Check for detections
        detections = results.xyxy[0]  # Get detections from results
        person_detected = False
        onebox = []
        for *box, conf, cls in detections:
            if model.names[int(cls)] == 'person' and conf > confidence_min:
                person_detected = True
                onebox.append(box)
                frame_boxes.append((frame_count,onebox))
                break

        if person_detected:
            consecutive_detections += 1  # Increment count of consecutive detections
            image_tuple = (frame, f'{video_name}_frame_{frame_count:04d}.jpg')
            image_tuples.append(image_tuple)
        else:
            if consecutive_detections >= save_threshold:
                # Create a tuple with the image and a title
                frame_fi = os.path.join(video_output_dir, str(output_segments))
                if not os.path.exists(frame_fi):
                    os.makedirs(frame_fi)
                image_tuples = image_tuples[cut_ends:-cut_ends] #cut last few images in list
                generate_silhouettes(frame_fi)
                #for tframe, tname in image_tuples:
                    
                    #frame_file = os.path.join(frame_fi, tname)

                    #cv2.imwrite(frame_file, tframe)
                    
                output_segments += 1
                savebox(os.path.join(frame_fi, 'boxes'))
            reset()
    else:
        consecutive_detections += 1  
        image_tuple = (frame, f'{video_name}_frame_{frame_count:04d}.jpg')
        image_tuples.append(image_tuple)

    

    # If consecutive detections exceed the threshold, save the frame
    if frame_count == total_frames-1 and consecutive_detections >= save_threshold:
        frame_fi = os.path.join(video_output_dir, str(output_segments))
        if not os.path.exists(frame_fi):
            os.makedirs(frame_fi)
        image_tuples = image_tuples[cut_ends:]
        generate_silhouettes(frame_fi)
        savebox(os.path.join(frame_fi, 'boxes'))
        reset()

        

    frame_count += 1
cap.release()
cv2.destroyAllWindows()



"""

silhouette = future.result()
    
    if silhouette is not None:
        # Display or save the silhouette
        print(frame_count)
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count}.png"), silhouette)
"""
import cv2
import torch
import os
import pandas as pd
import numpy as np

model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Load YOLOv5 model

video_path = r'c:\Users\lando\Videos\raw\testpipe.mp4' 
output_dir = r'c:\Users\lando\Desktop\processedframes'  

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


video_name = os.path.basename(video_path).split(".")[0]  # Get the base name without extension
video_output_dir = os.path.join(output_dir, video_name)  # New folder path
if not os.path.exists(video_output_dir):
    os.makedirs(video_output_dir)  # Create the new folder

# Read the video
confidence_min = 0.62
frame_count = 0
consecutive_detections = 0
image_tuples = []
frame_boxes = []
output_segments = 0
save_threshold = 20
check_frequency = 3
def generate_silhouette(base):
    i = 0
    for tframe, tname in image_tuples:
        frame_file = os.path.join(base, tname)

        # Create an empty black mask with the same dimensions as the frame
        mask = np.zeros_like(tframe)
        
        # Loop over each bounding box and fill the corresponding area on the mask
        
        if i < len(frame_boxes):
            boxes = frame_boxes[i//check_frequency][1]  # Get the list of boxes for the current frame

            # Loop over each bounding box and fill the corresponding area on the mask
            for box in boxes:
                x1, y1, x2, y2 = [int(coord.item()) for coord in box]  # Convert to integer pixel values
                human_image = frame[y1:y2, x1:x2]
                
                gray = cv2.cvtColor(human_image, cv2.COLOR_BGR2GRAY)
                _, silhouette = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

                highlighted_frame = cv2.bitwise_and(tframe, silhouette)


        # Resize the mask to 64x64 to create the silhouette (optional)
        #silhouette = cv2.resize(mask, (64, 64))

        # Save the silhouette image to the specified output path
        cv2.imwrite(frame_file, highlighted_frame)

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

cap = cv2.VideoCapture(video_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
while cap.isOpened():
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
                image_tuples = image_tuples[:-2] #cut last two images in list
                generate_silhouette(frame_fi)
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
        generate_silhouette(frame_fi)

        
        #for tframe, tname in image_tuples:
            #frame_file = os.path.join(frame_fi,box, tname)
            #cv2.imwrite(frame_file, tframe)
            #generate_silhouette(tframe,onebox,frame_file)

        
        savebox(os.path.join(frame_fi, 'boxes'))
        reset()

        

    frame_count += 1

cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close any OpenCV windows



#frame_file = os.path.join(video_output_dir, f'{video_name}_frame_{frame_count:04d}.jpg')
#cv2.imwrite(frame_file, frame)
    
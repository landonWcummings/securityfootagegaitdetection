import cv2
import torch
import os

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
cap = cv2.VideoCapture(video_path)

confidence_min = 0.62
frame_count = 0
consecutive_detections = 0
image_tuples = []
output_segments = 0
save_threshold = 20  # Minimum number of consecutive detections to start saving
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 3 ==0:
        results = model(frame)

        # Check for detections
        detections = results.xyxy[0]  # Get detections from results
        person_detected = False

        for *box, conf, cls in detections:
            if model.names[int(cls)] == 'person' and conf > confidence_min:
                person_detected = True
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
                for tframe, tname in image_tuples:
                    
                    frame_file = os.path.join(frame_fi, tname)

                    cv2.imwrite(frame_file, tframe)
                output_segments += 1

            image_tuples = []
            consecutive_detections = 0 
    else:
        consecutive_detections += 1  
        image_tuple = (frame, f'{video_name}_frame_{frame_count:04d}.jpg')
        image_tuples.append(image_tuple)

    

    # If consecutive detections exceed the threshold, save the frame
    if frame_count == total_frames-1 and consecutive_detections >= save_threshold:
        frame_fi = os.path.join(video_output_dir, str(output_segments))
        if not os.path.exists(frame_fi):
            os.makedirs(frame_fi)
        for tframe, tname in image_tuples:
            frame_file = os.path.join(frame_fi, tname)
            cv2.imwrite(frame_file, tframe)

        image_tuples = []
        consecutive_detections = 0

        frame_file = os.path.join(frame_fi, f'{video_name}_frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_file, frame)  # Save the frame
        

    frame_count += 1

cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close any OpenCV windows



#frame_file = os.path.join(video_output_dir, f'{video_name}_frame_{frame_count:04d}.jpg')
#cv2.imwrite(frame_file, frame)
    
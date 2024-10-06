import cv2

# Create a background subtractor object
backSub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=26, detectShadows=True)
video_path = r'c:\Users\lando\Videos\raw\testpipe.mp4' 
output_dir = r'c:\Users\lando\Desktop\processedframes'  
# Read the video frames
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the background subtractor to get the foreground mask
    fgMask = backSub.apply(frame)

    # Threshold the mask to get binary silhouette
    _, silhouette = cv2.threshold(fgMask, 137, 255, cv2.THRESH_BINARY)

    # Display or save the silhouette
    cv2.imshow('Silhouette', silhouette)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

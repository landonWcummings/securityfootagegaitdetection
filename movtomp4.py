from moviepy.editor import VideoFileClip

input_file = r'c:\Users\lando\Videos\unprocessed\test5.mov'
output_file = r'c:\Users\lando\Videos\raw\test5.mp4'

# Load the .mov file
clip = VideoFileClip(input_file)

# Resize while maintaining aspect ratio
# If you want to limit the width or height, you can use something like:
clip_resized = clip.resize(newsize=(1080,1920))

# Write the .mp4 file
clip_resized.write_videofile(output_file, codec='libx264')

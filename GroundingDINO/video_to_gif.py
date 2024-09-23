from moviepy.editor import VideoFileClip
import os

# Load the video
video_path = r"GroundingDINO\asset\outputs\HD CCTV Camera video 3MP 4MP iProx CCTV HDCCTVCameras.net retail store.mp4"
video = VideoFileClip(video_path)

# Set the new width and calculate the corresponding height to maintain the aspect ratio
new_width = 640  # Set the desired width
aspect_ratio = video.size[1] / video.size[0]  # height / width
new_height = int(new_width * aspect_ratio)

# Resize the video while keeping the aspect ratio
resized_video = video.resize(newsize=(new_width, new_height))

# Save the resized new video
output_path = "GroundingDINO/asset/outputs/1_"+os.path.basename(video_path)
resized_video.write_videofile(output_path)


# remove old video and give old video name to new video
os.remove(video_path)
os.rename(output_path,video_path)


# Load the video file
clip = VideoFileClip(video_path)
# Extract the segment from 35 to 45 seconds
clip = clip.subclip(35, 45)
# Convert to gif
clip.write_gif(video_path.replace(".mp4",".gif"))
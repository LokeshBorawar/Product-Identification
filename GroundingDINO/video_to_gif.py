from moviepy.editor import VideoFileClip

video_path=r"GroundingDINO\asset\outputs\HD CCTV Camera video 3MP 4MP iProx CCTV HDCCTVCameras.net retail store.mp4"
# Load the video file
clip = VideoFileClip(video_path)
# Extract the segment from 3 to 9 seconds
clip = clip.subclip(35, 45)
# Convert to gif
clip.write_gif(video_path.replace(".mp4",".gif"))
from moviepy.editor import VideoFileClip

def clip_video_to_gif(video_path, start_time, end_time, output_gif_path):
    # Load the video file
    video = VideoFileClip(video_path)

    # Set the new width and calculate the corresponding height to maintain the aspect ratio
    new_width = 640  # Set the desired width
    aspect_ratio = video.size[1] / video.size[0]  # height / width
    new_height = int(new_width * aspect_ratio)

    # Resize the video while keeping the aspect ratio
    resized_video = video.resize(newsize=(new_width, new_height))
    
    # Clip the video
    clip = resized_video.subclip(start_time, end_time)
    
    # Write the result to a GIF file
    clip.write_gif(output_gif_path)
    
    print(f"GIF saved as: {output_gif_path}")

# clip backgroundmatting + groundingdino
clip_video_to_gif(video_path='BackgroundMattingV2/outputs/processed_video.mp4', start_time=3, end_time=12, output_gif_path='BackgroundMattingV2/outputs/processed_video.gif')

# clip backgroundmatting + groundingdino
clip_video_to_gif(video_path='BackgroundMattingV2/outputs/boxed_product.mp4', start_time=3, end_time=12, output_gif_path='BackgroundMattingV2/outputs/groundingdino_matting_clip.gif')

# clip groundingdino
clip_video_to_gif(video_path='GroundingDINO/asset/outputs/HD CCTV Camera video 3MP 4MP iProx CCTV HDCCTVCameras.net retail store.mp4', start_time=3, end_time=12, output_gif_path='BackgroundMattingV2/outputs/groundingdino_clip.gif')
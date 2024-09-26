# Extracting product images after [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2.git)

- The [BGR-YOLO code](bgr_extraction.py) begins by preparing a background image through a process of detecting people within a given scene. It identifies areas of the image that do not contain any individuals and assigns those regions to a black background image. This approach results in a final image that is devoid of people, effectively isolating the [background](inputs/bgr.png). Importantly, if the YOLO model fails to detect any individuals, the background will have people.

- To ensure compatibility with the BackgroundMattingV2 model, a [resizing code](resize_video.py) is employed, adjusting the image dimensions so that they are divisible by four, which is a requirement for model input.

- With both the background and the video prepared, the [BackgroundMattingV2 model code (matting_demo.py)](matting_demo.py) can effectively process the footage. It blackens the background while retaining only the detected humans and any objects they carry, resulting in a visually [clean output](outputs/processed_video.mp4) that emphasizes the subjects of interest. 

- Finally, the [GroundingDINO model (product_extraction_demo.py)](product_extraction_demo.py) is set to detect objects that are held in the hands of individuals within the scene. This combination of technologies facilitates a sophisticated analysis of the video, enhancing the ability to track and understand human interactions with their environment.

- Output of [matting_demo.py](matting_demo.py)
  ![processed_video](outputs/processed_video.gif)
- Output of [matting + groundingdino](product_extraction_demo.py)
  ![groundingdino_matting_clip](outputs/groundingdino_matting_clip.gif)
- Output of [groundingdino](GroundingDINO/product_extract_demo.py)
  ![groundingdino_clip](outputs/groundingdino_clip.gif)
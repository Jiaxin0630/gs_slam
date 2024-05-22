import cv2
import os

def images_to_video(image_folder, video_name, fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith((".jpg", ".jpeg", ".png"))]
    images.sort() 

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    if frame is None:
        print("Failed to read the first image.")
        return

    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 'mp4v' 编码器
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height), True)

    for image in images:
        img = cv2.imread(os.path.join(image_folder, image))
        if img is None:
            print(f"Failed to read image {image}. Skipping.")
            continue
        video.write(img)

    video.release()
    cv2.destroyAllWindows()
    
    
images_to_video("/media/jiaxin/Jiaxin-usb/ma_final/GS-SLAM_JIAXIN_working/results/tum/rgbd_dataset_freiburg2_desk/rendering/render", "/media/jiaxin/Jiaxin-usb/ma_final/GS-SLAM_JIAXIN_working/results/tum/rgbd_dataset_freiburg2_desk/rendering/render.mp4")

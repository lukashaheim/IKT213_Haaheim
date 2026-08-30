import os.path
import cv2

def print_image_information(img):
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    print("height:", height)
    print("width:", width)
    print("channels:", channels)
    print("size:", img.size)
    print("data type:", img.dtype)

def print_camera_information(cam):
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = int(cam.get(cv2.CAP_PROP_FPS))

    output_dir = "./"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "camera_outputs.txt")

    with open(output_path, "w") as f:
        f.write(f"fps: {frame_fps}\n")
        f.write(f"height: {frame_height}\n")
        f.write(f"width: {frame_width}\n")

def main():
    img = cv2.imread("./iris-1.jpg")
    print_image_information(img)

    cam = cv2.VideoCapture(0)
    print_camera_information(cam)
    cam.release()

main()


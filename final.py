import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import Frame
def detect_car_colors(car_image):
    # Define color categories (you can expand this as needed)
    color_dict = {
    (255, 0, 0): "Red",
    (0, 255, 0): "Green",
    (0, 0, 255): "Blue",
    (0,0,0):"Black",
    (255,255,255):"White",
    (128, 128, 128): "Gray",
    (255, 255, 0): "Yellow",
    (255, 0, 255): "Magenta",
    (0, 255, 255): "Cyan",
    (128, 0, 0): "Maroon",
    (0, 128, 0): "Lime",
    (0, 0, 128): "Navy",
    (255, 128, 0): "Orange",
    (128, 255, 0): "Lime Green",
    (0, 128, 255): "Sky Blue",
    (255, 0, 128): "Rose",
    (128, 0, 128): "Purple",
    (128, 255, 128): "Pale Green",
    (0, 128, 128): "Teal",
    (128, 128, 0): "Olive",
    }
    # Split the car image into three vertical segments (for demonstration)
    height, width, _ = car_image.shape
    segments = [(0, height // 3), (height // 3, 2 * height // 3), (2 * height // 3, height)]
    # segments = [(0, height // 4), (height // 4, 2 * height // 4), (2 * height // 4, 3 * height // 4), (3 * height // 4, height)]
    car_colors = []

    for y1, y2 in segments:
        region = car_image[y1:y2, :]    
        hist = cv2.calcHist([region], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        max_color = tuple(hist.argmax(axis=0)[0])
        car_color = find_closest_color(max_color, color_dict)
        car_colors.append(car_color) 
    return car_colors
def start_detection(video_path, detect_color=False):
    is_paused=False
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():  
        print("Error: Could not open the video file.")
        return

    car_cascade = cv2.CascadeClassifier('carx.xml')

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    cv2.namedWindow('Car Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Car Detection', frame_width, frame_height)
    while True:
        if not is_paused:
            isTrue, frames = cap.read()
            gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            cars = car_cascade.detectMultiScale(gray, 1.1, 3)

            for (x, y, w, h) in cars:
                car_image = frames[y:y+h, x:x+w]  # Extract the car region
                cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 0, 255), 1)
                

                if detect_color:
                    car_colors = detect_car_colors(car_image)
                    text = "Car Colors: " + ", ".join(car_colors)
                    cv2.putText(frames, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

            cv2.imshow('Car Detection', frames)
        if cv2.waitKey(33) == ord('d'):
            break
        elif cv2.waitKey(33) == ord('f'):
            is_paused = not is_paused
    cap.release()
    cv2.destroyAllWindows()

# Function to find the closest color category
def find_closest_color(color, color_dict):
    min_distance = float("inf")
    closest_color = None

    for key, value in color_dict.items():
        # print(color,key)
        distance = sum((color[i] - key[i]) ** 2 for i in range(3))  # Euclidean distance
        if distance < min_distance:
            min_distance = distance
            closest_color = value

    return closest_color

def on_start_button_click(detect_color=False):
    video_path = filedialog.askopenfilename(filetypes=[("all files", "*.mp4")])
    if video_path:
        start_detection(video_path, detect_color)

root = tk.Tk()

root.title("Car Detection GUI by 22EEB0A34")
root.configure(bg="dark blue")
root.geometry("400x200")

background_frame = ttk.Frame(root, style="TFrameStyle.TFrame")
background_frame.pack(fill="both", expand=True)

title_label = ttk.Label(background_frame, text="     CAR DETECTION     ", style="TLabelTitle.TLabel")
title_label.pack(pady=20)

start_button = ttk.Button(root, text="Detect Cars", command=lambda: on_start_button_click(detect_color=False))
start_button.pack(pady=10)

start_button_color = ttk.Button(root, text="Detect Cars and Colors", command=lambda: on_start_button_click(detect_color=True))
start_button_color.pack(pady=10)

exit_button = ttk.Button(root, text="Exit", command=root.destroy)
exit_button.pack(pady=10)

style = ttk.Style()
style.configure("TFrameStyle.TFrame", background="dark blue")  # Background color

root.mainloop()
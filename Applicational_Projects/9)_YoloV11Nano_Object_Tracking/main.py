from ultralytics import YOLO
import cv2

model = YOLO(r"Applicational_Projects\9)_YoloV11Nano_Object_Tracking\yolo11n.pt")
video_path = r'Applicational_Projects\9)_YoloV11Nano_Object_Tracking\ReusingPrevInput.mp4'
cap = cv2.VideoCapture(video_path)

ret = True
while ret:
    ret, frame = cap.read()
    if ret:
        results = model.track(frame, persist=True)
        frame_ = results[0].plot()
        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
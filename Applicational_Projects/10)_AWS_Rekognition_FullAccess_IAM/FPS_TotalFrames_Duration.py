import cv2

cap = cv2.VideoCapture(r"Applicational_Projects\10)_AWS_Rekognition_FullAccess_IAM\Horses25FPS.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

print("FPS:", fps)
print("Total Frames:", frame_count)
print("Duration (s):", duration)

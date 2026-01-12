import cv2

input_file = r"Applicational_Projects\10)_AWS_Rekognition_FullAccess_IAM\Horses60FPS.mp4"
output_file = r"Applicational_Projects\10)_AWS_Rekognition_FullAccess_IAM\Horses25FPS.mp4"

cap = cv2.VideoCapture(input_file)

# Get original properties
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_in = cap.get(cv2.CAP_PROP_FPS)

# Define output writer at 25 FPS
out = cv2.VideoWriter(output_file,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      25,
                      (W, H))

frame_skip = int(round(fps_in / 25))  # ~2 or 3 for 60→25
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % frame_skip == 0:
        out.write(frame)

    frame_id += 1

cap.release()
out.release()

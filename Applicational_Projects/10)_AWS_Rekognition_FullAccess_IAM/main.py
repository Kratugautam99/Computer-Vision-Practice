import os
import cv2
import boto3
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

access_key = os.getenv('AWSRekoAccessKey')
secret_key = os.getenv('AWSRekoSecretKey')

# Output directories
output_dir = r'Applicational_Projects\10)_AWS_Rekognition_BBOX_Data\output_data'
anns_dir = os.path.join(output_dir, 'outp_frame_data')
imgs_dir = os.path.join(output_dir, 'outp_frame_imgs')

# Reset directories
for dir_ in [output_dir, anns_dir, imgs_dir]:
    if os.path.exists(dir_):
        shutil.rmtree(dir_)
    os.mkdir(dir_)

# AWS Rekognition client
reko_client = boto3.client(
    'rekognition',
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name='us-east-1'
)

# Input video
input_file = r"Applicational_Projects\10)_AWS_Rekognition_BBOX_Data\Horses25FPS.mp4"
cap = cv2.VideoCapture(input_file)

# Set the target class
target_class = 'Horse'

frame_nmr = -1
ret = True

# Read frames
while ret:
    ret, frame = cap.read()

    if ret:
        frame_nmr += 1
        H, W, _ = frame.shape

        # Convert frame to jpg
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()

        # Detect objects
        response = reko_client.detect_labels(
            Image={'Bytes': image_bytes},
            MinConfidence=50
        )

        # Save annotations
        with open(os.path.join(anns_dir, f'frame_{str(frame_nmr).zfill(6)}.txt'), 'w') as f:
            for label in response['Labels']:
                if label['Name'] == target_class:
                    for instance in label['Instances']:
                        bbox = instance['BoundingBox']
                        conf = instance['Confidence'] / 100.0  # normalize to 0–1

                        # Rekognition gives relative coords (0–1), convert to pixels
                        x1 = int(bbox['Left'] * W)
                        y1 = int(bbox['Top'] * H)
                        w = int(bbox['Width'] * W)
                        h = int(bbox['Height'] * H)

                        # Write detections (YOLO-style)
                        f.write('{} {} {} {} {} {}\n'.format(
                            0,
                            bbox['Left'] + bbox['Width'] / 2,
                            bbox['Top'] + bbox['Height'] / 2,
                            bbox['Width'],
                            bbox['Height'],
                            conf
                        ))

                        # Draw bounding box in BLUE
                        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 4)

                        # Put label + confidence
                        label_text = f"{target_class}: {conf:.2f}"
                        cv2.putText(frame,
                                label_text,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,              # font scale (bigger text)
                                (255, 0, 0),      # blue color
                                2,                # thickness (bolder)
                                cv2.LINE_AA)      # anti-aliased for smoother edges


        # Show and save frame
        cv2.namedWindow('10) AWS Rekognition BBOX Data Project', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('10) AWS Rekognition BBOX Data Project', 480, 852)
        cv2.imshow('10) AWS Rekognition BBOX Data Project', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
           break
        cv2.imwrite(os.path.join(imgs_dir, f'frame_{str(frame_nmr).zfill(6)}.jpg'), frame)

cap.release()
cv2.destroyAllWindows()

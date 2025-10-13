import os, cv2, argparse
import mediapipe as mp


def process_img(img, face_detection):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # print(x1, y1, w, h)
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (200, 200))
    return img


args = argparse.ArgumentParser()
args.add_argument("--mode", default='webcam')
args.add_argument("--filePath", default=None)
args = args.parse_args()


output_dir = './Applicational_Projects/2)_Face_Anonymizer_Image_Video_Webcam/data/outputs'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

mp_face_detection = mp.solutions.face_detection
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    if args.mode.lower() in ["image"]:
        img = cv2.imread(args.filePath)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {args.filePath}. Current working dir: {os.getcwd()}")
        img = process_img(img, face_detection)
        cv2.imwrite(os.path.join(output_dir, '1) faceblurred.png'), img)

    elif args.mode.lower() in ['video']:
        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()
        output_video = cv2.VideoWriter(os.path.join(output_dir, '2) presentationblurred.mp4'),
                                       cv2.VideoWriter_fourcc(*'MP4V'), 25, (frame.shape[1], frame.shape[0]))
        while ret:
            frame = process_img(frame, face_detection)
            output_video.write(frame)
            ret, frame = cap.read()
        cap.release()
        output_video.release()

    elif args.mode.lower() in ['webcam']:
        cam = cv2.VideoCapture(0)
        ret, frame = cam.read()
        while ret:
            frame = process_img(frame, face_detection)
            cv2.imshow('2) Face Anonymizer Image/Video/Webcam Project', frame)
            if chr(cv2.waitKey(25) & 0xFF).lower() == 'q':
                break
            ret, frame = cam.read()
        cam.release()
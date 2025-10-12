import os
import cv2
import easyocr
import matplotlib.pyplot as plt

# paths
input_dir = r"Applicational_Projects\3)_Text_Detection_OCR\data\inputs"
output_dir = r"Applicational_Projects\3)_Text_Detection_OCR\data\outputs"
os.makedirs(output_dir, exist_ok=True)

# instance text detector
reader = easyocr.Reader(['en'], gpu=False)

threshold = 0.25

for fname in os.listdir(input_dir):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        continue

    image_path = os.path.join(input_dir, fname)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read {image_path}")
        continue

    # detect text
    results = reader.readtext(img)

    # draw bbox and text
    for bbox, text, score in results:
        if score > threshold:
            cv2.rectangle(img, [int(j) for j in bbox[0]], [int(j) for j in bbox[2]], (0, 255, 0), 3)
            cv2.putText(img, text, [int(j) for j in bbox[0]], cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    # build output filename: "1) spear.jpg" → "1) spear_outlined.jpg"
    name, ext = os.path.splitext(fname)
    out_name = f"{name}_outlined{ext}"
    output_path = os.path.join(output_dir, out_name)

    cv2.imwrite(output_path, img)
    print(f"Saved {output_path}")


    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

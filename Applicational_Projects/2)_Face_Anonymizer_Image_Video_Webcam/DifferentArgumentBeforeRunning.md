# View these Different possible Args before Running

It supports three modes:
1. **Image** – Processes a single image file.
2. **Video** – Processes a video file.
3. **Webcam** – Processes live camera feed.

---
### Using `argparse` to Specify Mode

The script accepts two main arguments:
- `--mode`: One of `image`, `video`, or `webcam`
- `--filePath`: Path to the image/video file (required only for `image` and `video`)

#### Example Commands

| Mode | Command |
|------|--------|
| **Image** | `python main.py --mode image --filePath Applicational_Projects\2)_Face_Anonymizer_Image_Video_Webcam\data\outputs\1) faceblurred.png` |
| **Video** | `python main.py --mode video --filePath Applicational_Projects\2)_Face_Anonymizer_Image_Video_Webcam\data\outputs\2) presentationblurred.mp4` |
| **Webcam** | `python main.py --mode webcam` |

> Note: For `webcam`, `--filePath` is optional and ignored.

---

### 📁 Output Location

All processed outputs are saved in 
```bash
"Applicational_Projects\2)_Face_Anonymizer_Image_Video_Webcam\data\outputs"
```
import HandTrackingModule as htm
import time, math, cv2, numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv2.VideoCapture(0)
pTime = 0

detector = htm.handDetector()
device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
# Get volume range (in dB)
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        # Hand range 50 - 250
        # Volume Range -65 - 0 (dB)

        vol = np.interp(length, [50, 250], [minVol, maxVol])
        volBar = np.interp(length, [50, 250], [400, 150])
        volPer = np.interp(length, [50, 250], [0, 100])

        print(int(length), vol)
        fingers = detector.fingersUp()
        if not fingers[4]:
            volume.SetMasterVolumeLevelScalar(volPer / 100, None)
            colorVol = (0, 255, 0)
        else:
            colorVol = (255, 0, 0)
        # volume.SetMasterVolumeLevelScalar(volPer / 100.0, None)

        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (48, 150), (87, 402), (0, 0, 255), 4)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (130, 90, 130), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                1, (85, 170, 255), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    cv2.imshow("13) Hand Gesture Volume Control Project", img)
    key = cv2.waitKey(1) & 0xFF
    if chr(key).lower() == 'q':
        break
cap.release()
cv2.destroyAllWindows()
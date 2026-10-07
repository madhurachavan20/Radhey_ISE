import cv2
import math
import mediapipe as mp
from pycaw.pycaw import AudioUtilities

# ============================================================
# WINDOWS VOLUME SETUP
# ============================================================

devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume

volume_range = volume.GetVolumeRange()
min_volume = volume_range[0]
max_volume = volume_range[1]

# ============================================================
# MEDIAPIPE SETUP
# ============================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ============================================================
# CAMERA SETUP
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    print("Try changing VideoCapture(0) to VideoCapture(1).")
    exit()

print("======================================")
print("   HAND GESTURE VOLUME CONTROL")
print("======================================")
print("Thumb + Index Finger")
print("Close fingers = Lower volume")
print("Open fingers = Increase volume")
print("Press Q or ESC to quit")
print("======================================")

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, img = cap.read()

    if not success:
        print("ERROR: Could not read camera frame.")
        break

    # Mirror camera
    img = cv2.flip(img, 1)

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect hand
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Thumb tip
            thumb = hand_landmarks.landmark[
                mp_hands.HandLandmark.THUMB_TIP
            ]

            # Index finger tip
            index = hand_landmarks.landmark[
                mp_hands.HandLandmark.INDEX_FINGER_TIP
            ]

            # Image dimensions
            h, w, c = img.shape

            # Convert coordinates
            thumb_x = int(thumb.x * w)
            thumb_y = int(thumb.y * h)

            index_x = int(index.x * w)
            index_y = int(index.y * h)

            # Draw thumb
            cv2.circle(
                img,
                (thumb_x, thumb_y),
                10,
                (255, 0, 255),
                cv2.FILLED
            )

            # Draw index finger
            cv2.circle(
                img,
                (index_x, index_y),
                10,
                (255, 0, 255),
                cv2.FILLED
            )

            # Draw line
            cv2.line(
                img,
                (thumb_x, thumb_y),
                (index_x, index_y),
                (255, 0, 255),
                3
            )

            # =================================================
            # CALCULATE DISTANCE
            # =================================================

            distance = math.hypot(
                index_x - thumb_x,
                index_y - thumb_y
            )

            # =================================================
            # CONVERT DISTANCE TO VOLUME
            # =================================================

            vol = min_volume + (
                (distance - 30) / 220
            ) * (max_volume - min_volume)

            # Limit volume
            vol = max(
                min_volume,
                min(max_volume, vol)
            )

            # Set system volume
            volume.SetMasterVolumeLevel(vol, None)

            # Calculate percentage
            volume_percentage = int(
                ((vol - min_volume) /
                 (max_volume - min_volume)) * 100
            )

            # =================================================
            # VOLUME BAR
            # =================================================

            bar_x = 50
            bar_y = 100
            bar_width = 30
            bar_height = 300

            # Outer bar
            cv2.rectangle(
                img,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (255, 255, 255),
                3
            )

            # Filled bar
            filled_height = int(
                bar_height * volume_percentage / 100
            )

            cv2.rectangle(
                img,
                (
                    bar_x,
                    bar_y + bar_height - filled_height
                ),
                (
                    bar_x + bar_width,
                    bar_y + bar_height
                ),
                (0, 255, 0),
                cv2.FILLED
            )

            # =================================================
            # TEXT
            # =================================================

            cv2.putText(
                img,
                f"Volume: {volume_percentage}%",
                (100, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                img,
                f"Distance: {int(distance)}",
                (100, 480),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

    else:

        cv2.putText(
            img,
            "Show your hand",
            (200, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # ============================================================
    # TITLE
    # ============================================================

    cv2.putText(
        img,
        "Hand Gesture Volume Control",
        (100, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # ============================================================
    # SHOW WINDOW
    # ============================================================

    cv2.imshow("Volume Control", img)

    # Press Q or ESC
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()
hands.close()

print("Program closed.")
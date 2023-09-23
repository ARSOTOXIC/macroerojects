import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
index_y = 0

def drag_to(x, y):
    pyautogui.dragTo(x, y, duration=1)

def scroll(amount):
    pyautogui.scroll(amount)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark

            # Get the coordinates for index, middle, and ring fingers
            index_x = int(landmarks[8].x * frame_width)
            index_y = int(landmarks[8].y * frame_height)
            middle_x = int(landmarks[12].x * frame_width)
            middle_y = int(landmarks[12].y * frame_height)
            ring_x = int(landmarks[16].x * frame_width)
            ring_y = int(landmarks[16].y * frame_height)

            # Calculate the distances between fingers
            index_middle_distance = ((index_x - middle_x)**2 + (index_y - middle_y)**2)**0.5
            index_ring_distance = ((index_x - ring_x)**2 + (index_y - ring_y)**2)**0.5
            middle_ring_distance = ((middle_x - ring_x)**2 + (middle_y - ring_y)**2)**0.5

            # Trigger drag function if all three fingers are close
            if index_middle_distance < 15 and index_ring_distance < 15 and middle_ring_distance < 15:
                drag_to(index_x, index_y)

            # Trigger scroll function if distance between index and ring finger is less than 15
            elif index_ring_distance < 15:
                scroll(1)

    cv2.imshow('Virtual Mouse', frame)
    cv2.waitKey(1)

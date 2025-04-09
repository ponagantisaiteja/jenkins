import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Define gesture labels
GESTURE_LABELS = {0: "Fist", 1: "Open Palm", 2: "Thumbs Up"}

def get_hand_landmarks(image, results):
    """Extracts landmark positions from hand detection results."""                                              
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            return [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    return None

def recognize_gesture(landmarks):
    """Simple gesture recognition based on finger positions."""
    if not landmarks:
        return "No Hand"

    # Convert landmarks to numpy array
    landmarks = np.array(landmarks)

    # Thumb and fingers tip position
    thumb_tip = landmarks[4][0]  # x-coordinate of thumb tip
    index_tip = landmarks[8][1]  # y-coordinate of index finger tip
    middle_tip = landmarks[12][1]
    ring_tip = landmarks[16][1]
    pinky_tip = landmarks[20][1]

    # Fist Detection (All fingers down)
    if index_tip > landmarks[6][1] and middle_tip > landmarks[10][1]:
        return "Fist"

    # Open Palm Detection (All fingers up)
    if index_tip < landmarks[6][1] and middle_tip < landmarks[10][1]:
        return "Open Palm"

    # Thumbs Up Detection
    if thumb_tip > landmarks[3][0] and index_tip > landmarks[6][1]:
        return "Thumbs Up"
    
    

    return "Unknown Gesture"

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with MediaPipe Hands
    results = hands.process(rgb_frame)
    
    # Get hand landmarks
    landmarks = get_hand_landmarks(frame, results)

    # Recognize gesture
    gesture = recognize_gesture(landmarks)

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display gesture
    cv2.putText(frame, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("Gesture Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

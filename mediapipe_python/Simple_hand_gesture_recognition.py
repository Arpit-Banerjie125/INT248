import math
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def isThumbNearIndexFinger(posA, posB):
        return (math.sqrt((posA.x - posB.x)**2 + (posA.y - posB.y)**2)) < 0.1
def simpleGesture(handLandmarks):
    thumbIsOpen = False
    indexIsOpen = False
    middelIsOpen = False
    ringIsOpen = False
    pinkyIsOpen = False
        
    pseudoFixKeyPoint = handLandmarks[2].x
    if handLandmarks[3].x < pseudoFixKeyPoint and handLandmarks[4].x < pseudoFixKeyPoint:
        thumbIsOpen = True
        
    pseudoFixKeyPoint = handLandmarks[6].y
    if handLandmarks[7].y < pseudoFixKeyPoint and handLandmarks[8].y < pseudoFixKeyPoint:
        indexIsOpen = True
        
    pseudoFixKeyPoint = handLandmarks[10].y
    if handLandmarks[11].y < pseudoFixKeyPoint and handLandmarks[12].y < pseudoFixKeyPoint:
        middelIsOpen = True
        
    pseudoFixKeyPoint = handLandmarks[14].y
    if handLandmarks[15].y < pseudoFixKeyPoint and handLandmarks[16].y < pseudoFixKeyPoint:
        ringIsOpen = True
        
    pseudoFixKeyPoint = handLandmarks[18].y
    if handLandmarks[19].y < pseudoFixKeyPoint and handLandmarks[20].y < pseudoFixKeyPoint:
        pinkyIsOpen = True
        
    if thumbIsOpen and indexIsOpen and middelIsOpen and ringIsOpen and pinkyIsOpen:
        return "FIVE!"
        
    elif not thumbIsOpen and indexIsOpen and middelIsOpen and ringIsOpen and pinkyIsOpen:
        return "FOUR!"
        
    elif not thumbIsOpen and indexIsOpen and middelIsOpen and ringIsOpen and not pinkyIsOpen:
        return "THREE!"
        
    elif not thumbIsOpen and indexIsOpen and middelIsOpen and not ringIsOpen and not pinkyIsOpen:
        return "TWO!"
        
    elif not thumbIsOpen and indexIsOpen and not middelIsOpen and not ringIsOpen and not pinkyIsOpen:
        return "ONE!"
        
    elif not thumbIsOpen and indexIsOpen and not middelIsOpen and not ringIsOpen and pinkyIsOpen:
        return "ROCK!"
        
    elif thumbIsOpen and indexIsOpen and not middelIsOpen and not ringIsOpen and pinkyIsOpen:
        return "SPIDERMAN!"
        
    elif not thumbIsOpen and not indexIsOpen and not middelIsOpen and not ringIsOpen and not pinkyIsOpen:
        return "FIST!"
        
    elif not indexIsOpen and middelIsOpen and ringIsOpen and pinkyIsOpen and isThumbNearIndexFinger(handLandmarks[4], handLandmarks[8]):
        return "OK!"
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    detect=""
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      break

    # To increase performance, marking the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        detect=simpleGesture(hand_landmarks.landmark)
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flipping the image horizontally for a selfie-view display.
    image=cv2.flip(image,1)
    cv2.putText(image,detect, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2, cv2.LINE_AA)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
import time
import cv2
import mediapipe as mp

from helpers import get_names
from KinectClient import KinectClient
from PrompterClient import PrompterClient

RED = (180, 0, 0)
GREEN = (10, 127, 0)
BLUE = (30, 58, 138)
GRAY = (34, 34, 34)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def main():
    
    confirm = " "
    while confirm != 'y':
        print("Hello! Please enter your actual names.")
        user1, user2 = get_names()
        n_trials = int(input("Enter number of trials (default 10): ") or 10)
        print()
        print("REMEMBER YOUR NUMBERS")
        print(f"User 1: {user1}, User 2: {user2}")
        print(f"Number of trials: {n_trials}\n")
        confirm = input("Are these settings correct? (y/n): ").strip().lower()

    print("Starting recording sequence\n")
    prompter = PrompterClient()
    prompter.show("Get into position!", GRAY); time.sleep(5)
    prompter.show("Starting in 3", RED); time.sleep(1)
    prompter.show("Starting in 2", RED); time.sleep(1)
    prompter.show("Starting in 1", RED); time.sleep(1)
    
    kinect_client = KinectClient()
    hands = mp_hands.Hands(
        static_image_mode = False,
        max_num_hands = 2,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5
    )

    try:             
        for trial in range(1, n_trials + 1):
            print(f"Trial {trial} of {n_trials}")
            prompter.show(f"Trial {trial} of {n_trials}", RED); time.sleep(2.5)

            if trial % 2 != 0:
                prompter.show(f"{user1} hands to {user2}", RED); time.sleep(1)
                prompter.show("Start!", GREEN)
                kinect_client.start_mediapipe_recording(path=f"{trial}_{user1}_{user2}.mkv", n_seconds=4, hands = hands)
            else:
                prompter.show(f"{user2} hands to {user1}", RED); time.sleep(1)
                prompter.show("Start!", GREEN)
                kinect_client.start_mediapipe_recording(path=f"{trial}_{user2}_{user1}.mkv", n_seconds=4, hands = hands)

            prompter.show("Done!", BLUE); time.sleep(0.5)
        
        prompter.show("Recording complete", GRAY); time.sleep(1)
        print("Recording complete. Thank you!")
        prompter.close()
        hands.close()
        
    except KeyboardInterrupt:
        kinect_client.close()
        prompter.close()
        hands.close()

if __name__ == "__main__":
    main()













# import cv2
# import mediapipe as mp
# import pyk4a 
# from KinectClient import KinectClient

# kinect_client = KinectClient()

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=2,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )
# mp_drawing = mp.solutions.drawing_utils

# print("Starting camera feed. Press 'q' to exit.")

# try:
#     while True:
#         capture = kinect_client.get_capture()

#         if capture in None or capture.color is None:
#             print("Warning: Empty capture or color frame.")
#             continue
        
#         try:
#             image = cv2.cvtColor(capture.color, cv2.COLOR_BGRA2BGR)

#             if image is None or image.size == 0:
#                 print("Warning: Invalid image frame.")
#                 continue

#             if image.shape[0] >= 32767 or image.shape[1] >= 32767:
#                 print("Warning: Frame dimensions exceed SHRT_MAX.")
#                 continue

#             image = cv2.flip(image, 1)

#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             results = hands.process(image_rgb)

#             if results.multi_hand_landmarks:
            
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     mp_drawing.draw_landmarks(
#                         image, hand_landmarks, mp_hands.HAND_CONNECTIONS
#                     )
        
#             cv2.imshow('MediaPipe Hands', image)
        
#         except Exception as e:
#             print(f"Frame processing error: {e}")

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# finally:
#     print("Exiting...")
#     kinect_client.close()
#     hands.close()
#     cv2.destroyAllWindows()




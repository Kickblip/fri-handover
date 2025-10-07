import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation
import time # Import time for a quick loop delay check

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# --- CONFIGURATION ---
INPUT_VIDEO = 'demo-content/output_with_hands.mp4' # !! REMEMBER TO UPDATE THIS PATH !!
OUTPUT_VIDEO = 'outputs/video_with_axes_only.mp4'
AXIS_LENGTH_PIXELS = 70  # Length of the drawn axis lines in pixels
AXIS_THICKNESS = 3      # Thickness of the lines

# Define the landmarks used for orientation (same as your CSV script)
WRIST = 0
FORWARD_TARGET = 9 # MIDDLE_FINGER_MCP
UP_HINT_TARGET = 5 # INDEX_FINGER_MCP

def calculate_rotation_matrix(world_landmarks):
    """
    Calculates the 3x3 Rotation Matrix (R: Local -> World) based on the hand landmarks.
    """
    # Use MediaPipe's 3D World Coordinates (in meters)
    O = np.array(world_landmarks[WRIST])
    F_target = np.array(world_landmarks[FORWARD_TARGET])
    U_hint_target = np.array(world_landmarks[UP_HINT_TARGET])

    # 1. Define Vectors
    F_vec = F_target - O
    U_hint = U_hint_target - O
    
    # Check for near-zero length or collinearity before division/cross-product
    if np.linalg.norm(F_vec) < 1e-6 or np.linalg.norm(U_hint) < 1e-6:
        raise np.linalg.LinAlgError("Degenerate pose detected (near-zero vectors).")

    # 2. Build Orthonormal Basis (Right-Handed System: X=Forward, Z=Up, Y=Right)
    
    X_axis = F_vec / np.linalg.norm(F_vec)
    
    # Z-axis (Up/Normal to palm, normalized)
    Z_axis = np.cross(X_axis, U_hint)
    norm_Z = np.linalg.norm(Z_axis)
    if norm_Z < 1e-6:
        raise np.linalg.LinAlgError("Degenerate pose detected (collinear vectors).")
    Z_axis = Z_axis / norm_Z
    
    # Y-axis (Right, guaranteed orthogonal to X and Z, normalized)
    Y_axis = np.cross(Z_axis, X_axis)
    Y_axis = Y_axis / np.linalg.norm(Y_axis)

    # R_matrix columns are the hand's X, Y, Z axes in World Coordinates
    return np.column_stack((X_axis, Y_axis, Z_axis))


# --- MAIN VIDEO PROCESSING LOOP ---
cap = cv2.VideoCapture(INPUT_VIDEO)

# --- Check if the video opened successfully ---
if not cap.isOpened():
    print(f"ERROR: Could not open video file at {INPUT_VIDEO}")
    exit()

frame_count = 0

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Video stream finished or frame read failed.")
            break
            
        frame_count += 1
        print(f"Processing frame: {frame_count}") # Debug print to confirm loop runs

        # Process the image
        image.flags.writeable = False
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks, world_landmarks in zip(
                results.multi_hand_landmarks, results.multi_hand_world_landmarks):
                
                world_coords = [[lm.x, lm.y, lm.z] for lm in world_landmarks.landmark]
                
                # 1. Calculate the Hand's Rotation Matrix (R)
                try:
                    R_matrix = calculate_rotation_matrix(world_coords)
                except np.linalg.LinAlgError as e:
                    # Skip drawing axes if rotation calculation fails
                    print(f"Skipping axis drawing on frame {frame_count}: {e}")
                    continue 
                
                # 2. Get 2D Wrist Position (P0_2D) for Drawing Origin
                h, w, _ = image.shape
                wrist_2d = hand_landmarks.landmark[WRIST]
                P0_2D = (int(wrist_2d.x * w), int(wrist_2d.y * h))

                # 3. Define and Draw the Axes Endpoints

                # Direction vector (X, Y, Z) is scaled by AXIS_LENGTH_PIXELS
                
                # Note on Y-axis inversion: The Y-component of the 3D world vector (R_matrix[1, n]) 
                # is often inverted when mapping to 2D pixel space where Y increases downwards.

                # --- X-axis (Forward) - RED ---
                X_dir = R_matrix[:, 0]
                P_X_end = (
                    P0_2D[0] + int(X_dir[0] * AXIS_LENGTH_PIXELS), 
                    P0_2D[1] - int(X_dir[1] * AXIS_LENGTH_PIXELS) 
                )
                cv2.line(image, P0_2D, P_X_end, (0, 0, 255), AXIS_THICKNESS) # RED (BGR)

                # --- Y-axis (Right) - GREEN ---
                Y_dir = R_matrix[:, 1]
                P_Y_end = (
                    P0_2D[0] + int(Y_dir[0] * AXIS_LENGTH_PIXELS), 
                    P0_2D[1] - int(Y_dir[1] * AXIS_LENGTH_PIXELS)
                )
                cv2.line(image, P0_2D, P_Y_end, (0, 255, 0), AXIS_THICKNESS) # GREEN (BGR)
                
                # --- Z-axis (Up/Normal) - BLUE ---
                Z_dir = R_matrix[:, 2]
                P_Z_end = (
                    P0_2D[0] + int(Z_dir[0] * AXIS_LENGTH_PIXELS), 
                    P0_2D[1] - int(Z_dir[1] * AXIS_LENGTH_PIXELS)
                )
                cv2.line(image, P0_2D, P_Z_end, (255, 0, 0), AXIS_THICKNESS) # BLUE (BGR)

        
        # --- Display and Loop Control (CRITICAL SECTION) ---
        cv2.imshow('Hand Orientation Axes', image)
        
        # INCREASED DELAY: Use a larger delay (e.g., 20ms) to ensure the window has time to draw.
        key = cv2.waitKey(20) & 0xFF 
        if key == 27: # ESC key to break
            break
        # ----------------------------------------------------


# --- Cleanup (MUST BE CALLED) ---
cap.release()
cv2.destroyAllWindows()

print("Video processing finished.")




#####################################################################################################################
# Draws Axes on 9 and 5
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# --- CONFIGURATION ---
INPUT_VIDEO = 'demo-content/output_with_hands.mp4' # !! REMEMBER TO UPDATE THIS PATH !!
OUTPUT_VIDEO = 'outputs/video_with_triple_axes.mp4'
AXIS_LENGTH_PIXELS = 60  # Length of the drawn axis lines in pixels
AXIS_THICKNESS = 3      # Thickness of the lines

# Define the landmarks used for orientation and drawing anchors
WRIST = 0
MIDDLE_MCP = 9 
INDEX_MCP = 5 
DRAW_ANCHORS = [WRIST, MIDDLE_MCP, INDEX_MCP] # List of 2D landmark indices to draw axes on

def calculate_rotation_matrix(world_landmarks):
    """
    Calculates the 3x3 Rotation Matrix (R: Local -> World) based on the hand landmarks.
    (Origin is WRIST, X is towards MIDDLE_MCP, Z is derived using INDEX_MCP)
    """
    # Use MediaPipe's 3D World Coordinates (in meters)
    O = np.array(world_landmarks[WRIST])
    F_target = np.array(world_landmarks[MIDDLE_MCP])
    U_hint_target = np.array(world_landmarks[INDEX_MCP])

    # 1. Define Vectors
    F_vec = F_target - O
    U_hint = U_hint_target - O
    
    # Check for near-zero length or collinearity
    if np.linalg.norm(F_vec) < 1e-6 or np.linalg.norm(U_hint) < 1e-6:
        raise np.linalg.LinAlgError("Degenerate pose detected.")

    # 2. Build Orthonormal Basis (Right-Handed System: X=Forward, Z=Up, Y=Right)
    X_axis = F_vec / np.linalg.norm(F_vec)
    
    Z_axis = np.cross(X_axis, U_hint)
    norm_Z = np.linalg.norm(Z_axis)
    if norm_Z < 1e-6:
        raise np.linalg.LinAlgError("Degenerate pose detected (collinear vectors).")
    Z_axis = Z_axis / norm_Z
    
    Y_axis = np.cross(Z_axis, X_axis)
    Y_axis = Y_axis / np.linalg.norm(Y_axis)

    # R_matrix columns are the hand's X, Y, Z axes in World Coordinates
    return np.column_stack((X_axis, Y_axis, Z_axis))


def draw_axes(image, anchor_2d, R_matrix, length, thickness):
    """Helper function to draw the axes at a specific 2D point."""
    
    P0_2D = anchor_2d
    
    # Note on Y-axis inversion: World Y is up, Image Y is down. Invert Y components for screen drawing.

    # --- X-axis (Forward) - RED (BGR: 0, 0, 255) ---
    X_dir = R_matrix[:, 0]
    P_X_end = (
        P0_2D[0] + int(X_dir[0] * length), 
        P0_2D[1] - int(X_dir[1] * length) # Invert Y component
    )
    cv2.line(image, P0_2D, P_X_end, (0, 0, 255), thickness) 

    # --- Y-axis (Right) - GREEN (BGR: 0, 255, 0) ---
    Y_dir = R_matrix[:, 1]
    P_Y_end = (
        P0_2D[0] + int(Y_dir[0] * length), 
        P0_2D[1] - int(Y_dir[1] * length) # Invert Y component
    )
    cv2.line(image, P0_2D, P_Y_end, (0, 255, 0), thickness) 
    
    # --- Z-axis (Up/Normal) - BLUE (BGR: 255, 0, 0) ---
    Z_dir = R_matrix[:, 2]
    P_Z_end = (
        P0_2D[0] + int(Z_dir[0] * length), 
        P0_2D[1] - int(Z_dir[1] * length) # Invert Y component
    )
    cv2.line(image, P0_2D, P_Z_end, (255, 0, 0), thickness) 


# --- MAIN VIDEO PROCESSING LOOP ---
cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    print(f"ERROR: Could not open video file at {INPUT_VIDEO}")
    exit()

frame_count = 0

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Video stream finished or frame read failed.")
            break
            
        frame_count += 1
        
        image.flags.writeable = False
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks, world_landmarks in zip(
                results.multi_hand_landmarks, results.multi_hand_world_landmarks):
                
                # Convert 3D world landmarks to a list
                world_coords = [[lm.x, lm.y, lm.z] for lm in world_landmarks.landmark]
                
                # 1. Calculate the Hand's single Rotation Matrix (R)
                try:
                    R_matrix = calculate_rotation_matrix(world_coords)
                except np.linalg.LinAlgError as e:
                    print(f"Skipping axis drawing on frame {frame_count}: {e}")
                    continue 
                
                h, w, _ = image.shape

                # 2. Iterate through the desired anchor points (0, 9, 5) and draw the axes
                for anchor_index in DRAW_ANCHORS:
                    # Get 2D Pixel Coordinates for the current anchor point
                    anchor_2d = hand_landmarks.landmark[anchor_index]
                    P_anchor_2D = (int(anchor_2d.x * w), int(anchor_2d.y * h))

                    # Draw the three axes (X, Y, Z) at this anchor point
                    draw_axes(image, P_anchor_2D, R_matrix, AXIS_LENGTH_PIXELS, AXIS_THICKNESS)


        # --- Display and Loop Control ---
        cv2.imshow('Triple Hand Orientation Axes', image)
        
        key = cv2.waitKey(20) & 0xFF 
        if key == 27: # ESC key to break
            break
        # --------------------------------


# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("Video processing finished.")
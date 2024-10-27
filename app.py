import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

def get_landmark_coords(landmark):
    """Helper function to get coordinates from a landmark"""
    return np.array([landmark.x, landmark.y, landmark.z])

class HandExerciseGame:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Constructor to store game state. 
        self.score = 0
        self.current_exercise = None
        self.exercise_state = "ready"
        self.exercise_count = 0
        self.max_iterations = 10
        self.game_state = "home"
        self.last_detection_time = None
        self.cooldown_period = 1.0  # 1 second cooldown between moves
        self.prev_wrist_y = None
        self.movement_threshold = 0.02
        
        # Store previous positions for rotation detection
        self.prev_positions = []
        self.rotation_buffer = []
        
        # Store wrist rotation history
        self.wrist_rotation_history = []
        
        # Exercise descriptions for the home page. 
        self.exercise_descriptions = {
            "wrist_rotation": "Rotate your wrist in circular motions",
            "wrist_flexion": "Move your wrist up and down",
            "finger_lift": "Lift each finger individually while keeping others down",
            "thumb_touches": "Touch your thumb to each fingertip",
            "finger_spread": "Spread your fingers apart and bring them together",
            "wrist_supination": "Rotate your wrist so palm faces up, then down"
        }

# 100% Working. 
    def detect_wrist_rotation(self, landmarks):
        """Detect wrist rotation movements"""
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        middle_mcp = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        current_pos = np.array([middle_mcp.x - wrist.x, middle_mcp.y - wrist.y])
        self.rotation_buffer.append(current_pos)
        if len(self.rotation_buffer) > 30:
            self.rotation_buffer.pop(0)
        if len(self.rotation_buffer) >= 20: # Detect Rotation. 
            positions = np.array(self.rotation_buffer)
            variance = np.var(positions, axis=0)
            is_rotating = variance[0] > 0.001 and variance[1] > 0.001
            # Clear buffer if rotation detected
            if is_rotating:
                self.rotation_buffer = []
            
            return is_rotating
            
        return False

# 70% working
    def detect_wrist_flexion(self, landmarks):
        """Detect wrist moving up and down slowly"""
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        middle_mcp = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        # Calculate angle between wrist and middle finger
        angle = self.calculate_angle(wrist, middle_mcp, middle_tip)
        
        # Detect significant up/down movement
        if angle > 160 or angle < 20:
            # Check if the wrist movement is slow
            if self.prev_wrist_y is not None:
                movement = abs(wrist.y - self.prev_wrist_y)
                # If movement is below the threshold, consider it slow
                if movement < self.movement_threshold:
                    self.prev_wrist_y = wrist.y
                    return True
            else:
                # Store the initial wrist position
                self.prev_wrist_y = wrist.y
            
        return False

    def detect_finger_lift(self, landmarks):
        """Detect individual finger lifting"""
        tips = [
            landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
            landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP],
            landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
        ]
        
        mcps = [
            landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP],
            landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
            landmarks[self.mp_hands.HandLandmark.RING_FINGER_MCP],
            landmarks[self.mp_hands.HandLandmark.PINKY_MCP]
        ]
        
        # Count lifted fingers
        lifted = sum(1 for tip, mcp in zip(tips, mcps) if tip.y < mcp.y)
        
        # Success if exactly one finger is lifted
        return lifted == 1

# 100% working. 
    def detect_thumb_touches(self, landmarks):
        """Detect thumb touching each fingertip"""
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        fingertips = [
            landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
            landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP],
            landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
        ]
        
        # Check if thumb is close to any fingertip
        for tip in fingertips:
            distance = np.sqrt(
                (thumb_tip.x - tip.x)**2 + 
                (thumb_tip.y - tip.y)**2 + 
                (thumb_tip.z - tip.z)**2
            )
            if distance < 0.05:  # Threshold for touch detection
                return True
        return False

    def detect_finger_spread(self, landmarks):
        """Detect spreading fingers apart and bringing them together"""
        finger_mcps = [
            landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP],
            landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
            landmarks[self.mp_hands.HandLandmark.RING_FINGER_MCP],
            landmarks[self.mp_hands.HandLandmark.PINKY_MCP]
        ]
        
        finger_tips = [
            landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
            landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP],
            landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
        ]
        
        # Calculate average distance between adjacent fingertips
        total_distance = 0
        for i in range(len(finger_tips) - 1):
            tip1 = finger_tips[i]
            tip2 = finger_tips[i + 1]
            distance = np.sqrt(
                (tip1.x - tip2.x)**2 + 
                (tip1.y - tip2.y)**2
            )
            total_distance += distance
        
        avg_distance = total_distance / 3  # 3 gaps between 4 fingers
        
        # Calculate average MCP distance for normalization
        total_mcp_distance = 0
        for i in range(len(finger_mcps) - 1):
            mcp1 = finger_mcps[i]
            mcp2 = finger_mcps[i + 1]
            distance = np.sqrt(
                (mcp1.x - mcp2.x)**2 + 
                (mcp1.y - mcp2.y)**2
            )
            total_mcp_distance += distance
        
        avg_mcp_distance = total_mcp_distance / 3
        
        # Normalize finger spread by MCP distance to account for hand size and distance from camera
        spread_ratio = avg_distance / avg_mcp_distance
        
        # Define threshold for significant spread
        spread_threshold = 1.5  # Adjust this value based on testing
        
        return spread_ratio > spread_threshold

    def detect_wrist_supination(self, landmarks):
        """Detect wrist supination/pronation movement (palm rotating up and down)"""
        # Get key landmarks for tracking rotation
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        index_mcp = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        pinky_mcp = landmarks[self.mp_hands.HandLandmark.PINKY_MCP]
        middle_mcp = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        
        # Calculate hand plane normal vector using cross product of two vectors on the hand
        vector1 = np.array([index_mcp.x - pinky_mcp.x, 
                          index_mcp.y - pinky_mcp.y, 
                          index_mcp.z - pinky_mcp.z])
        vector2 = np.array([middle_mcp.x - wrist.x,
                          middle_mcp.y - wrist.y,
                          middle_mcp.z - wrist.z])
        normal = np.cross(vector1, vector2)
        normal = normal / np.linalg.norm(normal)  # Normalize the vector
        
        # Track the rotation of the normal vector over time
        self.wrist_rotation_history.append(normal)
        if len(self.wrist_rotation_history) > 30:  # Keep last 30 frames
            self.wrist_rotation_history.pop(0)
        
        # Need at least 20 frames to detect rotation
        if len(self.wrist_rotation_history) < 20:
            return False
            
        # Calculate the angle change between the first and last normal vector
        start_normal = self.wrist_rotation_history[0]
        end_normal = self.wrist_rotation_history[-1]
        dot_product = np.clip(np.dot(start_normal, end_normal), -1.0, 1.0)
        angle_change = np.abs(np.arccos(dot_product) * 180 / np.pi)
        
        # Check if there's significant rotation
        rotation_threshold = 45  # Degrees of rotation required
        if angle_change > rotation_threshold:
            # Clear history if rotation detected
            self.wrist_rotation_history = []
            return True
            
        return False

    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        vector1 = np.array([point1.x - point2.x, point1.y - point2.y])
        vector2 = np.array([point3.x - point2.x, point3.y - point2.y])
        
        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        
        return angle

    def draw_home_screen(self, frame):
        """Draw the home screen interface"""
        # Add semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Title
        cv2.putText(frame, "Hand Exercise Therapy", (int(frame.shape[1]/2) - 200, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press SPACE to Start", (int(frame.shape[1]/2) - 150, frame.shape[0] - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # List exercises
        start_y = 150
        for i, (exercise, description) in enumerate(self.exercise_descriptions.items()):
            exercise_name = exercise.replace('_', ' ').title()
            cv2.putText(frame, f"{i+1}. {exercise_name}", (50, start_y + i*50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"- {description}", (300, start_y + i*50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return frame

    def draw_interface(self, frame, hand_landmarks=None):
        """Draw game interface elements"""
        if self.game_state == "home":
            return self.draw_home_screen(frame)
            
        # Draw score and iteration count
        cv2.putText(frame, f"Score: {self.score}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Rep: {self.exercise_count}/{self.max_iterations}", 
                   (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw current exercise name
        exercise_name = self.current_exercise.replace('_', ' ').title()
        cv2.putText(frame, f"Exercise: {exercise_name}", 
                   (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw exercise description
        cv2.putText(frame, self.exercise_descriptions[self.current_exercise],
                   (20, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Draw hand landmarks if they exist
        if hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
        # Draw ready/success message
        if self.exercise_state == "success":
            cv2.putText(frame, "Good job!", (frame.shape[1]//2 - 100, frame.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        elif self.exercise_state == "ready":
            cv2.putText(frame, "Ready!", (frame.shape[1]//2 - 80, frame.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

        return frame

    def process_frame(self, frame):
        """Process a single frame"""
        if self.game_state == "home":
            return self.draw_home_screen(frame)
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            
            # Detect current exercise
            detected = False
            if self.current_exercise == "wrist_rotation":
                detected = self.detect_wrist_rotation(landmarks)
            elif self.current_exercise == "wrist_flexion":
                detected = self.detect_wrist_flexion(landmarks)
            elif self.current_exercise == "finger_lift":
                detected = self.detect_finger_lift(landmarks)
            elif self.current_exercise == "thumb_touches":
                detected = self.detect_thumb_touches(landmarks)
            elif self.current_exercise == "finger_spread":
                detected = self.detect_finger_spread(landmarks)
            elif self.current_exercise == "wrist_supination":
                detected = self.detect_wrist_supination(landmarks)
            
            # Update exercise state
            current_time = datetime.now()
            if detected and (self.last_detection_time is None or 
                           (current_time - self.last_detection_time).total_seconds() >= self.cooldown_period):
                self.score += 1
                self.exercise_count += 1
                self.exercise_state = "success"
                self.last_detection_time = current_time
                
                # Reset for next iteration or exercise
                if self.exercise_count >= self.max_iterations:
                    self.next_exercise()
            elif (self.last_detection_time is None or 
                  (current_time - self.last_detection_time).total_seconds() >= self.cooldown_period):
                self.exercise_state = "ready"
            
            # Draw the interface
            frame = self.draw_interface(frame, results.multi_hand_landmarks[0])
        else:
            frame = self.draw_interface(frame)
        
        return frame

    def next_exercise(self):
        """Switch to the next exercise"""
        exercises = list(self.exercise_descriptions.keys())
        if self.current_exercise is None:
            self.current_exercise = exercises[0]
        else:
            current_idx = exercises.index(self.current_exercise)
            next_idx = (current_idx + 1) % len(exercises)
            self.current_exercise = exercises[next_idx]
        
        self.exercise_count = 0
        self.exercise_state = "ready"
        self.last_detection_time = None
        print(f"Next exercise: {self.current_exercise}")

    def start_game(self):
        """Start the game loop"""
        cap = cv2.VideoCapture(0)  # Start video capture

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image")
                break

            # Process the frame
            frame = self.process_frame(frame)

            # Display the frame
            cv2.imshow("Hand Exercise Therapy", frame)

                        # Handle key events
            key = cv2.waitKey(1)
            if key == ord(' '):  # Start the game when SPACE is pressed
                if self.game_state == "home":
                    self.game_state = "playing"
                    self.current_exercise = None  # Start with the first exercise
                    self.next_exercise()
            elif key == ord('q'):  # Quit the game when 'q' is pressed
                print("Quitting the game...")
                break

        # Release the video capture and close windows
        cap.release()
        cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    game = HandExerciseGame()
    game.start_game()
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

NUM_HANDS = 2
NUM_HAND_KEYPOINTS = 21
FEATURES_PER_FRAME = 294  
REFERENCE_IMAGE_SIZE = (640, 480)

FINGER_COLORS = {
    'thumb': (255, 0, 0),    # Red
    'index': (0, 255, 0),    # Green
    'middle': (0, 0, 255),   # Blue
    'ring': (255, 255, 0),   # Yellow
    'pinky': (255, 0, 255)   # Magenta
}

FINGER_CONNECTIONS = {
    'thumb': [(0, 1), (1, 2), (2, 3), (3, 4)],
    'index': [(0, 5), (5, 6), (6, 7), (7, 8)],
    'middle': [(0, 9), (9, 10), (10, 11), (11, 12)],
    'ring': [(0, 13), (13, 14), (14, 15), (15, 16)],
    'pinky': [(0, 17), (17, 18), (18, 19), (19, 20)]
}

def extract_arm_features(results_pose, results_hands):
    features = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)
    offset = 0
    hand_count = 0
    
    if results_hands and results_hands.multi_hand_landmarks:
        hands = {
            "left": np.zeros(NUM_HAND_KEYPOINTS * 3),
            "right": np.zeros(NUM_HAND_KEYPOINTS * 3)
        }
        
        hand_count = len(results_hands.multi_hand_landmarks)
        for idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks[:NUM_HANDS]):
            handedness = results_hands.multi_handedness[idx].classification[0].label.lower()
            keypoints = np.array([
                (lm.x * REFERENCE_IMAGE_SIZE[0], 
                 lm.y * REFERENCE_IMAGE_SIZE[1], 
                 lm.z * REFERENCE_IMAGE_SIZE[0])
                for lm in hand_landmarks.landmark
            ]).flatten()
            hands[handedness][:len(keypoints)] = keypoints

        for hand in ["left", "right"]:
            wrist = hands[hand][:3]
            if np.any(wrist != 0):
                keypoints = hands[hand].reshape(-1, 3) - wrist
                hand_size = np.linalg.norm(keypoints[8] - keypoints[0]) + 1e-6
                keypoints /= hand_size
                features[offset:offset+63] = keypoints.flatten()[:63]
                offset += 63
            else:
                offset += 63

        geo_features = calculate_hand_geometrics(hands)
        features[offset:offset+len(geo_features)] = geo_features
    
    return features, hand_count

def calculate_hand_geometrics(hands):
    features = []
    for hand in ["left", "right"]:
        keypoints = hands[hand].reshape(-1, 3)
        if np.all(keypoints == 0):
            features.extend([0] * 84)  # 84 features per hand
            continue
        
        wrist = keypoints[0]
        thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip = keypoints[[4, 8, 12, 16, 20]]
        palm_center = np.mean(keypoints[[0, 5, 9, 13, 17]], axis=0)
        finger_tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
        
        # Cosine angles from palm center to finger tips
        for tip in finger_tips:
            v = tip - palm_center
            v_norm = np.linalg.norm(v) + 1e-6
            cos_angle = v[1] / v_norm
            features.append(cos_angle)
        
        # Distances from wrist to finger tips
        for tip in finger_tips:
            length = np.linalg.norm(tip - wrist)
            features.append(length)
        
        # Distances between adjacent finger tips
        for i in range(len(finger_tips) - 1):
            dist = np.linalg.norm(finger_tips[i] - finger_tips[i + 1])
            features.append(dist)
        
        # Palm size (wrist to middle finger MCP)
        palm_size = np.linalg.norm(keypoints[9] - wrist)
        features.append(palm_size)
        
        # Palm plane normal
        v1 = index_tip - wrist
        v2 = pinky_tip - wrist
        normal = np.cross(v1, v2)
        normal /= (np.linalg.norm(normal) + 1e-6)
        features.extend(normal)
        
        # Distances between finger bases
        base_points = keypoints[[5, 9, 13, 17]]
        for i in range(len(base_points) - 1):
            dist = np.linalg.norm(base_points[i] - base_points[i + 1])
            features.append(dist)
        
        # Angles between adjacent fingers (wrist as vertex)
        finger_pairs = [(4, 8), (8, 12), (12, 16), (16, 20)]  # Thumb-Index, Index-Middle, etc.
        for start_idx, end_idx in finger_pairs:
            v1 = keypoints[start_idx] - wrist
            v2 = keypoints[end_idx] - wrist
            dot_product = np.dot(v1, v2)
            norms = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
            cos_angle = dot_product / norms
            angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            features.append(angle_rad)
        
        # Angles within fingers (between segments)
        finger_keypoints = {
            'thumb': [1, 2, 3, 4],   # CMC to tip
            'index': [5, 6, 7, 8],   # MCP to tip
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        for finger, indices in finger_keypoints.items():
            for i in range(len(indices) - 2):
                p1, p2, p3 = keypoints[indices[i]], keypoints[indices[i+1]], keypoints[indices[i+2]]
                v1 = p2 - p1
                v2 = p3 - p2
                dot_product = np.dot(v1, v2)
                norms = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
                cos_angle = dot_product / norms
                angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                features.append(angle_rad)
        
        while len(features) % 84 != 0:
            features.append(0)

    return np.array(features[:168], dtype=np.float32)  # 168 features total (84 per hand × 2)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    try:
        results = model.process(image)
    except Exception as e:
        print(f"MediaPipe Error: {e}")
        results = None
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results

def visualize_landmarks(image, results_pose, results_hands):
    if results_hands and results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=1)
            )
            
            # Highlight wrist
            wrist = hand_landmarks.landmark[0]
            wrist_coords = (int(wrist.x * image.shape[1]), int(wrist.y * image.shape[0]))
            cv2.circle(image, wrist_coords, 6, (0, 255, 255), -1)  # Yellow wrist
            
            # Draw finger connections
            for finger, connections in FINGER_CONNECTIONS.items():
                for start, end in connections:
                    start_point = hand_landmarks.landmark[start]
                    end_point = hand_landmarks.landmark[end]
                    start_coords = (int(start_point.x * image.shape[1]), int(start_point.y * image.shape[0]))
                    end_coords = (int(end_point.x * image.shape[1]), int(end_point.y * image.shape[0]))
                    cv2.line(image, start_coords, end_coords, FINGER_COLORS[finger], 3)
            
            # Display angle between thumb and index
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            v1 = np.array([thumb_tip.x - wrist.x, thumb_tip.y - wrist.y, thumb_tip.z - wrist.z])
            v2 = np.array([index_tip.x - wrist.x, index_tip.y - wrist.y, index_tip.z - wrist.z])
            dot_product = np.dot(v1, v2)
            norms = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
            cos_angle = dot_product / norms
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            cv2.putText(image, f"T-I: {angle_deg:.1f}°", (wrist_coords[0] + 10, wrist_coords[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return image
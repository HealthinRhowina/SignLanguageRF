import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from scipy.interpolate import interp1d
from m import mediapipe_detection, extract_arm_features

DATA_PATH = 'new_data5'
VIDEO_PATH = r"D:\data"
ACTIONS = ['a', 'about', 'action', 'again', 'age', 'all', 'also', 'always', 'and', 'angry', 'answer', 'area', 'as', 'ask', 'back', 'bad', 'bath', 'bathroom', 'beautifully', 'because', 'bed', 'best', 'bible', 'big', 'bird', 'book', 'bread', 'busy', 'but', 'buy', 'call', 'can', 'canceled', 'car', 'cat', 'chair', 'child', 'children', 'christ', 'city', 'clean', 'clear', 'close', 'clothes', 'coffee', 'cold', 'color', 'come', 'computer', 'cook', 'cool', 'cute', 'dance', 'dancer', 'delicious', 'different', 'difficult', 'dinner', 'do', 'doctor', 'dog', "don't want", 'door', 'drink', 'early', 'easy', 'eat', 'egg', 'eight', 'enjoy', 'evening', 'every', 'face', 'family', 'feel', 'few', 'fight', 'find', 'finish', 'first', 'fish', 'food', 'for', 'free', 'friend', 'from', 'fruit', 'fun', 'game', 'give', 'go', 'good', 'goodbye', 'grandfather', 'guitar', 'happy', 'hard', 'hat', 'have', 'health', 'hear', 'help', 'her', 'here', 'his', 'home', 'hospital', 'hot', 'hours', 'house', 'how', 'hungry', 'I', 'idea', 'important', 'in', 'is', 'it', 'jealous', 'key', 'know', 'language', 'last', 'late', 'learn', 'leave', 'less', 'like', 'live', 'long', 'loves', 'mad', 'make', 'man', 'many', 'maybe', 'me', 'medicine', 'meet', 'meeting', 'milk', 'minute', 'mom', 'money', 'month', 'more', 'morning', 'mother', 'movie', 'music', 'my', 'name', 'need', 'new', 'next', 'nice', 'night', 'no', 'nothing', 'now', 'number', 'often', 'old', 'on', 'one', 'open', 'our', 'paper', 'park', 'phone', 'place', 'plan', 'playes', 'please', 'poor', 'pray', 'price', 'problem', 'project', 'question', 'rain', 'reading', 'ready', 'real', 'rich', 'right', 'run', 'sad', 'safe', 'same', 'say', 'school', 'see', 'she', 'shirt', 'shop', 'short', 'should', 'sick', 'sign', 'sleep', 'small', 'snow', 'soft', 'some', 'sometimes', 'songs', 'spanish', 'speak', 'start', 'stay', 'stop', 'store', 'street', 'strong', 'student', 'sun', 'sunrise', 'sure', 'take', 'talk', 'taste', 'tea', 'teach', 'teacher', 'thank you', 'that', 'the', 'their', 'them', 'there', 'they', 'think', 'this', 'three', 'time', 'tired', 'to', 'today', 'together', 'tomorrow', 'tree', 'trip', 'true', 'two', 'umbrella', 'understand', 'vegetable', 'very', 'wait', 'walk', 'want', 'warm', 'wash', 'water', 'we', 'weak', 'weather', 'week', 'weekend', 'welcome', 'what', 'when', 'where', 'which', 'who', 'why', 'window', 'woman', 'works', 'write', 'wrong', 'year', 'yes', 'yesterday', 'you', 'young', 'your']
SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 294 
MIN_FRAMES = 15

def temporal_augmentation(sequence):
    original_length = len(sequence)
    x = np.linspace(0, 1, original_length)
    new_length = int(original_length * np.random.uniform(0.7, 1.3))
    new_x = np.linspace(0, 1, new_length)
    interpolator = interp1d(x, sequence, axis=0, kind='cubic')
    return interpolator(new_x)

def compute_motion_features(sequence):
    sequence = np.array(sequence, dtype=np.float32)
    velocity = np.diff(sequence, axis=0, prepend=sequence[[0]])
    combined = np.concatenate([sequence, velocity], axis=1)  
    if combined.shape[0] < SEQUENCE_LENGTH:
        pad_size = SEQUENCE_LENGTH - combined.shape[0]
        combined = np.pad(combined, [(0, pad_size), (0, 0)], mode='reflect')
    print(f"Motion features shape: {combined.shape}, Mean: {np.mean(combined):.4f}, Std: {np.std(combined):.4f}")
    return combined[:SEQUENCE_LENGTH]

def process_video(video_path, hands):
    cap = cv2.VideoCapture(video_path)
    frames = []
    hand_counts = []
    last_valid = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        _, results_hands = mediapipe_detection(frame, hands)
        features, hand_count = extract_arm_features(None, results_hands)
        if features is not None and np.any(features):
            last_valid = features
            frames.append(features)
            hand_counts.append(hand_count)
        elif last_valid is not None:
            frames.append(last_valid)
            hand_counts.append(hand_count if hand_count else 0)
    cap.release()
    if len(frames) < MIN_FRAMES:
        return None, None
    if np.random.rand() < 0.7:
        frames = temporal_augmentation(frames)
    return frames, hand_counts

def process_action(action, hands):
    action_path = os.path.join(VIDEO_PATH, action)
    save_path = os.path.join(DATA_PATH, action)
    os.makedirs(save_path, exist_ok=True)
    video_files = sorted([f for f in os.listdir(action_path) if f.endswith('.mp4')])
    print(f"Processing {action}: {len(video_files)} videos")
    for idx, video_file in enumerate(video_files):
        video_path = os.path.join(action_path, video_file)
        try:
            frames, hand_counts = process_video(video_path, hands)
            if frames is None:
                print(f"⚠ Skipping short video: {video_file}")
                continue
            motion_features = compute_motion_features(frames)
            seq_path = os.path.join(save_path, str(idx))
            os.makedirs(seq_path, exist_ok=True)
            for frame_num in range(SEQUENCE_LENGTH):
                frame_data = {'features': motion_features[frame_num], 'hand_count': hand_counts[frame_num] if frame_num < len(hand_counts) else 0}
                np.save(os.path.join(seq_path, f'{frame_num}.npy'), frame_data)
        except Exception as e:
            print(f"🚨 Error processing {video_file}: {str(e)}")

def main():
    os.makedirs(DATA_PATH, exist_ok=True)
    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8
    ) as hands:
        for action in tqdm(ACTIONS, desc="Processing Actions"):
            action_path = os.path.join(VIDEO_PATH, action)
            if not os.path.exists(action_path):
                print(f"⚠ Missing action directory: {action}")
                continue
            process_action(action, hands)
    print("✅ Hand-only data processing complete.")

if __name__ == "__main__":
    main()
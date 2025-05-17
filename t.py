import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle

DATA_PATH = "new_data5"
SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 588 
ACTIONS = ['a', 'about', 'action', 'again', 'age', 'all', 'also', 'always', 'and', 'angry', 'answer', 'area', 'as', 'ask', 'back', 'bad', 'bath', 'bathroom', 'beautifully', 'because', 'bed', 'best', 'bible', 'big', 'bird', 'book', 'bread', 'busy', 'but', 'buy', 'call', 'can', 'canceled', 'car', 'cat', 'chair', 'child', 'children', 'christ', 'city', 'clean', 'clear', 'close', 'clothes', 'coffee', 'cold', 'color', 'come', 'computer', 'cook', 'cool', 'cute', 'dance', 'dancer', 'delicious', 'different', 'difficult', 'dinner', 'do', 'doctor', 'dog', "don't want", 'door', 'drink', 'early', 'easy', 'eat', 'egg', 'eight', 'enjoy', 'evening', 'every', 'face', 'family', 'feel', 'few', 'fight', 'find', 'finish', 'first', 'fish', 'food', 'for', 'free', 'friend', 'from', 'fruit', 'fun', 'game', 'give', 'go', 'good', 'goodbye', 'grandfather', 'guitar', 'happy', 'hard', 'hat', 'have', 'health', 'hear', 'help', 'her', 'here', 'his', 'home', 'hospital', 'hot', 'hours', 'house', 'how', 'hungry', 'I', 'idea', 'important', 'in', 'is', 'it', 'jealous', 'key', 'know', 'language', 'last', 'late', 'learn', 'leave', 'less', 'like', 'live', 'long', 'loves', 'mad', 'make', 'man', 'many', 'maybe', 'me', 'medicine', 'meet', 'meeting', 'milk', 'minute', 'mom', 'money', 'month', 'more', 'morning', 'mother', 'movie', 'music', 'my', 'name', 'need', 'new', 'next', 'nice', 'night', 'no', 'nothing', 'now', 'number', 'often', 'old', 'on', 'one', 'open', 'our', 'paper', 'park', 'phone', 'place', 'plan', 'playes', 'please', 'poor', 'pray', 'price', 'problem', 'project', 'question', 'rain', 'reading', 'ready', 'real', 'rich', 'right', 'run', 'sad', 'safe', 'same', 'say', 'school', 'see', 'she', 'shirt', 'shop', 'short', 'should', 'sick', 'sign', 'sleep', 'small', 'snow', 'soft', 'some', 'sometimes', 'songs', 'spanish', 'speak', 'start', 'stay', 'stop', 'store', 'street', 'strong', 'student', 'sun', 'sunrise', 'sure', 'take', 'talk', 'taste', 'tea', 'teach', 'teacher', 'thank you', 'that', 'the', 'their', 'them', 'there', 'they', 'think', 'this', 'three', 'time', 'tired', 'to', 'today', 'together', 'tomorrow', 'tree', 'trip', 'true', 'two', 'umbrella', 'understand', 'vegetable', 'very', 'wait', 'walk', 'want', 'warm', 'wash', 'water', 'we', 'weak', 'weather', 'week', 'weekend', 'welcome', 'what', 'when', 'where', 'which', 'who', 'why', 'window', 'woman', 'works', 'write', 'wrong', 'year', 'yes', 'yesterday', 'you', 'young', 'your']
def load_data():
    X, y, hand_counts = [], [], []
    valid_actions = set(ACTIONS)
    for action_idx, action in enumerate(ACTIONS):
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path):
            print(f"Skipping missing action: {action}")
            continue
        sequence_folders = sorted([f for f in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, f))])
        for seq_folder in sequence_folders:
            seq_path = os.path.join(action_path, seq_folder)
            try:
                sequence_data = [np.load(os.path.join(seq_path, f'{i}.npy'), allow_pickle=True).item() 
                                 for i in range(SEQUENCE_LENGTH)]
                sequence = np.stack([data['features'] for data in sequence_data]).astype(np.float32)
                avg_hand_count = np.mean([data['hand_count'] for data in sequence_data])
                if sequence.shape != (SEQUENCE_LENGTH, FEATURES_PER_FRAME):
                    print(f"Skipping invalid shape {sequence.shape} in {seq_path}")
                    continue
                X.append(sequence.flatten())
                y.append(action_idx)
                hand_counts.append(avg_hand_count)
            except Exception as e:
                print(f"Error loading {seq_path}: {e}")
    
    X = np.array(X)
    y = np.array(y)
    hand_counts = np.array(hand_counts)
    print(f"Loaded {len(X)} sequences across {len(np.unique(y))} classes")
    class_counts = np.bincount(y, minlength=len(ACTIONS))
    class_distribution = {ACTIONS[i]: count for i, count in enumerate(class_counts) if count > 0}
    print(f"Class distribution: {class_distribution}")
    return X, y, hand_counts

def train_rf_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
    )
    print("\nTraining Random Forest model...")
    rf_model.fit(X_train, y_train)
    return rf_model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"\n{'='*40}")
    print("Random Forest Model Evaluation")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=ACTIONS))

if __name__ == "__main__":
    X, y, hand_counts = load_data()
    if X.shape[0] == 0:
        raise ValueError("No valid training data found! Check new_data5 directory.")
    print(f"Dataset shape: {X.shape}, Hand counts: {hand_counts.shape}")
    try:
        rf_model, X_test, y_test = train_rf_model(X, y)
    except ValueError as e:
        print(f"\nError during training: {e}")
        exit(1)
    evaluate_model(rf_model, X_test, y_test)
    with open("hsign_language_rf6.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    print("Model saved successfully as hsign_language_rf6.pkl!")
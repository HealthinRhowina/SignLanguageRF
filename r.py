import cv2
import numpy as np
import mediapipe as mp
import pickle
from collections import deque, Counter
import time
from gtts import gTTS
import io
import pygame
from m import mediapipe_detection, extract_arm_features, visualize_landmarks
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from googletrans import Translator
import nltk
from nltk import ngrams
from cc import SENTENCES,STARTER_WORDS,MIN_CONFIDENCE
from collections import defaultdict
import pyttsx3
from deep_translator import GoogleTranslator
import language_tool_python
from collections import Counter
import time

SEQUENCE_LENGTH = 30
RF_MODEL_PATH = "hsign_language_rf6.pkl"
CONFIDENCE = 0.8
PREDICTION_WINDOW_SECONDS = 0.3
ANALYSIS_DURATION = 0.9

ACTIONS = ['i', 'about', 'action', 'again', 'age', 'all', 'also', 'always', 'and', 'angry', 'answer', 'area', 'as', 'ask', 'back', 'bad', 'bath', 'bathroom', 'beautifully', 'because', 'bed', 'best', 'bible', 'big', 'bird', 'book', 'bread', 'busy', 'but', 'buy', 'call', 'can', 'canceled', 'car', 'cat', 'chair', 'child', 'children', 'christ', 'city', 'clean', 'clear', 'close', 'clothes', 'coffee', 'cold', 'color', 'come', 'computer', 'cook', 'cool', 'cute', 'dance', 'dancer', 'delicious', 'different', 'difficult', 'dinner', 'do', 'doctor', 'dog', "don't want", 'door', 'drink', 'early', 'easy', 'eat', 'egg', 'eight', 'enjoy', 'evening', 'every', 'face', 'family', 'feel', 'few', 'fight', 'find', 'finish', 'first', 'fish', 'food', 'for', 'free', 'friend', 'from', 'fruit', 'fun', 'game', 'give', 'go', 'good', 'goodbye', 'grandfather', 'guitar', 'happy', 'hard', 'hat', 'have', 'health', 'hear', 'help', 'her', 'here', 'his', 'home', 'hospital', 'hot', 'hours', 'house', 'how', 'hungry', 'I', 'idea', 'important', 'in', 'is', 'it', 'jealous', 'key', 'know', 'language', 'last', 'late', 'learn', 'leave', 'less', 'like', 'live', 'long', 'loves', 'mad', 'make', 'man', 'many', 'maybe', 'me', 'medicine', 'meet', 'meeting', 'milk', 'minute', 'mom', 'money', 'month', 'more', 'morning', 'mother', 'movie', 'music', 'my', 'name', 'need', 'new', 'next', 'nice', 'night', 'no', 'nothing', 'now', 'number', 'often', 'old', 'on', 'one', 'open', 'our', 'paper', 'park', 'phone', 'place', 'plan', 'playes', 'please', 'poor', 'pray', 'price', 'problem', 'project', 'question', 'rain', 'reading', 'ready', 'real', 'rich', 'right', 'run', 'sad', 'safe', 'same', 'say', 'school', 'see', 'she', 'shirt', 'shop', 'short', 'should', 'sick', 'sign', 'sleep', 'small', 'snow', 'soft', 'some', 'sometimes', 'songs', 'spanish', 'speak', 'start', 'stay', 'stop', 'store', 'street', 'strong', 'student', 'sun', 'sunrise', 'sure', 'take', 'talk', 'taste', 'tea', 'teach', 'teacher', 'thank you', 'that', 'the', 'their', 'them', 'there', 'they', 'think', 'this', 'three', 'time', 'tired', 'to', 'today', 'together', 'tomorrow', 'tree', 'trip', 'true', 'two', 'umbrella', 'understand', 'vegetable', 'very', 'wait', 'walk', 'want', 'warm', 'wash', 'water', 'we', 'weak', 'weather', 'week', 'weekend', 'welcome', 'what', 'when', 'where', 'which', 'who', 'why', 'window', 'woman', 'works', 'write', 'wrong', 'year', 'yes', 'yesterday', 'you', 'young', 'your']
SINGLE_HAND_ACTIONS = ['you','cat','give','cute','water','i', 'a', 'she', 'he', 'my', 'no', 'yes', 'what', 'how', 'who', 'mom','his','please','sun','hot',
                       'they','child','cat','we','drink','for','idea','trip','your','and','see','talk','eat','food','happy']
DOUBLE_HAND_ACTIONS = ['as','is','very','some', 'ask', 'bird', 'can', 'friend', 'to', 'loves', 'dance','cook','house','big','open','the','window',
                       'good','student','the','learn','new','songs','every','week','go','school','open','window','now','soft','walk','together',
                       'rain' ,'morning','park','tea','evening','pray','good','health','have','early','plan','fun','clean','often','enjoy','music',
                       'sunrise', 'morning','about','bible','want','wash','clothes','feel','today']


def generate_followers(sentences):
    followers = defaultdict(set)
    for sentence in sentences:
        words = sentence.lower().split()
        for i in range(len(words) - 1):
            followers[words[i]].add(words[i + 1])
        if words:
            followers[words[-1]].add(None)
    return {k: sorted([w for w in v if w is not None]) for k, v in followers.items()}

FOLLOWERS = generate_followers(SENTENCES)

def build_ngram_model(sentences, n=2):
    ngram_counts = defaultdict(lambda: defaultdict(int))
    total_counts = defaultdict(int)
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence.lower())
        ngram_list = list(ngrams(tokens, n, pad_left=True, left_pad_symbol=None))
        for gram in ngram_list:
            prev_word, curr_word = gram[:-1], gram[-1]
            prev_key = prev_word[0] if n == 2 else tuple(prev_word)
            ngram_counts[prev_key][curr_word] += 1
            total_counts[prev_key] += 1
    ngram_probs = defaultdict(dict)
    for prev in ngram_counts:
        for curr in ngram_counts[prev]:
            ngram_probs[prev][curr] = ngram_counts[prev][curr] / total_counts[prev]
    return ngram_probs

def load_glove_embeddings(path="glove.6B.50d.txt"):
    embeddings = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype=np.float32)
                embeddings[word] = vector
        print("GloVe embeddings loaded successfully.")
    except FileNotFoundError:
        print(f"Warning: {path} not found. Skipping GloVe embeddings.")
    return embeddings

def get_phrase_embedding(phrase, embeddings, dim=50):
    if not embeddings:
        return np.zeros(dim)
    tokens = nltk.word_tokenize(phrase.lower())
    vectors = [embeddings.get(token, np.zeros(dim)) for token in tokens]
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)

def load_model():
    try:
        with open(RF_MODEL_PATH, 'rb') as f:
            rf_model = pickle.load(f)
        print(f"Loaded Random Forest model from {RF_MODEL_PATH}")
        return rf_model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def compute_motion_features(sequence):
    sequence = np.array(sequence, dtype=np.float32)
    velocity = np.diff(sequence, axis=0, prepend=sequence[[0]])
    combined = np.concatenate([sequence, velocity], axis=1)
    if combined.shape[0] < SEQUENCE_LENGTH:
        pad_size = SEQUENCE_LENGTH - combined.shape[0]
        combined = np.pad(combined, [(0, pad_size), (0, 0)], mode='reflect')
    return combined[:SEQUENCE_LENGTH]

def preprocess_sequence(frames, embeddings=None):
    frames = np.array(frames, dtype=np.float32)
    motion_features = compute_motion_features(frames)
    flat_features = motion_features.flatten()
    return flat_features

def translate_to_tamil(text):
    try:
        translator = GoogleTranslator(source='auto', target='ta')
        translated = translator.translate(text)
        return translated
    except Exception as e:
        print("Translation error:", e)
        return "Translation failed"

class RealTimeFeatureCollector:
    def __init__(self):
        self.buffer = deque(maxlen=SEQUENCE_LENGTH)
        
    def process_frame(self, raw_features):
        self.buffer.append(raw_features)
        if len(self.buffer) == SEQUENCE_LENGTH:
            return True, np.array(self.buffer)
        return False, None
lastvalid = 0
handcount = 0    
class SignLanguagApp:
    
    def ___init__(self, root):
        self.grammar_tool = language_tool_python.LanguageTool('en-US')  
        self.word_predictions = []  
        self.analysis_start_time = None  
        self.last_pred_time = time.time()  

    def correctgrammar(self, text):
        """Apply grammatical error correction to the input text."""
        try:
            corrected = self.grammar_tool.correct(text)
            print(f"Original: {text} | Corrected: {corrected}")
            return corrected
        except Exception as e:
            print(f"GEC error: {e}")
            return text 

    def updateframe(self):
        if self.is_recording:
            if lastvalid is not None and handcount > 0:
                ready, sequence = self.collector.process_frame(lastvalid)
                if ready and time.time() - self.last_pred_time >= PREDICTION_WINDOW_SECONDS:
                    try:
                        rf_features = preprocess_sequence(sequence, self.embeddings)
                        previous_word = self.sentence_words[-1] if self.sentence_words else None
                        is_first_word = len(self.sentence_words) == 0
                        rf_label, rf_conf = self.predict_gesture(
                            self.rf_model, rf_features, previous_word, self.ngram_probs, is_first_word, handcount
                        )
                        if rf_label and rf_conf >= MIN_CONFIDENCE:
                            self.predicted_gesture = rf_label
                            if self.analysis_start_time is None:
                                self.analysis_start_time = time.time()
                                self.word_predictions = []
                            self.word_predictions.append(rf_label)
                            self.last_pred_time = time.time()
                            
                            if time.time() - self.analysis_start_time >= ANALYSIS_DURATION:
                                if self.word_predictions:
                                    word_counts = Counter(self.word_predictions)
                                    max_word = max(word_counts, key=word_counts.get)
                                    if self.is_valid_continuation(self.sentence_words[-1] if self.sentence_words else None, max_word):
                                        temp_sentence = self.sentence_words + [max_word]
                                        temp_text = " ".join(temp_sentence)
                                        corrected_text = self.correct_grammar(temp_text)
                                        corrected_words = corrected_text.split()
                                        if len(corrected_words) >= len(self.sentence_words) + 1:
                                            self.sentence_words = corrected_words
                                        else:
                                            self.sentence_words.append(max_word)  
                                        print(f"Sentence so far: {' '.join(self.sentence_words)}")
                                        self.prediction_text.delete(1.0, tk.END)
                                        self.prediction_text.insert(tk.END, " ".join(self.sentence_words))
                                    else:
                                        print(f"Rejected word: {max_word}")
                                    self.word_predictions = []
                                    self.analysis_start_time = None
                    except Exception as e:
                        print(f"Prediction error: {e}")
    def quitapp(self):
        self.grammar_tool.close()  

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Recognition")
        
        self.rf_model = load_model()
        self.embeddings = load_glove_embeddings()
        self.ngram_probs = build_ngram_model(SENTENCES, n=2)
        self.collector = RealTimeFeatureCollector()
        self.predicted_gesture = ""
        self.is_recording = False
        self.last_pred_time = time.time()
        self.sentence_words = []
        self.word_predictions = []
        self.analysis_start_time = None
        
        pygame.mixer.init()
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        self.hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        self.video_label = tk.Label(root)
        self.video_label.pack()
        
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(pady=10)
        
        self.start_stop_btn = tk.Button(self.control_frame, text="Start", command=self.toggle_recording, bg="green", fg="white")
        self.start_stop_btn.grid(row=0, column=0, padx=5)
        
        self.voice_btn = tk.Button(self.control_frame, text="Voice", command=self.speak_prediction, bg="blue", fg="white")
        self.voice_btn.grid(row=0, column=1, padx=5)
        
        self.clear_btn = tk.Button(self.control_frame, text="Clear", command=self.clear_prediction, bg="red", fg="white")
        self.clear_btn.grid(row=0, column=2, padx=5)
        
        self.quit_btn = tk.Button(self.control_frame, text="Quit", command=self.quit_app, bg="white", fg="black")
        self.quit_btn.grid(row=0, column=3, padx=5)
        
        self.text_frame = tk.Frame(root)
        self.text_frame.pack(pady=15)  # Increased padding
    
    # Updated prediction_text with larger size and font
        self.prediction_text = tk.Text(self.text_frame, height=4, width=80, font=("Arial", 20))
        self.prediction_text.grid(row=0, column=0, padx=5, pady=5)
        self.voice_lang_label = tk.Label(self.text_frame, text="Voice Language:")
        self.voice_lang_label.grid(row=0, column=1, padx=5)
        
        self.voice_lang = ttk.Combobox(self.text_frame, values=["English", "Tamil"], state="readonly")
        self.voice_lang.set("English")
        self.voice_lang.grid(row=0, column=2, padx=5)
        
        self.update_frame()

    def speak(self, text, language="en"):
        engine = pyttsx3.init()
        if language == "ta":
            text = translate_to_tamil(text)
        print("Spoke:", text)
        engine.say(text)
        engine.runAndWait()

    def predict_gesture(self, rf_model, features, previous_word=None, ngram_probs=None, is_first_word=False, hand_count=0):
        features = features.reshape(1, -1)
        rf_pred_proba = rf_model.predict_proba(features)[0]
        
      
        if is_first_word:
            allowed_words = STARTER_WORDS
           
            if hand_count == 1:
                allowed_words = [w for w in allowed_words if w in SINGLE_HAND_ACTIONS or w not in DOUBLE_HAND_ACTIONS]
            elif hand_count == 2:
                allowed_words = [w for w in allowed_words if w in DOUBLE_HAND_ACTIONS or w not in SINGLE_HAND_ACTIONS]
            else:
                print(f"Invalid hand_count for starter: {hand_count}")
                return None, 0.0
        else:
            allowed_words = FOLLOWERS.get(previous_word, [])
            allowed_words = [w for w in allowed_words if self.is_valid_continuation(previous_word, w)]
            
            if hand_count == 1:
                allowed_words = [w for w in allowed_words if w in SINGLE_HAND_ACTIONS or w not in DOUBLE_HAND_ACTIONS]
            elif hand_count == 2:
                allowed_words = [w for w in allowed_words if w in DOUBLE_HAND_ACTIONS or w not in SINGLE_HAND_ACTIONS]
            else:
                print(f"Invalid hand_count: {hand_count}")
                return None, 0.0
        
        if not allowed_words:
            print(f"No valid words for hand_count={hand_count} after '{previous_word or 'start'}'")
            return None, 0.0
        
      
        filtered_proba = [rf_pred_proba[i] if ACTIONS[i] in allowed_words else 0.0 for i in range(len(ACTIONS))]
        max_proba = np.max(filtered_proba)
        if max_proba < MIN_CONFIDENCE:
            print(f"Confidence {max_proba:.4f} below threshold {MIN_CONFIDENCE}")
            return None, 0.0
        
        rf_label = ACTIONS[np.argmax(filtered_proba)]
        rf_conf = max_proba
        print(f"Prediction: {rf_label}, Confidence: {rf_conf:.4f}, Hands: {hand_count}, Allowed: {allowed_words}")
        return rf_label, rf_conf

    def is_valid_continuation(self, previous_word, candidate_word):
        if not previous_word:
            return candidate_word in STARTER_WORDS
        current_sequence = " ".join(self.sentence_words + [candidate_word]).lower()
        for sentence in SENTENCES:
            sentence_lower = sentence.lower()
            words = sentence_lower.split()
            for i in range(len(words) - len(self.sentence_words)):
                if " ".join(words[i:i + len(self.sentence_words) + 1]) == current_sequence:
                    return True
        print(f"Invalid continuation: {current_sequence}")
        return False

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame, results_hands = mediapipe_detection(frame, self.hands)
            
            if self.is_recording:
                if not results_hands.multi_hand_landmarks:
                    print("No hand landmarks detected.")
                
                hand_count = len(results_hands.multi_hand_landmarks) if results_hands.multi_hand_landmarks else 0
                features, feature_hand_count = extract_arm_features(None, results_hands)
                last_valid = features if features is not None and np.any(features) else None
                
                if last_valid is not None and hand_count > 0:
                    ready, sequence = self.collector.process_frame(last_valid)
                    if ready and time.time() - self.last_pred_time >= PREDICTION_WINDOW_SECONDS:
                        try:
                            rf_features = preprocess_sequence(sequence, self.embeddings)
                            previous_word = self.sentence_words[-1] if self.sentence_words else None
                            is_first_word = len(self.sentence_words) == 0
                            rf_label, rf_conf = self.predict_gesture(
                                self.rf_model, rf_features, previous_word, self.ngram_probs, is_first_word, hand_count
                            )
                            if rf_label and rf_conf >= MIN_CONFIDENCE:
                                self.predicted_gesture = rf_label
                                if self.analysis_start_time is None:
                                    self.analysis_start_time = time.time()
                                    self.word_predictions = []
                                self.word_predictions.append(rf_label)
                                self.last_pred_time = time.time()
                                
                                if time.time() - self.analysis_start_time >= ANALYSIS_DURATION:
                                    if self.word_predictions:
                                        word_counts = Counter(self.word_predictions)
                                        max_word = max(word_counts, key=word_counts.get)
                                        if self.is_valid_continuation(self.sentence_words[-1] if self.sentence_words else None, max_word):
                                            self.sentence_words.append(max_word)
                                            display_text = " ".join(self.sentence_words)
                                            print(f"Sentence so far: {display_text}")
                                            self.prediction_text.delete(1.0, tk.END)
                                            self.prediction_text.insert(tk.END, display_text)
                                        else:
                                            print(f"Rejected word: {max_word}")
                                        self.word_predictions = []
                                        self.analysis_start_time = None
                        except Exception as e:
                            print(f"Prediction error: {e}")
            
            frame = visualize_landmarks(frame, None, results_hands)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        self.root.after(10, self.update_frame)

    def toggle_recording(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.collector = RealTimeFeatureCollector()
            self.predicted_gesture = ""
            self.word_predictions = []
            self.analysis_start_time = None
            self.last_pred_time = time.time()
            self.start_stop_btn.config(text="Stop", bg="red")
        else:
            self.start_stop_btn.config(text="Start", bg="green")
            if self.sentence_words:
                display_text = " ".join(self.sentence_words)
                self.prediction_text.delete(1.0, tk.END)
                self.prediction_text.insert(tk.END, display_text)

    def speak_prediction(self):
        if self.sentence_words:
            text_to_speak = " ".join(self.sentence_words)
            lang = self.voice_lang.get()
            if lang == "Tamil":
                text_to_speak = translate_to_tamil(text_to_speak)
                tts_lang = 'ta'
            else:
                tts_lang = 'en'
            
            try:
                tts = gTTS(text=text_to_speak, lang=tts_lang)
                audio_bytes = io.BytesIO()
                tts.write_to_fp(audio_bytes)
                audio_bytes.seek(0)
                pygame.mixer.music.load(audio_bytes, 'mp3')
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                print(f"Spoke: {text_to_speak}")
            except Exception as e:
                print(f"Audio playback error: {e}")

    def clear_prediction(self):
        self.predicted_gesture = ""
        self.sentence_words = []
        self.word_predictions = []
        self.analysis_start_time = None
        self.prediction_text.delete(1.0, tk.END)

    def quit_app(self):
        self.cap.release()
        pygame.mixer.quit()
        self.root.quit()

def main():
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
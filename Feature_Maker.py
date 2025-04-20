import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) 
import warnings
warnings.filterwarnings('ignore')  

import cv2
import pandas as pd
import numpy as np
import torch
from tensorflow.keras.applications.xception import Xception, preprocess_input
from transformers import BertTokenizer, BertModel
import librosa
from tqdm import tqdm
import os
import tensorflow_hub as hub

import os
# Force CPU usage to avoid CuDNN version conflicts
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def ensure_feature_directories():
    """Create directories for storing features if they don't exist"""
    directories = ['features/video', 'features/audio', 'features/text']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def pool_features(features):
    """Pool features across temporal dimension using mean pooling"""
    return np.mean(features, axis=0)  # Average across temporal dimension

def save_feature(feature_data, feature_type, dialogue_id, utterance_id):
    """Save feature data to appropriate directory"""
    # Pool features if they have temporal dimension
    if len(feature_data.shape) > 1 and feature_type in ['video', 'audio']:
        feature_data = pool_features(feature_data)
    
    filename = f"{feature_type}_feature_{dialogue_id}_{utterance_id}.npy"
    filepath = os.path.join('features', feature_type, filename)
    if isinstance(feature_data, torch.Tensor):
        feature_data = feature_data.cpu().numpy()
    np.save(filepath, feature_data)
    return filepath

def save_last_processed_id(dialogue_id, utterance_id):
    """Save the last processed dialogue and utterance IDs to a file"""
    last_id = f"{dialogue_id}_{utterance_id}"
    with open('last_processed.txt', 'w') as f:
        f.write(last_id)

def get_last_processed_id():
    """Get the last processed dialogue and utterance IDs from file"""
    try:
        with open('last_processed.txt', 'r') as f:
            last_id = f.read().strip()
            if last_id:
                return last_id
    except FileNotFoundError:
        return None
    return None

def extract_frames(video_path, output_size=(299, 299), frame_skip=5, max_frames=None, verbose=False):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    
    frame_idx = 0
    extracted = 0

    while frame_idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize and convert
        frame = cv2.resize(frame, output_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

        if verbose:
            print(f"Extracted frame {frame_idx}/{total}")

        frame_idx += frame_skip
        extracted += 1
        if max_frames and extracted >= max_frames:
            break

    cap.release()
    return np.array(frames)

def get_video_features_sequence_xception(frames):
    model = Xception(weights='imagenet', include_top=False, pooling='avg')
    frames = np.array(frames, dtype=np.float32)
    frames = preprocess_input(frames)
    features = model.predict(frames, verbose=0)  # shape: (num_frames, 2048)
    return features

def extract_yamnet_features(audio_path):
    # Load YAMNet model from TensorFlow Hub
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

    # Load and preprocess audio with librosa
    waveform, sr = librosa.load(audio_path, sr=16000)
    waveform = waveform.astype(np.float32)

    # Run the model
    scores, embeddings, spectrogram = yamnet_model(waveform)
    return embeddings.numpy()

def extract_textual_features(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
    return cls_embedding.squeeze(0).numpy()  # Shape: (768,)

if __name__ == "__main__":
    # Create feature directories
    ensure_feature_directories()
    
    # Load CSV
    csv_path = "final.csv"
    df = pd.read_csv(csv_path)

    # Load BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()

    # Get last processed ID
    last_processed_id = get_last_processed_id()
    start_processing = False if last_processed_id else True

    if last_processed_id:
        print(f"Resuming from last processed ID: {last_processed_id}")
    else:
        print("Starting from beginning")

    # Iterate over rows with a progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        current_unique_id = f"{row['Dialogue_ID']}_{row['Utterance_ID']}"
        
        # Skip until we reach the last processed ID
        if last_processed_id and not start_processing:
            if current_unique_id == last_processed_id:
                start_processing = True
            continue
        elif last_processed_id and start_processing:
            # Start processing from the next row after last_processed_id
            pass
        
        video_path = row["VideoPath"]
        audio_path = row["AudioPath"]
        text = row["Utterance"]
        label = row["Label"]
        dialogue_id = row["Dialogue_ID"]
        utterance_id = row["Utterance_ID"]

        try:
            print(f"Processing row {idx}: {text} (Dialogue: {dialogue_id}, Utterance: {utterance_id})")

            # Extract and save video features
            frames = extract_frames(video_path)
            video_features = get_video_features_sequence_xception(frames)
            video_path = save_feature(video_features, 'video', dialogue_id, utterance_id)

            # Extract and save audio features
            audio_features = extract_yamnet_features(audio_path)
            audio_path = save_feature(audio_features, 'audio', dialogue_id, utterance_id)

            # Extract and save text features
            text_features = extract_textual_features(text, tokenizer, model)
            text_path = save_feature(text_features, 'text', dialogue_id, utterance_id)

            # Save last processed ID
            save_last_processed_id(dialogue_id, utterance_id)

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            # Clear any cached tensors
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

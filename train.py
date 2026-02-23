#!/usr/bin/env python3
"""
(V1.65Hands)

V2.1 Hands Only - Snapshot + Shift + Hand Dropout

Changes from V2.0:
- MODE: Switched to HANDS ONLY (No shoulders/body/face).
- INPUT: 126 features (21*3 Left + 21*3 Right).
- LOGIC: Augmentations updated to handle data without pose landmarks.
"""
import os
import time
import json
import argparse
import random
import math
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

try:
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import holistic as mp_holistic
except ImportError:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_holistic = mp.solutions.holistic

# ---------------------------
# Config
# ---------------------------
class Config:
    DATA_PATH = "G:\\fyp 2\\Dataset4_fixed"
    SAVE_DIR = "models_snapshot_handsonly" 
    NPY_CACHE_DIR = "npy_cache_handsonly" 
    
    # --- CRITICAL CHANGE ---
    USE_HANDS_ONLY = True  # Set to True to ignore shoulders/body
    
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    EPOCHS = 100
    EARLY_STOP_PATIENCE = 15
    LR_SCHEDULER_PATIENCE = 6
    
    # Snapshot Config
    MIN_SEQUENCE_LENGTH = 5      
    MAX_SEQUENCE_LENGTH = 60     
    TARGET_SEQUENCE_LENGTH = 10
    
    STATIC_MOTION_THRESHOLD = 0.008
    STATIC_SAMPLES_PER_VIDEO = 20
    
    STRIDE_TRAIN = 6
    STRIDE_VAL_TEST = 5
    
    AUGMENT_TRAIN = True
    NOISE_STD = 0.003
    
    # --- AUGMENTATIONS ---
    ROTATION_DEGREES = 15
    SCALE_RANGE = 0.20  
    SHIFT_RANGE = 0.10 
    HAND_DROPOUT_PROB = 0.5 
    
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.5
    BIDIRECTIONAL = True
    USE_ATTENTION = True
    SPATIAL_HIDDEN = 512
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MIRROR_TRAIN = True
    BALANCE_AFTER_LOAD = True
    FORCE_EXTRACT = False

cfg = Config()
os.makedirs(cfg.SAVE_DIR, exist_ok=True)
os.makedirs(cfg.NPY_CACHE_DIR, exist_ok=True)

# ---------------------------
# Feature extractors
# ---------------------------
class HandsOnlyFeatureExtractor:
    def __init__(self):
        self.mp_hands = mp_hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_dim = 21*3
        self.feature_dim = 2 * self.hand_dim

    def extract(self, frame):
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        left_hand = np.zeros(self.hand_dim, dtype=np.float32)
        right_hand = np.zeros(self.hand_dim, dtype=np.float32)
        if getattr(results, "multi_hand_landmarks", None) and getattr(results, "multi_handedness", None):
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten().astype(np.float32)
                if hand_label == 'Left':
                    left_hand = landmarks
                else:
                    right_hand = landmarks
        return np.concatenate([left_hand, right_hand]).astype(np.float32)

    def close(self):
        self.hands.close()

class HolisticFeatureExtractor:
    def __init__(self):
        self.mp_holistic = mp_holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False, 
            model_complexity=1,
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        self.pose_dim = 33*4
        self.hand_dim = 21*3
        self.feature_dim = self.pose_dim + 2*self.hand_dim

    def extract(self, frame):
        res = self.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            pose = np.array([[lm.x,lm.y,lm.z,lm.visibility] for lm in res.pose_landmarks.landmark]).flatten().astype(np.float32)
        else:
            pose = np.zeros(self.pose_dim, dtype=np.float32)
        if res.left_hand_landmarks:
            left = np.array([[lm.x,lm.y,lm.z] for lm in res.left_hand_landmarks.landmark]).flatten().astype(np.float32)
        else:
            left = np.zeros(self.hand_dim, dtype=np.float32)
        if res.right_hand_landmarks:
            right = np.array([[lm.x,lm.y,lm.z] for lm in res.right_hand_landmarks.landmark]).flatten().astype(np.float32)
        else:
            right = np.zeros(self.hand_dim, dtype=np.float32)
        return np.concatenate([pose, left, right]).astype(np.float32)

    def close(self):
        self.holistic.close()

# ---------------------------
# NPY Pipeline
# ---------------------------
def extract_video_to_npy(video_path, npy_path, extractor):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        try:
            features = extractor.extract(frame)
            frames.append(features)
        except Exception: continue
    cap.release()
    if len(frames) > 0:
        frames_array = np.array(frames, dtype=np.float32)
        np.save(npy_path, frames_array)
        return len(frames)
    return 0

def preprocess_dataset(data_path, npy_cache_dir, force_extract=False, use_hands_only=False):
    print("\n" + "="*60)
    print("PREPROCESSING: Extracting landmarks to .npy files")
    print("MODE:", "Hands Only (126 features)" if use_hands_only else "Holistic (258 features)")
    print("="*60)
    extractor = HandsOnlyFeatureExtractor() if use_hands_only else HolisticFeatureExtractor()
    video_to_npy = {}
    splits = ['train', 'val', 'test']
    try:
        for split in splits:
            split_path = os.path.join(data_path, split)
            if not os.path.exists(split_path): continue
            print(f"\n[{split.upper()}]")
            classes = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
            for cls in classes:
                cls_path = os.path.join(split_path, cls)
                video_files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))]
                print(f"  {cls}: {len(video_files)} videos", end=' ')
                cls_extracted = 0
                for video_file in video_files:
                    video_path = os.path.join(cls_path, video_file)
                    npy_subdir = os.path.join(npy_cache_dir, split, cls)
                    os.makedirs(npy_subdir, exist_ok=True)
                    npy_filename = os.path.splitext(video_file)[0] + '.npy'
                    npy_path = os.path.join(npy_subdir, npy_filename)
                    if os.path.exists(npy_path) and not force_extract:
                        video_to_npy[video_path] = npy_path
                    else:
                        num_frames = extract_video_to_npy(video_path, npy_path, extractor)
                        if num_frames > 0:
                            video_to_npy[video_path] = npy_path
                            cls_extracted += 1
                print(f"(extracted {cls_extracted})" if cls_extracted > 0 else "(cached)")
    finally:
        extractor.close()
    return video_to_npy

def load_npy_landmarks(npy_path):
    return np.load(npy_path).astype(np.float32)

# ---------------------------
# Normalization
# ---------------------------
def normalize_sequence_holistic(seq, feature_dim, pose_dim=33*4, hand_dim=21*3):
    T = seq.shape[0]
    coords = seq.copy().reshape(T, -1)
    normed = np.zeros_like(coords)
    for t in range(T):
        frame = coords[t]
        try:
            pose = frame[:pose_dim].reshape(-1,4)
            left_sh = pose[11][:2]; right_sh = pose[12][:2]
            center = (left_sh + right_sh) / 2.0
            shoulder_dist = np.linalg.norm(left_sh - right_sh) + 1e-6
            scale = shoulder_dist
            all_coords = []
            for p in pose:
                x = (p[0] - center[0]) / scale; y = (p[1] - center[1]) / scale; z = p[2] / scale
                all_coords.extend([x,y,z,p[3]])
            left = frame[pose_dim:pose_dim+hand_dim].reshape(-1,3)
            for p in left:
                all_coords.extend([(p[0]-center[0])/scale, (p[1]-center[1])/scale, p[2]/scale])
            right = frame[pose_dim+hand_dim:pose_dim+2*hand_dim].reshape(-1,3)
            for p in right:
                all_coords.extend([(p[0]-center[0])/scale, (p[1]-center[1])/scale, p[2]/scale])
            normed[t,:len(all_coords)] = np.array(all_coords)[:coords.shape[1]]
        except: normed[t] = frame
    return normed

def normalize_sequence_hands_only(seq, hand_dim=21*3):
    T = seq.shape[0]
    normed = np.zeros_like(seq)
    for t in range(T):
        frame = seq[t]
        left_hand = frame[:hand_dim].reshape(-1, 3)
        right_hand = frame[hand_dim:2*hand_dim].reshape(-1, 3)
        normalized_coords = []
        for hand in [left_hand, right_hand]:
            if np.all(np.isclose(hand, 0.0)):
                normalized_coords.extend(hand.flatten())
            else:
                wrist = hand[0].copy(); hand_centered = hand - wrist
                palm_size = np.linalg.norm(hand_centered[9]) + 1e-6
                hand_normalized = hand_centered / palm_size
                normalized_coords.extend(hand_normalized.flatten())
        normed[t] = np.array(normalized_coords)
    return normed

def normalize_sequence(seq, feature_dim, use_hands_only=False):
    if use_hands_only: return normalize_sequence_hands_only(seq, hand_dim=21*3)
    return normalize_sequence_holistic(seq, feature_dim, pose_dim=33*4, hand_dim=21*3)

def fix_hand_consistency(seq, use_hands_only=False, pose_dim=33*4, hand_dim=21*3):
    seq2 = seq.copy()
    if use_hands_only:
        left = seq2[:, :hand_dim]; right = seq2[:, hand_dim:2*hand_dim]
    else:
        left = seq2[:, pose_dim:pose_dim+hand_dim]; right = seq2[:, pose_dim+hand_dim:pose_dim+2*hand_dim]
    left_empty = np.all(np.isclose(left, 0.0), axis=1).all()
    right_empty = np.all(np.isclose(right, 0.0), axis=1).all()
    if left_empty and not right_empty:
        if use_hands_only: seq2[:, :hand_dim] = right
        else: seq2[:, pose_dim:pose_dim+hand_dim] = right
    elif right_empty and not left_empty:
        if use_hands_only: seq2[:, hand_dim:2*hand_dim] = left
        else: seq2[:, pose_dim+hand_dim:pose_dim+2*hand_dim] = left
    return seq2

# ---------------------------
# AUGMENTATIONS (Updated for Hands-Only Support)
# ---------------------------
def add_gaussian_noise(seq, std=0.003):
    return seq + np.random.normal(0, std, seq.shape).astype(np.float32)

def rotate_sequence(seq, max_degrees=15):
    theta = np.deg2rad(np.random.uniform(-max_degrees, max_degrees))
    c, s = np.cos(theta), np.sin(theta)
    new_seq = seq.copy()
    
    is_holistic = (seq.shape[1] > 200)
    pose_dim = 33*4 if is_holistic else 0
    hand_dim = 21*3
    
    if is_holistic:
        for i in range(0, pose_dim, 4):
            x = new_seq[:, i]; y = new_seq[:, i+1]
            new_seq[:, i] = x*c - y*s; new_seq[:, i+1] = x*s + y*c
            
    # Hands (both holistic and hands-only)
    start_hands = pose_dim
    for i in range(start_hands, seq.shape[1], 3):
        if i+1 < seq.shape[1]:
            x = new_seq[:, i]; y = new_seq[:, i+1]
            new_seq[:, i] = x*c - y*s; new_seq[:, i+1] = x*s + y*c
    return new_seq

def scale_sequence(seq, scale_range=0.2):
    factor = np.random.uniform(1-scale_range, 1+scale_range)
    return seq * factor

def shift_sequence(seq, max_shift=0.1):
    dx = np.random.uniform(-max_shift, max_shift)
    dy = np.random.uniform(-max_shift, max_shift)
    new_seq = seq.copy()
    
    is_holistic = (seq.shape[1] > 200)
    pose_dim = 33*4 if is_holistic else 0
    hand_dim = 21*3
    
    if is_holistic:
        for i in range(0, pose_dim, 4):
            new_seq[:, i] += dx; new_seq[:, i+1] += dy
            
    # Hands
    start_hands = pose_dim
    for i in range(start_hands, seq.shape[1], 3):
        if i+1 < seq.shape[1]:
            new_seq[:, i] += dx; new_seq[:, i+1] += dy
    return new_seq

def horizontal_flip(seq, pose_dim=33*4, hand_dim=21*3):
    seq2 = seq.copy()
    is_holistic = (seq.shape[1] > 200)
    
    if is_holistic: 
        for i in range(0, pose_dim, 4): seq2[:, i] = -seq2[:, i] 
        left = seq2[:, pose_dim:pose_dim+hand_dim].copy()
        right = seq2[:, pose_dim+hand_dim:pose_dim+2*hand_dim].copy()
        for i in range(0, hand_dim, 3):
            left[:, i] = -left[:, i]; right[:, i] = -right[:, i]
        seq2[:, pose_dim:pose_dim+hand_dim] = right
        seq2[:, pose_dim+hand_dim:pose_dim+2*hand_dim] = left
    else: # Hands Only Flip
        # For hands only, we don't have the pose dim offset
        # Left hand is 0:hand_dim, Right is hand_dim:end
        left = seq2[:, :hand_dim].copy()
        right = seq2[:, hand_dim:].copy()
        for i in range(0, hand_dim, 3):
            left[:, i] = -left[:, i]; right[:, i] = -right[:, i]
        # Swap
        seq2[:, :hand_dim] = right
        seq2[:, hand_dim:] = left
        
    return seq2

def random_hand_dropout(seq, probability=0.5):
    """
    Randomly zeros out Left or Right hand.
    Handles both Holistic and Hands-Only layouts automatically.
    """
    if np.random.rand() > probability: return seq 
    
    seq_aug = seq.copy()
    is_holistic = (seq.shape[1] > 200)
    pose_dim = 33*4 if is_holistic else 0
    hand_dim = 21*3
    
    start_left = pose_dim
    start_right = pose_dim + hand_dim
    
    drop_idx = np.random.randint(0, 2) # 0=Left, 1=Right
    
    if drop_idx == 0: # Drop Left
        seq_aug[:, start_left : start_right] = 0.0
    else: # Drop Right
        seq_aug[:, start_right : ] = 0.0
    return seq_aug

# ---------------------------
# Snapshot Logic
# ---------------------------
def extract_sequences_smart(frames, cfg, is_training=True):
    num_frames = len(frames)
    if num_frames < cfg.MIN_SEQUENCE_LENGTH: return [], False
    trim_start = int(num_frames * 0.20) 
    trim_end = int(num_frames * 0.80)
    if trim_end - trim_start < cfg.TARGET_SEQUENCE_LENGTH: valid_frames = np.array(frames)
    else: valid_frames = np.array(frames[trim_start:trim_end])

    sequences = []
    L = len(valid_frames); tgt = cfg.TARGET_SEQUENCE_LENGTH
    
    if is_training:
        for _ in range(cfg.STATIC_SAMPLES_PER_VIDEO):
            if L <= tgt: seq = valid_frames 
            else:
                start = random.randint(0, L - tgt)
                seq = valid_frames[start : start + tgt]
            sequences.append(seq)
    else:
        stride = max(1, tgt // 2) 
        for i in range(0, L - tgt + 1, stride):
            seq = valid_frames[i : i + tgt]
            sequences.append(seq)
    if not sequences and L > 0: sequences.append(valid_frames[:min(L, tgt)])
    return sequences, True 

# ---------------------------
# Loading
# ---------------------------
def load_folder_sequences_from_npy(base_path, split, npy_cache_dir, cfg, augment=False, mirror=False):
    split_path = os.path.join(base_path, split)
    if not os.path.exists(split_path): return [], [], [], {}
    classes = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
    sequences = []; labels = []; sign_types = {}
    
    # Dynamic Feature Dim based on Config
    feature_dim = 2 * 21 * 3 if cfg.USE_HANDS_ONLY else 33 * 4 + 2 * 21 * 3
    
    print(f"Loading {split} sequences from NPY cache...")
    for idx, cls in enumerate(classes):
        npy_cls_path = os.path.join(npy_cache_dir, split, cls)
        if not os.path.exists(npy_cls_path): continue
        npy_files = [f for f in os.listdir(npy_cls_path) if f.endswith('.npy')]
        for npy_file in npy_files:
            npy_path = os.path.join(npy_cls_path, npy_file)
            frames = load_npy_landmarks(npy_path)
            if len(frames) < cfg.MIN_SEQUENCE_LENGTH: continue
            seqs, _ = extract_sequences_smart(frames, cfg, is_training=(split=='train'))
            for seq in seqs:
                seq = normalize_sequence(seq, feature_dim, use_hands_only=cfg.USE_HANDS_ONLY)
                seq = fix_hand_consistency(seq, use_hands_only=cfg.USE_HANDS_ONLY)
                sequences.append(seq); labels.append(idx)
                
                if augment and split == 'train':
                    # Rotate
                    aug_rot = rotate_sequence(seq.copy(), cfg.ROTATION_DEGREES)
                    sequences.append(aug_rot); labels.append(idx)
                    # Scale
                    aug_scale = scale_sequence(seq.copy(), cfg.SCALE_RANGE)
                    sequences.append(aug_scale); labels.append(idx)
                    # Shift
                    aug_shift = shift_sequence(seq.copy(), cfg.SHIFT_RANGE)
                    sequences.append(aug_shift); labels.append(idx)
                    # Hand Dropout
                    aug_drop = random_hand_dropout(seq.copy(), probability=cfg.HAND_DROPOUT_PROB)
                    sequences.append(aug_drop); labels.append(idx)
                    # Noise
                    aug_seq = add_gaussian_noise(seq.copy())
                    sequences.append(aug_seq); labels.append(idx)
                    # Mirror
                    if mirror:
                        aug_seq = horizontal_flip(seq.copy())
                        sequences.append(aug_seq); labels.append(idx)
        sign_types[cls] = 'snapshot'
        print(f"  {cls}: {len([l for l in labels if l == idx])} snapshots")
    return sequences, labels, classes, sign_types

class HybridDataset(Dataset):
    def __init__(self, sequences, labels, target_length):
        self.sequences = sequences
        self.labels = labels
        self.target_length = target_length
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx):
        seq = self.sequences[idx]; lab = self.labels[idx]; L = len(seq)
        if L < self.target_length:
            padding = np.zeros((self.target_length - L, seq.shape[1]), dtype=np.float32)
            seq_p = np.vstack([seq, padding])
        else: seq_p = seq[:self.target_length]
        return torch.from_numpy(seq_p).float(), min(L, self.target_length), lab

def collate_fn(batch):
    seqs, lengths, labels = zip(*batch)
    seqs = torch.stack(seqs); lengths = torch.tensor(lengths); labels = torch.tensor(labels)
    lengths, perm = lengths.sort(0, descending=True)
    seqs = seqs[perm]; labels = labels[perm]
    return seqs, lengths, labels

# ---------------------------
# Balancing
# ---------------------------
def balance_classes(seqs, labels):
    class_to_items = defaultdict(list)
    for s, l in zip(seqs, labels):
        class_to_items[l].append(s)
    counts = {k: len(v) for k, v in class_to_items.items()}
    max_count = max(counts.values())
    new_seqs = []
    new_labels = []
    for l, items in class_to_items.items():
        if len(items) < max_count:
            add = random.choices(items, k=max_count - len(items))
            items = items + add
        random.shuffle(items)
        new_seqs.extend(items)
        new_labels.extend([l] * max_count)
    combined = list(zip(new_seqs, new_labels))
    random.shuffle(combined)
    new_seqs[:], new_labels[:] = zip(*combined)
    return list(new_seqs), list(new_labels)

# ---------------------------
# Model
# ---------------------------
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
    def forward(self, lstm_output, lengths):
        scores = self.attention(lstm_output).squeeze(-1)
        max_len = lstm_output.size(1)
        mask = torch.arange(max_len, device=lstm_output.device)[None,:] < lengths[:,None]
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(scores, dim=1)
        ctx = torch.bmm(attn.unsqueeze(1), lstm_output).squeeze(1)
        return ctx, attn

class HybridSignRecognitionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, spatial_hidden=512, dropout=0.5, bidirectional=True, use_attention=True):
        super().__init__()
        self.hidden_size = hidden_size; self.num_layers = num_layers
        self.bidirectional = bidirectional; self.use_attention = use_attention
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0, bidirectional=bidirectional)
        lstm_out = hidden_size * (2 if bidirectional else 1)
        if use_attention: self.att = AttentionLayer(lstm_out)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_size, spatial_hidden), nn.LayerNorm(spatial_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(spatial_hidden, spatial_hidden // 2), nn.LayerNorm(spatial_hidden // 2), nn.ReLU(), nn.Dropout(dropout)
        )
        fusion_input = lstm_out + spatial_hidden // 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, (h, c) = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        if self.use_attention: temporal_features, _ = self.att(out, lengths)
        else:
            if self.bidirectional:
                h = h.view(self.num_layers, 2, x.size(0), self.hidden_size)
                temporal_features = torch.cat([h[-1, 0], h[-1, 1]], dim=1)
            else: temporal_features = h[-1]
        spatial_input = torch.mean(x, dim=1)
        spatial_features = self.spatial_encoder(spatial_input)
        combined = torch.cat([temporal_features, spatial_features], dim=1)
        return self.fusion(combined)

# ---------------------------
# Main
# ---------------------------
def train_main(data_path, cfg, mirror=False):
    start_time = time.time()
    preprocess_dataset(data_path, cfg.NPY_CACHE_DIR, force_extract=cfg.FORCE_EXTRACT, use_hands_only=cfg.USE_HANDS_ONLY)
    print(f"Preprocessing took: {time.time() - start_time:.2f}s\n")

    X_train, y_train, classes, sign_types = load_folder_sequences_from_npy(data_path, 'train', cfg.NPY_CACHE_DIR, cfg, augment=cfg.AUGMENT_TRAIN, mirror=mirror)
    X_val, y_val, _, _ = load_folder_sequences_from_npy(data_path, 'val', cfg.NPY_CACHE_DIR, cfg, augment=False, mirror=False)
    X_test, y_test, _, _ = load_folder_sequences_from_npy(data_path, 'test', cfg.NPY_CACHE_DIR, cfg, augment=False, mirror=False)
    
    if not X_train: raise RuntimeError("No training data found")
    print(f"Classes: {classes}")

    models_dir = Path(cfg.SAVE_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump({'labels': classes, 'num_classes': len(classes), 'sign_types': sign_types}, f, indent=2)

    if cfg.BALANCE_AFTER_LOAD:
        print("Balancing training data...")
        X_train, y_train = balance_classes(X_train, y_train)

    unique_labels = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=y_train)
    class_weights_tensor = torch.ones(len(classes), dtype=torch.float32).to(cfg.DEVICE)
    for i, cls_idx in enumerate(unique_labels): class_weights_tensor[cls_idx] = float(class_weights[i])

    train_ds = HybridDataset(X_train, y_train, cfg.TARGET_SEQUENCE_LENGTH)
    val_ds = HybridDataset(X_val, y_val, cfg.TARGET_SEQUENCE_LENGTH) if X_val else None
    test_ds = HybridDataset(X_test, y_test, cfg.TARGET_SEQUENCE_LENGTH) if X_test else None
    
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn) if val_ds else None

    # Dynamic input size calculation
    feature_dim = 2 * 21 * 3 if cfg.USE_HANDS_ONLY else 33 * 4 + 2 * 21 * 3
    
    model = HybridSignRecognitionModel(
        input_size=feature_dim, hidden_size=cfg.HIDDEN_SIZE, num_layers=cfg.NUM_LAYERS,
        num_classes=len(classes), spatial_hidden=cfg.SPATIAL_HIDDEN, dropout=cfg.DROPOUT,
        bidirectional=cfg.BIDIRECTIONAL, use_attention=cfg.USE_ATTENTION
    ).to(cfg.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=cfg.LR_SCHEDULER_PATIENCE, factor=0.5)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    best_val = -1.0; patience = 0
    print("\n" + "="*60); print("🚀 SNAPSHOT TRAINING (Hands-Only) STARTED"); print("="*60 + "\n")

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        running_loss = 0.0; correct = 0; total = 0
        for seqs, lengths, labels in train_loader:
            seqs = seqs.to(cfg.DEVICE); lengths = lengths.to(cfg.DEVICE); labels = labels.to(cfg.DEVICE)
            optimizer.zero_grad(); outputs = model(seqs, lengths); loss = criterion(outputs, labels)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            running_loss += loss.item() * seqs.size(0); _, preds = torch.max(outputs, 1)
            total += labels.size(0); correct += (preds == labels).sum().item()
        train_loss = running_loss / total; train_acc = 100.0 * correct / total

        if val_loader:
            model.eval()
            val_total = 0; val_correct = 0; val_running_loss = 0.0
            with torch.no_grad():
                for seqs, lengths, labels in val_loader:
                    seqs = seqs.to(cfg.DEVICE); lengths = lengths.to(cfg.DEVICE); labels = labels.to(cfg.DEVICE)
                    outputs = model(seqs, lengths); loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * seqs.size(0); _, preds = torch.max(outputs, 1)
                    val_total += labels.size(0); val_correct += (preds == labels).sum().item()
            val_loss = val_running_loss / val_total; val_acc = 100.0 * val_correct / val_total; scheduler.step(val_loss)
            print(f"[{epoch}/{cfg.EPOCHS}] Train: {train_acc:.2f}% | Val: {val_acc:.2f}% (Loss: {val_loss:.4f})")
            
            if val_acc > best_val:
                best_val = val_acc; patience = 0
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'config': {'input_size': feature_dim, 'hidden_size': cfg.HIDDEN_SIZE, 'num_layers': cfg.NUM_LAYERS, 'num_classes': len(classes), 'spatial_hidden': cfg.SPATIAL_HIDDEN, 'dropout': cfg.DROPOUT, 'bidirectional': cfg.BIDIRECTIONAL, 'use_attention': cfg.USE_ATTENTION}, 'label_names': classes}, models_dir / "best_model.pth")
                print(f"   ✅ Saved best model ({val_acc:.2f}%)")
            else:
                patience += 1
                if patience >= cfg.EARLY_STOP_PATIENCE: print("Early stopping."); break
        else: print(f"[{epoch}/{cfg.EPOCHS}] Train: {train_acc:.2f}%")

    if test_ds:
        print("\n" + "="*60); print("🧪 TEST EVALUATION"); print("="*60)
        best_path = models_dir / "best_model.pth"
        if best_path.exists(): model.load_state_dict(torch.load(best_path, map_location=cfg.DEVICE)['model_state_dict'])
        model.eval()
        test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        all_preds = []; all_labels = []
        with torch.no_grad():
            for seqs, lengths, labels in test_loader:
                seqs = seqs.to(cfg.DEVICE); lengths = lengths.to(cfg.DEVICE); labels = labels.to(cfg.DEVICE)
                outputs = model(seqs, lengths); _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
        print("\nClassification Report:"); print(classification_report(all_labels, all_preds, target_names=classes))
        print("\nConfusion Matrix:"); print(confusion_matrix(all_labels, all_preds))
        with open(models_dir / "test_results.json", 'w') as f: json.dump({'classification_report': classification_report(all_labels, all_preds, target_names=classes, output_dict=True), 'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()}, f, indent=2)

if __name__ == "__main__":
    train_main(cfg.DATA_PATH, cfg, mirror=cfg.MIRROR_TRAIN)
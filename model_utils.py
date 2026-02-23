import torch
import torch.nn as nn
import numpy as np
import cv2

try:
    from mediapipe.python.solutions import hands as mp_hands
except ImportError:
    import mediapipe as mp
    mp_hands = mp.solutions.hands

# --- CONFIGURATION MATCHING V1.65 ---
HAND_DIM = 21 * 3
FEATURE_DIM = 126  # 2 * HAND_DIM

class HandsOnlyFeatureExtractor:
    def __init__(self, min_conf=0.5):
        self.mp_hands = mp_hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_conf,
            min_tracking_confidence=min_conf
        )

    def extract(self, frame):
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        left_hand = np.zeros(HAND_DIM, dtype=np.float32)
        right_hand = np.zeros(HAND_DIM, dtype=np.float32)
        hands_present = False

        if results.multi_hand_landmarks and results.multi_handedness:
            hands_present = True
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                # Flatten landmarks [x, y, z]
                arr = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten().astype(np.float32)
                
                if label == 'Left':
                    left_hand = arr
                else:
                    right_hand = arr
                    
        features = np.concatenate([left_hand, right_hand]).astype(np.float32)
        return features, results, hands_present

def normalize_frame_hands_only(frame):
    """
    Normalization logic matching V1.65 Training.
    Centers hands on wrist and scales by palm size.
    """
    try:
        f = frame.copy()
        left_hand = f[:HAND_DIM].reshape(-1, 3)
        right_hand = f[HAND_DIM:].reshape(-1, 3)
        norm_coords = []
        
        for hand in [left_hand, right_hand]:
            if np.all(np.isclose(hand, 0.0)):
                norm_coords.extend(hand.flatten())
            else:
                wrist = hand[0].copy()
                hand_centered = hand - wrist
                palm_size = np.linalg.norm(hand_centered[9]) + 1e-6
                hand_normalized = hand_centered / palm_size
                norm_coords.extend(hand_normalized.flatten())
        
        normed = np.array(norm_coords, dtype=np.float32)
        
        # Consistency Check (Swap if left is empty but right is full)
        left = normed[:HAND_DIM]
        right = normed[HAND_DIM:]
        if np.all(np.isclose(left, 0.0)) and not np.all(np.isclose(right, 0.0)):
            normed[:HAND_DIM] = right
        elif np.all(np.isclose(right, 0.0)) and not np.all(np.isclose(left, 0.0)):
            normed[HAND_DIM:] = left
            
        return normed
    except Exception:
        return frame

# --- MODEL ARCHITECTURE ---
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
    def __init__(self, input_size=126, hidden_size=128, num_layers=2, num_classes=10, 
                 spatial_hidden=512, dropout=0.5, bidirectional=True, use_attention=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                           dropout=dropout if num_layers>1 else 0, bidirectional=bidirectional)
        
        lstm_out = hidden_size * (2 if bidirectional else 1)
        if use_attention:
            self.att = AttentionLayer(lstm_out)
        
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_size, spatial_hidden), 
            nn.LayerNorm(spatial_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(spatial_hidden, spatial_hidden // 2), 
            nn.LayerNorm(spatial_hidden // 2), nn.ReLU(), nn.Dropout(dropout)
        )
        
        fusion_input = lstm_out + spatial_hidden // 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x, lengths):
        # Pack padded sequence
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, (h, c) = self.lstm(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        
        if self.use_attention: 
            temporal_features, _ = self.att(out, lengths)
        else:
            if self.bidirectional:
                h = h.view(self.num_layers, 2, x.size(0), self.hidden_size)
                temporal_features = torch.cat([h[-1, 0], h[-1, 1]], dim=1)
            else:
                temporal_features = h[-1]
            
        spatial_input = torch.mean(x, dim=1)
        spatial_features = self.spatial_encoder(spatial_input)
        combined = torch.cat([temporal_features, spatial_features], dim=1)
        return self.fusion(combined)
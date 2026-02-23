import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import time
from collections import deque, Counter
from pathlib import Path
import queue
from gtts import gTTS
import io

# --- CUSTOM IMPORTS ---
from model_utils import HybridSignRecognitionModel, HandsOnlyFeatureExtractor, normalize_frame_hands_only
from urdu_sentence_generator import UrduSentenceGenerator

# --- STUN SERVERS FOR CLOUD ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- CONSTANTS ---
BASE_DIR = Path(__file__).parent
MODEL_PATH = str(BASE_DIR / "models_snapshot_handsonly" / "best_model.pth")
LABELS_PATH = str(BASE_DIR / "models_snapshot_handsonly" / "labels.json")
TARGET_SEQUENCE_LENGTH = 10
CONFIDENCE_THRESHOLD = 0.75

# Page Setup
st.set_page_config(page_title="IsharaAI", layout="wide")

# --- SESSION STATE INITIALIZATION ---
if "word_queue" not in st.session_state:
    st.session_state.word_queue = queue.Queue()

if "detected_words" not in st.session_state: 
    st.session_state.detected_words = []
if "history" not in st.session_state: 
    st.session_state.history = []
if "sentence_maker" not in st.session_state:
    st.session_state.sentence_maker = UrduSentenceGenerator()
if "processed_words" not in st.session_state:
    st.session_state.processed_words = set()

if "model" not in st.session_state: 
    st.session_state["model"] = None
if "labels" not in st.session_state: 
    st.session_state["labels"] = []
if "device" not in st.session_state: 
    st.session_state["device"] = torch.device("cpu")
if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = False

# --- UTILS ---
def speak_text(text, lang='ur'):
    """Converts Urdu text to speech and returns audio bytes."""
    try:
        tts = gTTS(text=text, lang=lang)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        return fp.getvalue()
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

# --- VIDEO PROCESSOR ---
class PyTorchProcessor(VideoProcessorBase):
    def __init__(self, model, labels, device, sensitivity, out_queue):
        self.model = model
        self.labels = labels
        self.device = device
        self.sensitivity = sensitivity
        self.out_queue = out_queue
        self.extractor = HandsOnlyFeatureExtractor()
        self.buf = deque(maxlen=TARGET_SEQUENCE_LENGTH)
        self.votes = deque(maxlen=5)
        self.cooldown = 0
        self.last_label = None
        self.last_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        if self.model is None:
            h, w, _ = img.shape
            cv2.rectangle(img, (0, h-60), (w, h), (50, 50, 50), -1)
            cv2.putText(img, "MODEL NOT LOADED", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        features, results, hands_present = self.extractor.extract(img)
        
        is_active = False
        if hands_present and results.multi_hand_landmarks:
            try:
                if any(hand_lm.landmark[0].y < self.sensitivity for hand_lm in results.multi_hand_landmarks):
                    is_active = True
            except (AttributeError, IndexError):
                pass
        
        norm_feat = normalize_frame_hands_only(features)
        
        if is_active:
            self.buf.append(norm_feat)
        else:
            self.votes.clear()
            if self.buf: 
                self.buf.append(self.buf[-1])
            else: 
                self.buf.append(norm_feat)
            
        smooth_label = "--"
        
        if len(self.buf) == TARGET_SEQUENCE_LENGTH and is_active:
            seq = np.array(self.buf, dtype=np.float32)
            x = torch.from_numpy(seq).unsqueeze(0).to(self.device)
            lengths = torch.LongTensor([TARGET_SEQUENCE_LENGTH]).to(self.device)
            
            with torch.no_grad():
                logits = self.model(x, lengths)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                
            pred = int(probs.argmax())
            conf = float(probs[pred])
            self.votes.append((pred, conf))
            
            counts = Counter(p for p, c in self.votes)
            if counts:
                best_pred = counts.most_common(1)[0][0]
                avg_conf = np.mean([c for p, c in self.votes if p == best_pred])
                
                if avg_conf > CONFIDENCE_THRESHOLD and len(self.votes) >= 3 and self.cooldown == 0:
                    final_label = self.labels[best_pred]
                    
                    if final_label in ["Theek Hu", "Thumbs Up"] and results.multi_hand_landmarks:
                        try:
                            for hand_lm in results.multi_hand_landmarks:
                                if hand_lm.landmark[8].z < (hand_lm.landmark[5].z - 0.05):
                                    final_label = "Aap"
                        except (AttributeError, IndexError):
                            pass
                    
                    now = time.time()
                    if self.last_label != final_label and (now - self.last_time) > 1.0:
                        self.last_label = final_label
                        self.last_time = now
                        self.cooldown = 10
                        
                        try:
                            self.out_queue.put_nowait(final_label)
                        except Exception as e:
                            print(f"❌ Queue error: {e}")
                    
                    smooth_label = f"{final_label} ({avg_conf:.2f})"
        
        if self.cooldown > 0: 
            self.cooldown -= 1
        
        h, w, _ = img.shape
        cv2.rectangle(img, (0, h-60), (w, h), (0, 0, 0), -1)
        color = (0, 255, 0) if is_active else (0, 0, 255)
        text = f"SIGN: {smooth_label}" if is_active else "RESTING"
        cv2.putText(img, text, (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI LAYOUT ---
st.title("✋ IsharaAI - Urdu Sign Language")

with st.sidebar:
    st.header("⚙️ Settings")
    sensitivity = st.slider("Hand Height Sensitivity", 0.5, 1.0, 0.9)
    st.divider()
    
    if st.button("🔄 Load Model", type="primary"):
        with st.spinner("Loading model..."):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
                try:
                    checkpoint = torch.load(MODEL_PATH, map_location=device)
                    with open(LABELS_PATH, 'r', encoding='utf-8') as f: 
                        labels = json.load(f)['labels']
                    
                    model = HybridSignRecognitionModel(
                        input_size=126, 
                        num_classes=len(labels),
                        hidden_size=int(checkpoint.get('config', {}).get('hidden_size', 128))
                    )
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    model.to(device).eval()
                    
                    st.session_state.update({
                        "model": model, "labels": labels, "device": device, "model_loaded": True
                    })
                    st.success("✅ Model Loaded!")
                    st.balloons()
                    time.sleep(0.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            else:
                st.error("❌ Model files not found!")
    
    st.divider()
    if st.session_state.get("model_loaded", False):
        st.success("✅ Model Ready")
        with st.expander("📋 Classes"):
            for i, label in enumerate(st.session_state["labels"]):
                st.caption(f"{i+1}. {label}")
    else:
        st.warning("⚠️ Model not loaded")

# Drain queue
q = st.session_state.word_queue
while not q.empty():
    try:
        word = q.get_nowait().strip()
        if word and (not st.session_state.detected_words or word != st.session_state.detected_words[-1]):
            st.session_state.detected_words.append(word)
    except queue.Empty:
        break

# Main Area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Detection")
    if st.session_state.get("model_loaded", False):
        webrtc_streamer(
            key="sign-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: PyTorchProcessor(
                st.session_state.model, st.session_state.labels, 
                st.session_state.device, sensitivity, st.session_state.word_queue
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
    else:
        st.info("👈 Load the model from sidebar")

with col2:
    st.subheader("📝 Detected Words")
    if st.session_state.detected_words:
        st.success("✅ Detected:")
        st.write(" → ".join(st.session_state.detected_words))
    else:
        st.caption("Waiting for signs...")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("✨ Generate", use_container_width=True, disabled=not st.session_state.detected_words):
            with st.spinner("🤖 Generating..."):
                sent = st.session_state.sentence_maker.make_sentence(st.session_state.detected_words)
            st.success(sent)
            
            # Audio Playback
            audio_bytes = speak_text(sent)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")
            
            st.session_state.history.append({
                'words': list(st.session_state.detected_words),
                'sentence': sent,
                'timestamp': time.strftime("%H:%M:%S"),
                'audio': audio_bytes
            })
            st.session_state.detected_words = []
            st.rerun()
    
    with col_btn2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.detected_words = []
            st.rerun()

    st.divider()
    st.subheader("📜 History")
    if st.session_state.history:
        for entry in reversed(st.session_state.history[-5:]):
            with st.expander(f"🕐 {entry['timestamp']}"):
                st.write("**Words:**", " → ".join(entry['words']))
                st.write("**Sentence:**", entry['sentence'])
                if entry.get('audio'):
                    st.audio(entry['audio'], format="audio/mp3")
    else:
        st.caption("No history yet")

st.divider()
st.caption("IsharaAI - Urdu Sign Language Recognition | © 2024")

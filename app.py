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

# WebRTC Configuration for Streamlit Cloud
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- CUSTOM IMPORTS ---
from model_utils import HybridSignRecognitionModel, HandsOnlyFeatureExtractor, normalize_frame_hands_only
from urdu_sentence_generator import UrduSentenceGenerator

# --- CONSTANTS ---
BASE_DIR = Path(__file__).parent
MODEL_PATH = str(BASE_DIR / "models_snapshot_handsonly" / "best_model.pth")
LABELS_PATH = str(BASE_DIR / "models_snapshot_handsonly" / "labels.json")
TARGET_SEQUENCE_LENGTH = 10
CONFIDENCE_THRESHOLD = 0.75

# Page Setup
st.set_page_config(page_title="IsharaAI", layout="wide")

# --- SESSION STATE INITIALIZATION (CRITICAL: Queue must be created ONCE) ---
# Create queue in session state so it persists across reruns
if "word_queue" not in st.session_state:
    st.session_state.word_queue = queue.Queue()

# Data storage
if "detected_words" not in st.session_state: 
    st.session_state.detected_words = []
if "history" not in st.session_state: 
    st.session_state.history = []
if "sentence_maker" not in st.session_state:
    st.session_state.sentence_maker = UrduSentenceGenerator()
if "processed_words" not in st.session_state:
    st.session_state.processed_words = set()

# Model-related
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
    """Converts text to speech and returns audio bytes."""
    try:
        tts = gTTS(text=text, lang=lang)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        return fp.getvalue()
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

# --- VIDEO PROCESSOR (Modified to accept queue reference) ---
class PyTorchProcessor(VideoProcessorBase):
    def __init__(self, model, labels, device, sensitivity, out_queue):
        self.model = model
        self.labels = labels
        self.device = device
        self.sensitivity = sensitivity
        self.out_queue = out_queue  # Direct queue reference
        self.extractor = HandsOnlyFeatureExtractor()
        self.buf = deque(maxlen=TARGET_SEQUENCE_LENGTH)
        self.votes = deque(maxlen=5)
        self.cooldown = 0
        self.last_label = None
        self.last_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Early return if model not loaded
        if self.model is None:
            h, w, _ = img.shape
            cv2.rectangle(img, (0, h-60), (w, h), (50, 50, 50), -1)
            cv2.putText(img, "MODEL NOT LOADED - Click 'Load Model' in sidebar", 
                       (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        features, results, hands_present = self.extractor.extract(img)
        
        # Active Check (Height Threshold)
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
            
            # Smoothing
            counts = Counter(p for p, c in self.votes)
            if counts:
                best_pred = counts.most_common(1)[0][0]
                avg_conf = np.mean([c for p, c in self.votes if p == best_pred])
                
                if avg_conf > CONFIDENCE_THRESHOLD and len(self.votes) >= 3 and self.cooldown == 0:
                    final_label = self.labels[best_pred]
                    
                    # Heuristic Correction (Aap vs Theek Hu)
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
                            
                            # Add to queue
                            try:
                                self.out_queue.put_nowait(final_label)
                                print(f"[OK] DETECTED: {final_label} (Queue size: {self.out_queue.qsize()})")
                            except Exception as e:
                                print(f"[!] Queue error: {e}")
                        
                        smooth_label = f"{final_label} ({avg_conf:.2f})"
        
        if self.cooldown > 0: 
            self.cooldown -= 1
        
        # Draw UI on Frame
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
                    
                    cfg_dict = checkpoint.get('config', {})
                    model = HybridSignRecognitionModel(
                        input_size=126, 
                        num_classes=len(labels),
                        hidden_size=int(cfg_dict.get('hidden_size', 128))
                    )
                    
                    try:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    except Exception:
                        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    
                    model.to(device).eval()
                    
                    st.session_state["model"] = model
                    st.session_state["labels"] = labels
                    st.session_state["device"] = device
                    st.session_state["model_loaded"] = True
                    
                    st.success(f"✅ Model Loaded!")
                    st.info(f"Device: {device}")
                    st.info(f"Classes: {len(labels)}")
                    st.balloons()
                    time.sleep(0.5)
                    st.rerun()
                    
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    st.error(f"❌ Error: {str(e)}")
                    st.code(error_details, language="text")
                    print(f"[!] MODEL LOAD ERROR: {error_details}")
                    st.session_state["model_loaded"] = False
            else:
                st.error("❌ Model files not found!")
    
    st.divider()
    
    if st.session_state.get("model_loaded", False):
        st.success("✅ Model Ready")
        if st.session_state.get("labels"):
            with st.expander("📋 Classes"):
                for i, label in enumerate(st.session_state["labels"]):
                    st.caption(f"{i+1}. {label}")
    else:
        st.warning("⚠️ Model not loaded")

# CRITICAL: Drain queue BEFORE rendering UI (just like working version)
q = st.session_state.word_queue
while not q.empty():
    try:
        detected_word = q.get_nowait()
        if detected_word and str(detected_word).strip():
            word_clean = str(detected_word).strip()
            word_id = id(detected_word)
            
            # Deduplication using processed_words set
            if word_id not in st.session_state.processed_words:
                if (not st.session_state.detected_words) or (word_clean != st.session_state.detected_words[-1]):
                    st.session_state.detected_words.append(word_clean)
                    st.session_state.processed_words.add(word_id)
                    print(f"[+] Added word: {word_clean}")
                    print(f"[*] Current list: {st.session_state.detected_words}")
    except queue.Empty:
        break

# Main Area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Detection")
    
    if st.session_state.get("model_loaded", False):
        # Capture references BEFORE creating processor
        model_ref = st.session_state.get("model")
        labels_ref = st.session_state.get("labels", [])
        device_ref = st.session_state.get("device", torch.device("cpu"))
        queue_ref = st.session_state.word_queue  # Direct queue reference
        
        def processor_factory():
            return PyTorchProcessor(model_ref, labels_ref, device_ref, sensitivity, queue_ref)
        
        webrtc_streamer(
            key="sign-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=processor_factory,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
    else:
        queue_ref = st.session_state.get("word_queue", queue.Queue())
        webrtc_streamer(
            key="sign-detection-loading",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: PyTorchProcessor(None, [], torch.device("cpu"), sensitivity, queue_ref),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
        st.info("👈 Load the model from sidebar")

with col2:
    st.subheader("📝 Detected Words")
    
    # Show queue status
    queue_size = st.session_state.word_queue.qsize()
    if queue_size > 0:
        st.info(f"🔄 Queue: {queue_size} pending")
    
    # Display Words
    if st.session_state.detected_words:
        st.success("✅ Detected:")
        st.write(" → ".join(st.session_state.detected_words))
        st.caption(f"Total: {len(st.session_state.detected_words)} words")
    else:
        st.caption("Waiting for signs...")
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("✨ Generate", use_container_width=True, disabled=len(st.session_state.detected_words) == 0):
            if st.session_state.detected_words:
                with st.spinner("🤖 Generating..."):
                    sent = st.session_state.sentence_maker.make_sentence(st.session_state.detected_words)
                st.success(sent)
                
                # Audio Playback Option
                audio_bytes = speak_text(sent)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
                
                st.session_state.history.append({
                    'words': st.session_state.detected_words.copy(),
                    'sentence': sent,
                    'timestamp': time.strftime("%H:%M:%S"),
                    'audio': audio_bytes
                })
                st.session_state.detected_words = []
                st.session_state.processed_words = set()
                st.rerun()
    
    with col_btn2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.detected_words = []
            st.session_state.processed_words = set()
            st.rerun()

    st.divider()
    st.subheader("📜 History")
    
    if st.session_state.history:
        for entry in reversed(st.session_state.history[-5:]):
            if isinstance(entry, dict):
                with st.expander(f"🕐 {entry.get('timestamp', 'N/A')}"):
                    st.write("**Words:**", " → ".join(entry.get('words', [])))
                    st.write("**Sentence:**", entry.get('sentence', ''))
                    if entry.get('audio'):
                        st.audio(entry['audio'], format="audio/mp3")
            else:
                st.write(f"- {entry}")
        
        if len(st.session_state.history) > 5:
            st.caption(f"Showing 5 of {len(st.session_state.history)} total")
    else:
        st.caption("No history yet")

st.divider()
st.caption("IsharaAI - Real-time Urdu Sign Language Recognition | © 2024")

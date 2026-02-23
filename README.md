# ✋ IsharaAI - Real-time Urdu Sign Language Recognition

![IsharaAI Logo](https://img.shields.io/badge/IsharaAI-Urdu_Sign_Language-green.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)

IsharaAI is a cutting-edge, real-time sign language recognition system tailored for **Urdu**. Leveraging Computer Vision, Deep Learning, and Generative AI, IsharaAI bridges the communication gap by translating hand signs into meaningful Urdu sentences and providing audio playback.

## ✨ Key Features

-   **📹 Real-time Detection**: High-speed hand landmark tracking using MediaPipe.
-   **🧠 Hybrid Deep Learning**: A sophisticated PyTorch-based model (LSTM + Attention) for accurate sign recognition.
-   **🤖 AI Sentence Generation**: Integrated Google Gemini API to transform a sequence of detected signs into grammatically correct Urdu sentences.
-   **🔊 Urdu Text-to-Speech (TTS)**: Built-in gTTS support to read out translations in a natural Urdu voice.
-   **📜 History Tracking**: Keeps a log of previous translations with audio playback support.
-   **⚡ Robust Performance**: Optimized for Windows environments with specific fixes for MediaPipe and Unicode encoding.

## 🛠 Tech Stack

-   **Frontend**: Streamlit
-   **Computer Vision**: MediaPipe, OpenCV
-   **Deep Learning**: PyTorch (LSTM + Attention Architecture)
-   **Generative AI**: Google Generative AI (Gemini Flash)
-   **Audio**: gTTS (Google Text-to-Speech)
-   **Language**: Python 3.12

## 🚀 Getting Started

### Prerequisites

-   **Python 3.12**: (Required for MediaPipe legacy solutions support on Windows)
-   **Google Gemini API Key**: (Included in the configuration for easy setup)

### Installation

1.  **Clone the Repository**:
    ```bash
    git clone [repository-url]
    cd "GEN AI hackathon"
    ```

2.  **Create a Virtual Environment**:
    ```powershell
    # Using the py launcher for version control
    py -3.12 -m venv .venv312
    ```

3.  **Activate and Install Dependencies**:
    ```powershell
    .\.venv312\Scripts\activate
    pip install -r requirements.txt
    pip install torch gTTS  # Additional requirements
    ```

### Running the App

```powershell
.\.venv312\Scripts\streamlit run app.py
```

## 📖 How to Use

1.  **Launch the App**: Open the Streamlit URL provided in your terminal (usually `http://localhost:8501`).
2.  **Load the Model**: In the sidebar, click the **"🔄 Load Model"** button. This initializes the weights and the detection engine.
3.  **Perform Signs**: Use the camera to perform Urdu sign language gestures. The app will detect signs in real-time.
4.  **Generate Sentence**: Once you've completed a phrase, click **"✨ Generate"**. The AI will clean the detection noise and provide a proper Urdu sentence.
5.  **Listen**: Click the audio player to hear the Urdu translation.
6.  **Review**: Check the **"History"** section to see and hear your previous interactions.

## 📁 Project Structure

```text
├── app.py                      # Main Streamlit UI and Video Processing
├── model_utils.py              # Neural Network Architecture and Feature Extraction
├── urdu_sentence_generator.py  # Gemini API Integration and Caching
├── models_snapshot_handsonly/  # Trained Model Weights (.pth) and Labels
├── requirements.txt            # Project Dependencies
└── .urdu_cache/               # Local cache for AI-generated sentences
```

## ⚠️ Important Notes (Windows Users)

-   **Python Version**: Ensure you are using Python **3.12**. Python 3.13 currently has compatibility issues with MediaPipe's `solutions` API.
-   **Encoding**: The project has been patched to avoid `UnicodeEncodeError` in Windows terminals by using ASCII status tags instead of emojis in logs.

---
© 2024 IsharaAI Team - Bridging Silence with Intelligence.

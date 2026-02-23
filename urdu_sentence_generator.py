import json
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

# --- CONFIGURATION ---
MY_API_KEY = "AIzaSyCbCbbAFSiKtV2S1PP3sBDg2q856RH44Vs"
# ---------------------

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class UrduSentenceGenerator:
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = ".urdu_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "sentence_cache.json"
        self.cache = self._load_cache()
        
        self.api_key = api_key or MY_API_KEY or os.getenv("GEMINI_API_KEY")
        self.model = None
        self.api_ready = False
        
        if GEMINI_AVAILABLE and self.api_key and "AIzaSyCbCbbAFSiKtV2S1PP3sBDg2q856RH44Vs" in self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel("gemini-flash-latest") 
                self.api_ready = True
                print("[OK] Gemini API initialized successfully")
            except Exception as e:
                print(f"[!] Gemini API initialization failed: {e}")
        
        # Offline Templates
        self.templates = {
            frozenset(["Aap", "Kese"]): "Aap kese hain?",
            frozenset(["Salam", "Aap"]): "Salam, aap kaise ho?",
            frozenset(["Salam", "Aap", "Kese"]): "Salam, aap kaise ho?",
            frozenset(["Theek Hu"]): "Main theek hoon."
        }

    def _load_cache(self) -> Dict[str, str]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: pass
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except: pass

    def _get_cache_key(self, words: List[str]) -> str:
        normalized = tuple(sorted([w.strip().lower() for w in words if w.strip()]))
        return hashlib.md5("|".join(normalized).encode()).hexdigest()

    def make_sentence(self, word_sequence: List[str]) -> str:
        if not word_sequence: return ""
        
        clean_words = [w.strip().title() for w in word_sequence if w.strip()]
        unique_words = list(dict.fromkeys(clean_words)) # Remove dupes preserve order
        
        # 1. Check Cache
        key = self._get_cache_key(unique_words)
        if key in self.cache:
            return self.cache[key]
        
        # 2. Try API
        if self.api_ready:
            try:
                words_str = ", ".join(unique_words)
                
                # New "Smart Filter" Prompt
                prompt = f"""
Act as an intelligent Urdu Sign Language interpreter. I will give you a sequence of words detected from a camera.
IMPORTANT: The sequence contains "noise" (wrong words detected while hands were moving between signs).

Input Words: {words_str}

Strict Rules:
1. **Identify the Core Message:** Look for the strongest logical phrase (e.g., "Aap" + "Kese" = "How are you?").
2. **Delete Transition Noise:** Aggressively REMOVE words that don't fit the main context. 
   - Example: If input is ["Aap", "Theek hu", "Kese"], remove "Theek hu" because it's likely noise between "Aap" and "Kese".
   - Example: If "Salam" appears in the middle of a sentence, ignore it or move it to the start.
3. **Handle Contradictions:** If you see "Theek hu" AND "Theek nhi hu" together, they are likely noise—ignore both unless the sentence clearly contrasts them.
4. **Output Format:** Valid Urdu Script ONLY (No Roman Urdu).

Now, generate the single best Urdu sentence based on the valid words only:
"""
                response = self.model.generate_content(prompt)
                sentence = response.text.strip()
                
                # Basic cleanup if the model adds quotes
                sentence = sentence.replace('"', '').replace("'", "")
                
                self.cache[key] = sentence
                self._save_cache()
                return sentence
            except Exception as e:
                print(f"API Error: {e}")
        
        # 3. Fallback
        return " ".join(unique_words)

    def clear_cache(self):
        self.cache = {}
        self._save_cache()
        
    def get_cache_stats(self):
        return {"total_cached": len(self.cache)}
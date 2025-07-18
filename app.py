# import asyncio
import streamlit as st
import os
import pdfplumber
import tempfile
import random
import io
import requests
import speech_recognition as sr
import time
import warnings
import murf as murf

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from dotenv import load_dotenv

# Use FAISS instead of DocArrayInMemorySearch for better compatibility
from langchain_community.vectorstores import FAISS

# Try to import murf, handle if not available
try:
    from murf import Murf
    MURF_AVAILABLE = True
except ImportError:
    MURF_AVAILABLE = False
    st.warning("Murf library not available. TTS features will be disabled.")

import base64

load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Satyukt Virtual Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API Key Loading ---
google_api_keys = [
    os.getenv("GOOGLE_API_KEY_1"),
    os.getenv("GOOGLE_API_KEY_2"),
    os.getenv("GOOGLE_API_KEY_3"),
    os.getenv("GOOGLE_API_KEY_4")
]
google_api_keys = [key for key in google_api_keys if key]

if not google_api_keys:
    st.error("❌ No valid GOOGLE_API_KEYs found. Please set at least one in your .env file (e.g., GOOGLE_API_KEY_1).")
    st.stop()

murf_api_key = os.getenv("MURF_API_KEY")
if not murf_api_key or not MURF_AVAILABLE:
    st.warning(
        "⚠️ MURF_API_KEY not found or Murf library not available. Text-to-Speech output will be disabled.")

# --- Custom CSS for Agriculture Theme ---
st.markdown(
    """
    <style>
    /* Main app styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }

    /* Welcome box styling */
    .welcome-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        text-align: center;
        color: white;
    }

    .logo-title {
        font-size: 2.5em;
        font-weight: 700;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }

    .welcome-subtitle {
        font-size: 1.2em;
        margin-bottom: 20px;
        opacity: 0.9;
    }

    /* Chat interface styling */
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        max-height: 500px;
        overflow-y: auto;
    }

    .user-message {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 15px;
        border-radius: 18px 18px 5px 18px;
        margin: 10px 0;
        margin-left: 50px;
        box-shadow: 0 3px 10px rgba(76, 175, 80, 0.3);
    }

    .bot-message {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #333;
        padding: 15px;
        border-radius: 18px 18px 18px 5px;
        margin: 10px 0;
        margin-right: 50px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #4CAF50;
    }

    .message-label {
        font-weight: 600;
        font-size: 0.9em;
        margin-bottom: 5px;
    }

    .user-label {
        color: #2E7D32;
        text-align: right;
        margin-right: 50px;
    }

    .bot-label {
        color: #1976D2;
        margin-left: 50px;
    }

    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #4CAF50;
        padding: 12px 20px;
        font-size: 16px;
    }

    .stTextInput > div > div > input:focus {
        border-color: #45a049;
        box-shadow: 0 0 10px rgba(76, 175, 80, 0.3);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #45a049 0%, #4CAF50 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
    }

    /* Selectbox styling */
    .stSelectbox > div > div > select {
        border-radius: 10px;
        border: 2px solid #4CAF50;
        padding: 8px 12px;
    }

    /* Spinner styling */
    .thinking-spinner {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
        background: #f8f9fa;
        border-radius: 15px;
        margin: 10px 0;
    }

    .spinner-text {
        margin-left: 10px;
        color: #4CAF50;
        font-weight: 600;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #4CAF50 0%, #2E7D32 100%);
    }

    .css-1d391kg .css-1v0mbdj {
        color: green;
    }

    /* Hide streamlit menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Language selector */
    .language-selector {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }

    .language-label {
        color: red;
        font-weight: 600;
        margin-bottom: 5px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "tts_audio_bytes" not in st.session_state:
    st.session_state.tts_audio_bytes = None
if "is_listening" not in st.session_state:
    st.session_state.is_listening = False
if "voice_input_text" not in st.session_state:
    st.session_state.voice_input_text = ""
if "tts_enabled" not in st.session_state:
    st.session_state.tts_enabled = (murf_api_key is not None and MURF_AVAILABLE)
if "input_method" not in st.session_state:
    st.session_state.input_method = "text"
if "initial_greeting_shown" not in st.session_state:
    st.session_state.initial_greeting_shown = False

# --- Initialize Speech Recognition ---
@st.cache_resource
def get_speech_recognizer():
    """Initializes and caches the speech recognizer."""
    try:
        recognizer = sr.Recognizer()
        # Try to get microphone, but don't fail if not available
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
            # st.success("Microphone ready!") # Removed to avoid persistent message
        except OSError:
            st.warning("No microphone detected. Voice input will be disabled.")
            return None
        return recognizer
    except Exception as e:
        st.error(f"❌ Error initializing speech recognition: {e}. Voice input may not work.")
        return None

# --- Language Mappings ---
sr_lang_codes = {
    "English": "en-US",
    "हिंदी": "hi-IN",
    "ಕನ್ನಡ": "kn-IN",
    "தமிழ்": "ta-IN",
    "తెలుగు": "te-IN",
    "বাংলা": "bn-IN",
    "मराठी": "mr-IN",
    "ગુજરાતી": "gu-IN",
    "ਪੰਜਾਬੀ": "pa-IN"
}

murf_voice_ids = {
    "English": "en-US-natalie",
    "हिंदी": "en-US-carter",
    "ಕನ್ನಡ": "en-US-miles",
    "தமிழ்": "en-US-amara",
    "తెలుగు": "en-US-riley",
    "বাংলা": "en-US-julia",
    "মराठी": "en-US-terrell",
    "ગુજરાતી": "en-US-charles",
    "ਪੰਜਾਬੀ": "en-US-alicia"
}

murf_multi_native_locales = {
    "English": None, # English typically doesn't need multi_native_locale
    "हिंदी": "hi-IN",
    "ಕನ್ನಡ": "kn-IN",
    "தமிழ்": "ta-IN",
    "తెలుగు": "te-IN",
    "বাংলা": "bn-IN",
    "মराठी": "mr-IN",
    "ગુજરાતી": "gu-IN",
    "ਪੰਜਾਬੀ": "pa-IN"
}

# --- Utility Functions for PDF, Gemini, and Voice ---

def extract_text_with_pdfplumber(pdf_path):
    """Extract text from PDF using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_page_text = page.extract_text()
                if extracted_page_text:
                    text += extracted_page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def initialize_vector_db(pdf_file, api_keys_list):
    """Initializes the vector store from PDF content, using FAISS."""
    if st.session_state.vector_store is None:
        loading_placeholder = st.empty()
        loading_placeholder.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: center; padding: 20px; background: #f8f9fa; border-radius: 15px; margin: 10px 0;">
                <div style="font-size: 24px; margin-right: 10px;">🤖</div>
                <div style="color: #4CAF50; font-weight: 600;">Initializing Satyukt Assistant... Please wait</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        pdf_path = None
        try:
            # Handle both BytesIO (from st.file_uploader) and file-like objects
            if isinstance(pdf_file, io.BytesIO):
                file_content_bytes = pdf_file.getvalue()
            elif hasattr(pdf_file, 'read'): # Generic file-like object
                file_content_bytes = pdf_file.read()
            else: # Assuming it's a path or similar if not BytesIO/file-like
                with open(pdf_file.path, 'rb') as f:
                    file_content_bytes = f.read()
            
            if not file_content_bytes:
                st.error("📄 Uploaded PDF file appears empty or corrupted.")
                return False

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_f:
                temp_f.write(file_content_bytes)
                pdf_path = temp_f.name

            text_data = extract_text_with_pdfplumber(pdf_path)

            if not text_data.strip():
                st.error("📄 PDF appears empty or unreadable after text extraction.")
                return False

            doc = Document(page_content=text_data)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
            chunks = text_splitter.split_documents([doc])

            if not chunks:
                st.error("🚨 No text chunks could be created from the PDF.")
                return False

            try:
                # Initialize embeddings
                st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=random.choice(api_keys_list)
                )
                
                # Test embeddings to catch API key issues early
                _ = st.session_state.embeddings.embed_query("test")
                
                # Use FAISS
                st.session_state.vector_store = FAISS.from_documents(
                    chunks, st.session_state.embeddings
                )

                return True

            except Exception as e:
                st.error(f"❌ Error with embeddings or vector store: {e}")
                return False

        except Exception as e:
            st.error(f"❌ Unexpected error during initialization: {str(e)}")
            return False
        finally:
            loading_placeholder.empty()
            if pdf_path and os.path.exists(pdf_path):
                os.unlink(pdf_path)
    return True

def generate_audio_bytes_murf(text, language="English"):
    """Generate audio bytes for text using Murf AI API."""
    if not murf_api_key or not MURF_AVAILABLE:
        # st.warning("Murf AI API key is not set or library not available. Cannot generate audio.") # Already warned
        return None
    if not text.strip():
        return None

    voice_id = murf_voice_ids.get(language, murf_voice_ids["English"])
    multi_native_locale = murf_multi_native_locales.get(language)

    try:
        client = Murf(api_key=murf_api_key)

        response = client.text_to_speech.generate(
            text=text,
            voice_id=voice_id,
            format="MP3",
            sample_rate=44100.0,
            encode_as_base_64=True,
            multi_native_locale=multi_native_locale if multi_native_locale else None
        )

        if response.encoded_audio:
            audio_content_bytes = base64.b64decode(response.encoded_audio)
            return audio_content_bytes
        else:
            st.error("Failed to receive audio from Murf AI.")
            return None

    except Exception as e:
        st.error(f"Error generating speech with Murf AI: {e}")
        return None

def listen_for_voice_input(language_code="en-US"):
    """Listen for voice input using speech recognition."""
    recognizer = get_speech_recognizer()
    if not recognizer:
        return "Speech recognition not available."

    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... Please speak clearly.")
            st.session_state.tts_audio_bytes = None # Clear previous audio
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=10)
            text = recognizer.recognize_google(audio, language=language_code)
            return text
    except sr.UnknownValueError:
        return "Could not understand audio. Please try speaking more clearly."
    except sr.WaitTimeoutError:
        return "No speech detected within the timeout. Please try again."
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"
    except Exception as e:
        return f"An unexpected error occurred during voice input: {e}"

contact_messages = {
    "English": "🤝 Let me connect you with our agricultural experts! Please contact support@satyukt.com or call 8970700045 | 7019992797 for specialized assistance.",
    "हिंदी": "🤝 मैं आपको हमारे कृषि विशेषज्ञों से जोड़ता हूं! विशेष सहायता के लिए कृपया support@satyukt.com पर संपर्क करें या 8970700045 | 7019992797 पर कॉल करें।",
    "ಕನ್ನಡ": "🤝 ನಮ್ಮ ಕೃಷಿ ತಜ್ಞರೊಂದಿಗೆ ನಿಮ್ಮನ್ನು ಸಂಪರ್ಕಿಸುತ್ತೇನೆ! ವಿಶೇಷ ಸಹಾಯಕ್ಕಾಗಿ support@satyukt.com ಗೆ ಸಂಪರ್ಕಿಸಿ ಅಥವಾ 8970700045 | 7019992797 ಗೆ ಕರೆ ಮಾಡಿ.",
    "தமிழ்": "🤝 எங்கள் விவசாய நிபுணர்களுடன் உங்களை இணைக்கிறேன்! சிறப்பு உதவிக்கு support@satyukt.com ஐ தொடர்பு கொள்ளவும் அல்லது 8970700045 | 7019992797 ஐ அழைக்கவும்.",
    "తెలుగు": "🤝 మా వ్యవసాయ నిపుణులతో మిమ్మల్ని కనెక్ట్ చేస్తాను! ప్రత్యేక సహాయం కోసం దయచేసి support@satyukt.com ని సంప్రదించండి లేదా 8970700045 | 7019992797 కు కాల్ చేయండి。",
    "বাংলা": "🤝 আমি আপনাকে আমাদের কৃষি বিশেষজ্ঞদের সাথে সংযুক্ত করব! বিশেষ সহায়তার জন্য অনুগ্রহ করে support@satyukt.com এ যোগাযোগ করুন অথবা 8970700045 | 7019992797 নম্বরে কল করুন।",
    "মराठी": "🤝 मी तुम्हाला आमच्या कृषी तज्ञांशी जोडतो! विशेष मदतीसाठी कृपया support@satyukt.com वर संपर्क साधा किंवा 8970700045 | 7019992797 वर कॉल करा।",
    "ગુજરાતી": "🤝 હું તમને અમારા કૃષિ નિષ્ણાત સાથે જોડું છું! વિશેષ સહાયતા માટે કૃપા કરીને support@satyukt.com નો સંપર્ક કરો અથવા 8970700045 | 7019992797 પર કૉલ કરો।",
    "ਪੰਜਾਬੀ": "🤝 ਮੈਂ ਤੁਹਾਨੂੰ ਸਾਡੇ ਖੇਤੀਬਾੜੀ ਮਾਹਿਰਾਂ ਨਾਲ ਜੋੜਦਾ ਹਾਂ! ਵਿਸ਼ੇਸ਼ ਸਹਾਇਤਾ ਲਈ ਕਿਰਪਾ ਕਰਕੇ support@satyukt.com 'ਤੇ ਸੰਪਰਕ ਕਰੋ ਜਾਂ 8970700045 | 7019992797 'ਤੇ ਕਾਲ ਕਰੋ।"
}

def is_out_of_context(answer, current_selected_lang):
    """Checks if the answer indicates an out-of-context response."""
    contact_message_template = contact_messages.get(current_selected_lang, contact_messages['English']).lower()

    if answer.strip().lower() == contact_message_template.strip().lower():
        return True

    keywords = [
        "i'm sorry", "i don't know", "not sure", "out of context",
        "invalid", "no mention", "cannot", "unable", "not available",
        "जानकारी उपलब्ध नहीं", "मुझे नहीं पता", "संदर्भ में नहीं",
        "माहिती उपलब्ध नाही", "नನಗೆ ಗೊತ್ತಿಲ್ಲ", "ನನಗೆ ಗೊತ್ತಿಲ್ಲ",
        "தகவல் இல்லை", "எனக்குத் தெரியாது",
        "సమాచారం అందుబాటులో లేదు", "నాకు తెలియదు",
        "তথ্য উপলব্ধ নয়", "আমি জানি না",
        "माहिती उपलब्ध नाही", "मला माहित नाही",
        "માહિતી ઉપલબ્ધ નથી", "મને ખબર નથી",
        "ਜਾਣਕਾਰੀ ਉਪਲਬਧ ਨਹੀਂ", "ਮੈਨੂੰ ਨਹੀਂ ਪਤਾ"
    ]
    # Check if the answer contains any of the keywords
    for keyword in keywords:
        if keyword in answer.lower():
            return True
            
    return False


# Initialize the Gemini LLM
@st.cache_resource
def get_llm(api_keys_list):
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest", 
            google_api_key=random.choice(api_keys_list),
            temperature=0.7
        )
    except Exception as e:
        st.error(f"Error initializing Gemini LLM: {e}")
        return None

llm = get_llm(google_api_keys)

if llm is None:
    st.error("Failed to initialize AI model. Please check your API keys.")
    st.stop()

# --- Sidebar UI ---
with st.sidebar:
    st.markdown('<div class="language-selector">', unsafe_allow_html=True)
    st.markdown('<div class="language-label">🌍 Select Language / भाषा चुनें</div>', unsafe_allow_html=True)

    languages = list(sr_lang_codes.keys())
    selected_lang = st.selectbox("Select Language", languages, key="language_selector")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Input & Output Settings")
    st.session_state.input_method = st.radio(
        "Choose your input method:",
        ('Text', 'Voice'),
        index=0 if st.session_state.input_method == 'text' else 1,
        key="input_method_radio",
        horizontal=True
    ).lower()

    if murf_api_key and MURF_AVAILABLE:
        st.session_state.tts_enabled = st.checkbox(
            "Enable Text-to-Speech Output",
            value=st.session_state.tts_enabled,
            key="tts_toggle"
        )
    else:
        st.session_state.tts_enabled = False
        st.info("💡 Enable TTS by providing a MURF_API_KEY and installing murf library.")

    st.markdown("---")
    st.markdown("### 🌾 About Satyukt 🌾")
    st.markdown("**Virtual Assistant** powered by AI and Satellite Intelligence")
    st.markdown("**Services:**")
    st.markdown("- 🛰️ Crop Monitoring")
    st.markdown("- 📊 Risk Analytics")
    st.markdown("- 💰 Insurance Claims")
    st.markdown("- 🏦 Agricultural Credit")

    st.markdown("---")
    st.markdown("### 📞 Contact")
    st.markdown("📧 support@satyukt.com")
    st.markdown("📱 8970700045 | 7019992797")

# --- Main Welcome Container ---
st.markdown(
    """
    <div class="welcome-container">
        <div class="logo-title">🌾 Satyukt Virtual Assistant🌾</div>
        <div class="welcome-subtitle">Empowering Agriculture with Satellite Intelligence & AI Technology</div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Feature Cards Section ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div style="background: rgba(76, 175, 80, 0.1); padding: 20px; border-radius: 10px; text-align: center; margin: 10px 0;">
            <div style="font-size: 2em; margin-bottom: 10px;">🛰️</div>
            <div style="font-weight: 600;">Satellite Monitoring</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div style="background: rgba(33, 150, 243, 0.1); padding: 20px; border-radius: 10px; text-align: center; margin: 10px 0;">
            <div style="font-size: 2em; margin-bottom: 10px;">📊</div>
            <div style="font-weight: 600;">Risk Analysis</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div style="background: rgba(255, 152, 0, 0.1); padding: 20px; border-radius: 10px; text-align: center; margin: 10px 0;">
            <div style="font-size: 2em; margin-bottom: 10px;">🤖</div>
            <div style="font-weight: 600;">AI Assistant</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        """
        <div style="background: rgba(139, 195, 74, 0.1); padding: 20px; border-radius: 10px; text-align: center; margin: 10px 0;">
            <div style="font-size: 2em; margin-bottom: 10px;">🏦</div>
            <div style="font-weight: 600;">Agricultural Credit</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---") # Fix: Wrapped in st.markdown()
st.markdown("### **Chat Interface**") # Fix: Wrapped in st.markdown()
st.markdown("---") # Fix: Wrapped in st.markdown()

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat history
for i, message in enumerate(st.session_state.chat_history):
    if message["role"] == "user":
        st.markdown(f'<div class="message-label user-label">You</div><div class="user-message">{message["parts"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="message-label bot-label">Satyukt Assistant</div><div class="bot-message">{message["parts"]}</div>', unsafe_allow_html=True)
        if message.get("audio_bytes"):
            st.audio(message["audio_bytes"], format="audio/mp3", start_time=0)

st.markdown('</div>', unsafe_allow_html=True) # Close chat-container div

# Initial greeting for first-time users
if not st.session_state.initial_greeting_shown:
    initial_bot_message_en = "Hello! I am your Satyukt Virtual Assistant. How can I assist you with your agricultural queries today?"
    st.session_state.chat_history.append({"role": "model", "parts": initial_bot_message_en, "audio_bytes": None})
    st.session_state.initial_greeting_shown = True
    st.rerun() # Rerun to display the initial greeting

# --- File Uploader ---
st.markdown("---")
uploaded_file = st.file_uploader("Upload a PDF document for context (optional)", type="pdf")

if uploaded_file and st.session_state.vector_store is None:
    if initialize_vector_db(uploaded_file, google_api_keys):
        st.success("✅ PDF processed successfully and knowledge base updated!")
    else:
        st.error("❌ Failed to process PDF. Please try again.")

# --- Chat Input ---
st.markdown("---")
if st.session_state.input_method == "text":
    user_query = st.text_input("Ask a question about agriculture or Satyukt services:", key="user_text_input")
elif st.session_state.input_method == "voice":
    voice_placeholder = st.empty()
    if st.button("Start Voice Input 🎤", key="voice_start_button"):
        st.session_state.is_listening = True
        st.session_state.voice_input_text = "" # Clear previous voice input
        st.rerun() # Rerun to show "Listening..." immediately

    if st.session_state.is_listening:
        with voice_placeholder.container():
            st.info("Listening... Speak now.")
            recognized_text = listen_for_voice_input(sr_lang_codes[selected_lang])
            if recognized_text:
                st.session_state.voice_input_text = recognized_text
                st.success(f"Recognized: {recognized_text}")
            else:
                st.warning("Voice input failed or no speech detected.")
            st.session_state.is_listening = False # Stop listening after attempt
            st.rerun() # Rerun to update input field and hide "Listening..."

    user_query = st.text_input("Your Voice Input (edit if needed):", value=st.session_state.voice_input_text, key="voice_text_display")
    
    # Clear voice input after use, but allow it to be displayed
    if st.button("Clear Voice Input", key="clear_voice_button"):
        st.session_state.voice_input_text = ""
        user_query = ""
        st.rerun()

# Process user query
if (st.session_state.input_method == "text" and user_query) or \
   (st.session_state.input_method == "voice" and st.session_state.voice_input_text and st.button("Submit Query", key="submit_button_for_voice")): # Changed button key
    
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "parts": user_query})
    
    thinking_placeholder = st.markdown(
        """
        <div class="thinking-spinner">
            <div class="spinner-border text-success" role="status">
                <span class="sr-only"></span>
            </div>
            <div class="spinner-text">Satyukt Assistant is thinking...</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    try:
        # Define the prompt template
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are the Satyukt Virtual Assistant, an AI expert in agriculture and Satyukt's services, powered by satellite intelligence. Provide concise and accurate answers based on the provided context and your knowledge. If the question is outside the context or your knowledge, politely state that you cannot answer and suggest contacting Satyukt support (support@satyukt.com, 8970700045 | 7019992797). Ensure your responses are helpful and informative for farmers and agricultural businesses. Always respond in the language the user is speaking in, or the selected language."),
                ("human", "Answer the following question based on the provided context:\n\n{context}\n\nQuestion: {input}"),
            ]
        )

        document_chain = create_stuff_documents_chain(llm, prompt_template)

        # Retrieve relevant documents if a vector store exists
        if st.session_state.vector_store:
            retriever = st.session_state.vector_store.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            response = retrieval_chain.invoke({"input": user_query})
            bot_response = response["answer"]
        else:
            # If no vector store, use a simpler chain or direct LLM call
            # For direct LLM call without context
            response = llm.invoke(f"Answer the following question about agriculture or Satyukt services: {user_query}. Respond in {selected_lang}. If you cannot answer, suggest contacting Satyukt support.")
            bot_response = response.content

        # Handle out-of-context responses
        if is_out_of_context(bot_response, selected_lang):
            final_bot_response = contact_messages.get(selected_lang, contact_messages["English"])
        else:
            final_bot_response = bot_response

        audio_bytes = None
        if st.session_state.tts_enabled and final_bot_response:
            audio_bytes = generate_audio_bytes_murf(final_bot_response, selected_lang)
            st.session_state.tts_audio_bytes = audio_bytes # Store for immediate playback

        st.session_state.chat_history.append({"role": "model", "parts": final_bot_response, "audio_bytes": audio_bytes})

    except Exception as e:
        error_message = f"An error occurred: {e}. Please try again or contact support if the issue persists."
        st.session_state.chat_history.append({"role": "model", "parts": error_message, "audio_bytes": None})
    finally:
        thinking_placeholder.empty()
        st.session_state.voice_input_text = "" # Clear voice input after processing
        st.rerun() # Rerun to update the chat display and clear input fields

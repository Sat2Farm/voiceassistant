import streamlit as st
import os
import pdfplumber
import tempfile
import random
import io
import requests
import speech_recognition as sr
import time

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.documents import Document
from dotenv import load_dotenv

from murf import Murf
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
if not murf_api_key:
    st.warning(
        "⚠️ MURF_API_KEY not found. Text-to-Speech output will be disabled. Please set it in your .env file to enable TTS.")

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
    st.session_state.tts_enabled = (murf_api_key is not None)
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
        with sr.Microphone() as source:
            st.info("Adjusting for ambient noise... Please wait a moment.")
            recognizer.adjust_for_ambient_noise(source, duration=1)
        st.success("Microphone ready!")
        return recognizer
    except Exception as e:
        st.error(f"❌ Error initializing speech recognition: {e}. Voice input may not work. "
                 f"Ensure you have PyAudio installed and your microphone is set up correctly.")
        return None


# --- Language Mappings ---
sr_lang_codes = {
    "English": "en-US",
    "हिंदी": "hi-IN",
    "ಕನ್ನಡ": "kn-IN",
    "தமிழ்": "ta-IN",
    "తెలుగు": "te-IN",
    "বাংলা": "bn-IN",
    "মराठी": "mr-IN",
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
    "English": None,
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


def initialize_vector_db(pdf_file, api_keys_list):  # Changed parameter name
    """Initializes the vector store from PDF content, caching it."""
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
            file_content_bytes = pdf_file.read().getvalue()
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
                st.error(
                    "🚨 No text chunks could be created from the PDF. This might be due to very short or no usable text in the PDF after splitting.")
                return False

            try:
                st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=random.choice(api_keys_list)  # Use the passed list
                )
                _ = st.session_state.embeddings.embed_query("hello world")
            except Exception as e:
                st.error(
                    f"❌ Error with Google Generative AI Embeddings. Check your GOOGLE_API_KEYs and network connection: {e}")
                st.session_state.embeddings = None
                return False

            st.session_state.vector_store = DocArrayInMemorySearch.from_documents(
                chunks, st.session_state.embeddings
            )

            if st.session_state.vector_store is None:
                st.error(
                    "❌ DocArrayInMemorySearch could not be initialized from documents. This might be a dependency conflict (e.g., pydantic, docarray versions).")
                return False

            return True

        except requests.exceptions.RequestException as e:
            st.error(
                f"❌ Network error or API issue during embedding initialization: {e}. Check your internet connection and API keys.")
            return False
        except ImportError as e:
            st.error(
                f"❌ Missing library for vector store or embeddings: {e}. Please ensure all required packages are installed (`pip install docarray pydantic==1.10.9` if issues persist).")
            return False
        except Exception as e:
            st.error(
                f"❌ An unexpected error occurred during assistant initialization: {str(e)}. Please check your PDF and API keys.")
            return False
        finally:
            loading_placeholder.empty()
            if pdf_path and os.path.exists(pdf_path):
                os.unlink(pdf_path)
    return True


def generate_audio_bytes_murf(text, language="English"):
    """Generate audio bytes for text using Murf AI API."""
    if not murf_api_key:
        st.warning("Murf AI API key is not set. Cannot generate audio.")
        return None
    if not text.strip():
        return None

    voice_id = murf_voice_ids.get(language, murf_voice_ids["English"])
    multi_native_locale = murf_multi_native_locales.get(language)

    try:
        client = Murf(api_key=murf_api_key)

        print(
            f"Generating Murf AI audio for text: '{text[:50]}...' with voice_id: {voice_id}, multi_native_locale: {multi_native_locale}")
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
            print("Murf AI response did not contain encoded audio.")
            if response.warning:
                print(f"Warning from Murf AI: {response.warning}")
            st.error("Failed to receive audio from Murf AI.")
            return None

    except Exception as e:
        st.error(f"Error generating speech with Murf AI: {e}")
        st.warning(
            "Please check your Murf AI API key, internet connection, and character limit on your Murf AI plan.")
        return None


def listen_for_voice_input(language_code="en-US"):
    """Listen for voice input using speech recognition."""
    recognizer = get_speech_recognizer()
    if not recognizer:
        return "Speech recognition not available due to initialization error."

    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... Please speak clearly.")
            st.session_state.tts_audio_bytes = None
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=10)
            text = recognizer.recognize_google(audio, language=language_code)
            return text
    except sr.UnknownValueError:
        return "Could not understand audio. Please try speaking more clearly."
    except sr.WaitTimeoutError:
        return "No speech detected within the timeout. Please try again."
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}. Check your internet connection."
    except Exception as e:
        return f"An unexpected error occurred during voice input: {e}"


contact_messages = {
    "English": "🤝 Let me connect you with our agricultural experts! Please contact support@satyukt.com or call 8970700045 | 7019992797 for specialized assistance.",
    "हिंदी": "🤝 मैं आपको हमारे कृषि विशेषज्ञों से जोड़ता हूँ! विशेष सहायता के लिए कृपया support@satyukt.com पर संपर्क करें या 8970700045 | 7019992797 पर कॉल करें।",
    "ಕನ್ನಡ": "🤝 ನಮ್ಮ ಕೃಷಿ ತಜ್ಞರೊಂದಿಗೆ ನಿಮ್ಮನ್ನು ಸಂಪರ್ಕಿಸುತ್ತೇನೆ! ವಿಶೇಷ ಸಹಾಯಕ್ಕಾಗಿ support@satyukt.com ಗೆ ಸಂಪರ್ಕಿಸಿ ಅಥವಾ 8970700045 | 7019992797 ಗೆ ಕರೆ ಮಾಡಿ.",
    "தமிழ்": "🤝 எங்கள் விவசாய நிபுணர்களுடன் உங்களை இணைக்கிறேன்! சிறப்பு உதவிக்கு support@satyukt.com ஐ தொடர்பு கொள்ளவும் அல்லது 8970700045 | 7019992797 ஐ அழைக்கவும்.",
    "తెలుగు": "🤝 మా వ్యవసాయ నిపుణులతో మిమ్మల్ని కనెక్ట్ చేస్తాను! ప్రత్యేక సహాయం కోసం దయచేసి support@satyukt.com ని సంప్రదించండి లేదా 8970700045 | 7019992797 కు కాల్ చేయండి。",
    "বাংলা": "🤝 আমি আপনাকে আমাদের কৃষি বিশ৻জ্ঞদের সাথে সংযুক্ত করব! বিশেষ সহায়তার জন্য অনুগ্রহ করে support@satyukt.com এ যোগাযোগ করুন অথবা 8970700045 | 7019992797 নম্বরে কল করুন।",
    "মराठी": "🤝 मी तुम्हाला आमच्या कृषी तज्ञांशी जोडतो! विशेष मदतीसाठी कृपया support@satyukt.com वर संपर्क साधा किंवा 8970700045 | 7019992797 वर कॉल करा。",
    "ગુજરાતી": "🤝 હું તમને અમારા કૃષિ નિષ્ણાત સાથે જોડું છું! વિશેષ સહાયતા માટે કૃપા કરીને support@satyukt.com નો સંપર્ક કરો અથવા 8970700045 | 7019992797 પર કૉલ કરો。",
    "ਪੰਜਾਬੀ": "🤝 ਮੈਂ ਤੁਹਾਨੂੰ ਸਾਡੇ ਖੇਤੀਬਾੜੀ ਮਾਹਿਰਾਂ ਨਾਲ ਜੋੜਦਾ ਹਾਂ! ਵਿਸ਼ੇਸ਼ ਸਹਾਇਤਾ ਲਈ ਕਿਰਪਾ ਕਰਕੇ support@satyukt.com 'ਤੇ ਸੰਪਰਕ ਕਰੋ ਜਾਂ 8970700045 | 7019992797 'ਤੇ ਕਾਲ ਕਰੋ।"
}


def is_out_of_context(answer, current_selected_lang):
    """Checks if the answer indicates an out-of-context response or a predefined contact message."""
    contact_message_template = contact_messages.get(current_selected_lang, contact_messages['English']).lower()

    if answer.strip().lower() == contact_message_template.strip().lower():
        return True

    keywords = [
        "i'm sorry", "i don't know", "not sure", "out of context",
        "invalid", "no mention", "cannot", "unable", "not available",
        "जानकारी उपलब्ध नहीं", "मुझे नहीं पता", "संदर्भ में नहीं",
        "ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ", "ನನಗೆ ಗೊತ್ತಿಲ್ಲ",
        "தகவல் இல்லை", "எனக்குத் தெரியாது",
        "సమాచారం అందుబాటులో లేదు", "నాకు తెలియదు",
        "তথ্য উপলব্ধ নয়", "আমি জানি না",
        "माहिती उपलब्ध नाही", "मला माहित नाही",
        "માહિતી ઉપલબ્ધ નથી", "મને ખબર નથી",
        "ਜਾਣਕਾਰੀ ਉਪਲਬਧ ਨਹੀਂ", "ਮੈਨੂੰ ਨਹੀਂ ਪਤਾ"
    ]
    return any(k in answer.lower() for k in keywords)


# Initialize the Gemini LLM (cached resource for efficiency)
@st.cache_resource
def get_llm(api_keys_list):  # Changed parameter name
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=random.choice(api_keys_list))


llm = get_llm(google_api_keys)  # Corrected: Pass google_api_keys here

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

    if murf_api_key:
        st.session_state.tts_enabled = st.checkbox(
            "Enable Text-to-Speech Output",
            value=st.session_state.tts_enabled,
            key="tts_toggle"
        )
    else:
        st.session_state.tts_enabled = False
        st.info("💡 Enable TTS by providing a MURF_API_KEY in your .env file.")

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
            <div style="font-size: 2em; margin-bottom: 10px;">🌾</div>
            <div style="font-weight: 600;">Crop Insights</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Auto-load PDF for RAG context ---
default_pdf_path = "SatyuktQueries.pdf"


class DummyFile:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self._buffer = None

    def read(self):
        if self._buffer is None:
            with open(self.path, "rb") as f:
                self._buffer = io.BytesIO(f.read())
        self._buffer.seek(0)
        return self._buffer

    @property
    def size(self):
        return os.path.getsize(self.path)

    @property
    def type(self):
        return "application/pdf"


if os.path.exists(default_pdf_path):
    pdf_input_from_user = DummyFile(default_pdf_path)

    if initialize_vector_db(pdf_input_from_user, google_api_keys):
        if not st.session_state.initial_greeting_shown:
            st.success(
                "✅ Hi there! 👋 Satyukt Virtual Assistant is ready to assist you! Ask me anything about agriculture, farming, or our services.")
            st.session_state.initial_greeting_shown = True
else:
    st.error(
        f"❌ PDF file '{default_pdf_path}' not found in the project directory. Please ensure it's in the same directory as your Streamlit app."
    )
    st.session_state.vector_store = None

# --- Chat Interface ---
if st.session_state.vector_store is not None:
    st.markdown("### 💬 Chat with Satyukt Virtual Assistant")

    chat_placeholder = st.container()
    with chat_placeholder:
        for i, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                st.markdown(f'<div class="message-label user-label">🧑‍🌾 You</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="message-label bot-label">🤖 Satyukt</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="bot-message">{msg["content"]}</div>', unsafe_allow_html=True)

        # THE MODIFIED ST.AUDIO CALL (WITHOUT 'KEY')
        if st.session_state.tts_audio_bytes:
            st.audio(st.session_state.tts_audio_bytes, format="audio/mp3", autoplay=True, loop=False)
            st.session_state.tts_audio_bytes = None

        st.markdown(
            """
            <script>
                var chatContainer = document.querySelector('.chat-container');
                if (chatContainer) {
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            </script>
            """,
            unsafe_allow_html=True
        )

    # --- Input Section (Text or Voice) ---
    st.markdown("### Ask your question:")

    user_prompt_text_input = ""
    if st.session_state.input_method == "text":
        user_prompt_text_input = st.text_input(
            "Type your question here...",
            placeholder=f"Ask me anything in {selected_lang}... 🌾",
            key="text_input_main",
            label_visibility="collapsed",
            value=st.session_state.get("last_text_input", "")
        )
    elif st.session_state.input_method == "voice":
        st.text_area(
            "Recognized Voice Input:",
            value=st.session_state.voice_input_text,
            height=68,
            disabled=True,
            key="voice_input_display"
        )
        if st.session_state.is_listening:
            st.info("🎤 Listening... Speak clearly into your microphone.")
        else:
            st.info("Click 'Start Listening' to speak your question.")

    col_input_btn, col_send_btn = st.columns([0.4, 0.15])

    with col_input_btn:
        if st.session_state.input_method == "voice":
            if st.button("Start Listening" if not st.session_state.is_listening else "Stop Listening",
                         key="voice_toggle_btn"):
                if not st.session_state.is_listening:
                    st.session_state.is_listening = True
                    st.session_state.voice_input_text = ""

                    language_code_for_sr = sr_lang_codes.get(selected_lang, "en-US")
                    with st.spinner(f"Listening for {selected_lang} voice input..."):
                        recognized_text = listen_for_voice_input(language_code_for_sr)

                    st.session_state.is_listening = False

                    if recognized_text and not (
                            "Could not understand audio" in recognized_text or
                            "No speech detected" in recognized_text or
                            "Could not request results" in recognized_text or
                            "An unexpected error occurred" in recognized_text
                    ):
                        st.session_state.voice_input_text = recognized_text
                    else:
                        st.warning(recognized_text)
                        st.session_state.voice_input_text = ""
                    st.rerun()
                else:
                    st.session_state.is_listening = False
                    st.warning("Listening stopped manually.")
                    st.rerun()

    with col_send_btn:
        send_button_clicked = st.button("Send 🚀", key="send_btn_final")

    final_user_query = ""
    if st.session_state.input_method == "text":
        final_user_query = user_prompt_text_input.strip()
        if send_button_clicked or (user_prompt_text_input and st.session_state.get(
                "last_text_input") != user_prompt_text_input and st.session_state.get("text_input_main_touched",
                                                                                      False)):
            st.session_state.last_text_input = user_prompt_text_input
            if final_user_query:
                process_query_flag = True
            else:
                st.warning("⚠️ Please enter a question before sending.")
                process_query_flag = False
        else:
            process_query_flag = False

    elif st.session_state.input_method == "voice":
        final_user_query = st.session_state.voice_input_text.strip()
        if send_button_clicked and final_user_query:
            process_query_flag = True
        elif send_button_clicked and not final_user_query:
            st.warning("⚠️ Please record your voice question first.")
            process_query_flag = False
        else:
            process_query_flag = False

    if process_query_flag and final_user_query:
        st.session_state.chat_history.append({"role": "user", "content": final_user_query})

        if st.session_state.input_method == "voice":
            st.session_state.voice_input_text = ""

        with st.spinner("🤖 Satyukt is thinking..."):
            retriever = st.session_state.vector_store.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever,
                                                     create_stuff_documents_chain(llm, ChatPromptTemplate.from_template("""
                You are a helpful AI assistant specialized in agriculture and Satyukt's services.
                Answer the user's questions based only on the provided context.
                If the answer is not in the context, politely state that you cannot provide information on that specific topic and suggest they contact support@satyukt.com or call 8970700045 | 7019992797 for specialized assistance.
                Do NOT make up answers.
                Keep your answers concise and directly to the point.
                If the user asks in a language other than English, respond in that language if possible, otherwise use English.

                Context:
                {context}

                Question: {input}

                Chat History:
                {chat_history}
                """)))

            chat_history_for_prompt = "\n".join([
                f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history
            ])

            response = retrieval_chain.invoke({
                "input": final_user_query,
                "chat_history": chat_history_for_prompt
            })

            ai_response_content = response["answer"]

            if is_out_of_context(ai_response_content, selected_lang):
                ai_response_content = contact_messages.get(selected_lang, contact_messages["English"])

            st.session_state.chat_history.append({"role": "assistant", "content": ai_response_content})

            if st.session_state.tts_enabled and murf_api_key:
                st.session_state.tts_audio_bytes = generate_audio_bytes_murf(ai_response_content, selected_lang)
            else:
                st.session_state.tts_audio_bytes = None

        st.session_state.last_text_input = ""
        st.rerun()


elif st.session_state.vector_store is None:
    st.info(
        "⬆️ Please ensure the 'SatyuktQueries.pdf' file is in the same directory as this script to enable the Virtual Assistant.")



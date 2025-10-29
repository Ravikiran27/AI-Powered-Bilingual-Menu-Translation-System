"""
Bilingual Hotel Menu Display System - Streamlit UI
===================================================

This application allows users to:
1. Upload a menu image
2. Extract text using OCR
3. Translate to Hindi and Kannada using trained models
4. Display results in a table
5. Download translations as CSV
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import torch
from transformers import MarianMTModel, MarianTokenizer
import easyocr
import io
import os

# Page configuration
st.set_page_config(
    page_title="Bilingual Menu Translator",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #34495e;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #3498db;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .result-table {
        margin-top: 2rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'translations' not in st.session_state:
    st.session_state.translations = None
if 'ocr_text' not in st.session_state:
    st.session_state.ocr_text = None

@st.cache_resource
def load_models():
    """Load the trained translation models"""
    try:
        # Check if models exist
        if not os.path.exists('saved_model/en_hi') or not os.path.exists('saved_model/en_kn'):
            st.error("⚠️ Models not found! Please run the Jupyter notebook first to train the models.")
            return None, None, None, None
        
        # Load Hindi model
        tokenizer_hi = MarianTokenizer.from_pretrained('saved_model/en_hi')
        model_hi = MarianMTModel.from_pretrained('saved_model/en_hi')
        
        # Load Kannada model
        tokenizer_kn = MarianTokenizer.from_pretrained('saved_model/en_kn')
        model_kn = MarianMTModel.from_pretrained('saved_model/en_kn')
        
        # Set to evaluation mode
        model_hi.eval()
        model_kn.eval()
        
        return tokenizer_hi, model_hi, tokenizer_kn, model_kn
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

@st.cache_resource
def load_ocr_reader():
    """Initialize EasyOCR reader"""
    try:
        reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        return reader
    except Exception as e:
        st.error(f"Error loading OCR: {str(e)}")
        return None

def translate_text(text, model, tokenizer, max_length=128):
    """Translate text using the model"""
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_length)
        
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated
    except Exception as e:
        return f"Error: {str(e)}"

def extract_text_from_image(image, reader):
    """Extract text from image using OCR"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Perform OCR
        results = reader.readtext(img_array)
        
        # Extract text
        extracted_text = []
        for (bbox, text, prob) in results:
            if prob > 0.5:  # Only include high-confidence detections
                extracted_text.append(text.strip())
        
        return extracted_text
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return []

def main():
    # Header
    st.markdown('<div class="main-header">🍽️ Bilingual Hotel Menu Display System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Translate your menu from English to Hindi & Kannada using AI</div>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading translation models..."):
        tokenizer_hi, model_hi, tokenizer_kn, model_kn = load_models()
    
    if model_hi is None or model_kn is None:
        st.stop()
    
    # Load OCR
    with st.spinner("Loading OCR engine..."):
        ocr_reader = load_ocr_reader()
    
    if ocr_reader is None:
        st.warning("⚠️ OCR not available. You can still enter text manually.")
    
    # Sidebar
    st.sidebar.title("📋 Options")
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Upload Image (OCR)", "Enter Text Manually"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(
        "This application uses fine-tuned MarianMT models to translate "
        "English menu items to Hindi and Kannada. Perfect for hotels and "
        "restaurants serving multilingual customers!"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Info")
    st.sidebar.markdown("""
    - **English → Hindi**: MarianMT
    - **English → Kannada**: MarianMT
    - **OCR Engine**: EasyOCR
    """)
    
    # Main content
    if input_method == "Upload Image (OCR)":
        st.markdown("### 📸 Upload Menu Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file (JPG, PNG, JPEG)",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of your menu for best results"
        )
        
        if uploaded_file is not None:
            # Display image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
            
            with col2:
                st.markdown("#### OCR Results")
                
                if st.button("🔍 Extract Text"):
                    if ocr_reader is None:
                        st.error("OCR engine not available!")
                    else:
                        with st.spinner("Extracting text from image..."):
                            extracted_items = extract_text_from_image(image, ocr_reader)
                            
                            if extracted_items:
                                st.session_state.ocr_text = extracted_items
                                st.success(f"✅ Found {len(extracted_items)} text items!")
                                
                                # Display extracted text
                                st.markdown("**Extracted Items:**")
                                for i, item in enumerate(extracted_items, 1):
                                    st.write(f"{i}. {item}")
                            else:
                                st.error("❌ No text found in image. Try a clearer image or enter text manually.")
            
            # Translate button
            if st.session_state.ocr_text:
                st.markdown("---")
                if st.button("🌐 Translate All Items"):
                    with st.spinner("Translating..."):
                        translations = []
                        
                        for item in st.session_state.ocr_text:
                            hindi = translate_text(item, model_hi, tokenizer_hi)
                            kannada = translate_text(item, model_kn, tokenizer_kn)
                            
                            translations.append({
                                'English': item,
                                'Hindi (हिंदी)': hindi,
                                'Kannada (ಕನ್ನಡ)': kannada
                            })
                        
                        st.session_state.translations = pd.DataFrame(translations)
    
    else:  # Manual text entry
        st.markdown("### ✍️ Enter Menu Items")
        
        manual_input = st.text_area(
            "Enter menu items (one per line):",
            height=200,
            placeholder="Butter Chicken\nPaneer Tikka\nMasala Dosa\n...",
            help="Enter each menu item on a new line"
        )
        
        if st.button("🌐 Translate"):
            if manual_input.strip():
                items = [item.strip() for item in manual_input.split('\n') if item.strip()]
                
                with st.spinner("Translating..."):
                    translations = []
                    
                    for item in items:
                        hindi = translate_text(item, model_hi, tokenizer_hi)
                        kannada = translate_text(item, model_kn, tokenizer_kn)
                        
                        translations.append({
                            'English': item,
                            'Hindi (हिंदी)': hindi,
                            'Kannada (ಕನ್ನಡ)': kannada
                        })
                    
                    st.session_state.translations = pd.DataFrame(translations)
            else:
                st.warning("⚠️ Please enter at least one menu item!")
    
    # Display results
    if st.session_state.translations is not None:
        st.markdown("---")
        st.markdown("### 📊 Translation Results")
        
        # Display table
        st.dataframe(
            st.session_state.translations,
            use_container_width=True,
            hide_index=True
        )
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download as CSV
            csv = st.session_state.translations.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="menu_translations.csv",
                mime="text/csv",
            )
        
        with col2:
            # Copy to clipboard (instructions)
            if st.button("📋 Copy Instructions"):
                st.info("Select the table above and use Ctrl+C (Windows) or Cmd+C (Mac) to copy!")
        
        with col3:
            # Clear results
            if st.button("🗑️ Clear Results"):
                st.session_state.translations = None
                st.session_state.ocr_text = None
                st.rerun()
        
        # Statistics
        st.markdown("---")
        st.markdown("### 📈 Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Items", len(st.session_state.translations))
        
        with col2:
            avg_en_len = st.session_state.translations['English'].str.len().mean()
            st.metric("Avg. English Length", f"{avg_en_len:.1f} chars")
        
        with col3:
            avg_hi_len = st.session_state.translations['Hindi (हिंदी)'].str.len().mean()
            st.metric("Avg. Hindi Length", f"{avg_hi_len:.1f} chars")

if __name__ == "__main__":
    main()

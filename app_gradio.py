"""
Bilingual Hotel Menu Display System - Gradio UI
================================================

Alternative UI using Gradio for menu translation
"""

import gradio as gr
import pandas as pd
import numpy as np
from PIL import Image
import torch
from transformers import MarianMTModel, MarianTokenizer
import easyocr
import os

# Global variables for models
tokenizer_hi = None
model_hi = None
tokenizer_kn = None
model_kn = None
ocr_reader = None

def load_models():
    """Load translation models"""
    global tokenizer_hi, model_hi, tokenizer_kn, model_kn
    
    try:
        # Load Hindi model
        tokenizer_hi = MarianTokenizer.from_pretrained('saved_model/en_hi')
        model_hi = MarianMTModel.from_pretrained('saved_model/en_hi')
        model_hi.eval()
        
        # Load Kannada model
        tokenizer_kn = MarianTokenizer.from_pretrained('saved_model/en_kn')
        model_kn = MarianMTModel.from_pretrained('saved_model/en_kn')
        model_kn.eval()
        
        return "✅ Models loaded successfully!"
    except Exception as e:
        return f"❌ Error loading models: {str(e)}"

def load_ocr():
    """Load OCR reader"""
    global ocr_reader
    
    try:
        ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        return "✅ OCR loaded successfully!"
    except Exception as e:
        return f"❌ Error loading OCR: {str(e)}"

def translate_text(text, model, tokenizer):
    """Translate text using model"""
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128)
        
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated
    except Exception as e:
        return f"Error: {str(e)}"

def process_image(image):
    """Process image with OCR and translate"""
    global ocr_reader, model_hi, tokenizer_hi, model_kn, tokenizer_kn
    
    if image is None:
        return "Please upload an image", None
    
    if ocr_reader is None:
        return "OCR not loaded. Click 'Initialize OCR' first.", None
    
    if model_hi is None or model_kn is None:
        return "Models not loaded. Click 'Load Models' first.", None
    
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Perform OCR
        results = ocr_reader.readtext(img_array)
        
        if not results:
            return "No text found in image", None
        
        # Extract and translate
        translations = []
        for (bbox, text, prob) in results:
            if prob > 0.5:
                english = text.strip()
                hindi = translate_text(english, model_hi, tokenizer_hi)
                kannada = translate_text(english, model_kn, tokenizer_kn)
                
                translations.append({
                    'English': english,
                    'Hindi (हिंदी)': hindi,
                    'Kannada (ಕನ್ನಡ)': kannada,
                    'Confidence': f"{prob:.2%}"
                })
        
        if translations:
            df = pd.DataFrame(translations)
            message = f"✅ Successfully extracted and translated {len(translations)} items!"
            return message, df
        else:
            return "No high-confidence text found", None
            
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def process_text(text):
    """Process manual text input and translate"""
    global model_hi, tokenizer_hi, model_kn, tokenizer_kn
    
    if not text.strip():
        return "Please enter some text", None
    
    if model_hi is None or model_kn is None:
        return "Models not loaded. Click 'Load Models' first.", None
    
    try:
        items = [item.strip() for item in text.split('\n') if item.strip()]
        
        translations = []
        for item in items:
            hindi = translate_text(item, model_hi, tokenizer_hi)
            kannada = translate_text(item, model_kn, tokenizer_kn)
            
            translations.append({
                'English': item,
                'Hindi (हिंदी)': hindi,
                'Kannada (ಕನ್ನಡ)': kannada
            })
        
        df = pd.DataFrame(translations)
        message = f"✅ Successfully translated {len(translations)} items!"
        return message, df
        
    except Exception as e:
        return f"❌ Error: {str(e)}", None

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Bilingual Menu Translator") as demo:
    
    gr.Markdown("""
    # 🍽️ Bilingual Hotel Menu Display System
    ### Translate your menu from English to Hindi & Kannada using AI
    """)
    
    # Setup section
    with gr.Row():
        with gr.Column():
            load_models_btn = gr.Button("🔄 Load Translation Models", variant="primary")
            model_status = gr.Textbox(label="Model Status", interactive=False)
        
        with gr.Column():
            load_ocr_btn = gr.Button("🔄 Initialize OCR Engine", variant="primary")
            ocr_status = gr.Textbox(label="OCR Status", interactive=False)
    
    load_models_btn.click(load_models, outputs=model_status)
    load_ocr_btn.click(load_ocr, outputs=ocr_status)
    
    gr.Markdown("---")
    
    # Main interface tabs
    with gr.Tabs():
        # Image upload tab
        with gr.TabItem("📸 Image Upload (OCR)"):
            gr.Markdown("Upload a menu image to extract and translate text")
            
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Upload Menu Image")
                    image_btn = gr.Button("🌐 Extract & Translate", variant="primary")
                
                with gr.Column():
                    image_output_msg = gr.Textbox(label="Status", interactive=False)
                    image_output_df = gr.Dataframe(
                        label="Translations",
                        headers=['English', 'Hindi (हिंदी)', 'Kannada (ಕನ್ನಡ)', 'Confidence'],
                        interactive=False
                    )
            
            image_btn.click(
                process_image,
                inputs=image_input,
                outputs=[image_output_msg, image_output_df]
            )
            
            gr.Examples(
                examples=[],
                inputs=image_input,
                label="Example Images"
            )
        
        # Manual text tab
        with gr.TabItem("✍️ Manual Text Entry"):
            gr.Markdown("Enter menu items manually (one per line)")
            
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="Menu Items",
                        placeholder="Butter Chicken\nPaneer Tikka\nMasala Dosa\n...",
                        lines=10
                    )
                    text_btn = gr.Button("🌐 Translate", variant="primary")
                
                with gr.Column():
                    text_output_msg = gr.Textbox(label="Status", interactive=False)
                    text_output_df = gr.Dataframe(
                        label="Translations",
                        headers=['English', 'Hindi (हिंदी)', 'Kannada (ಕನ್ನಡ)'],
                        interactive=False
                    )
            
            text_btn.click(
                process_text,
                inputs=text_input,
                outputs=[text_output_msg, text_output_df]
            )
            
            gr.Examples(
                examples=[
                    ["Butter Chicken\nPaneer Tikka Masala\nMasala Dosa"],
                    ["Dal Makhani\nTandoori Roti\nGulab Jamun"],
                    ["Chicken Biryani\nVeg Pulao\nRaita"]
                ],
                inputs=text_input,
                label="Example Menu Items"
            )
    
    gr.Markdown("---")
    
    # Information section
    gr.Markdown("""
    ### 📊 Model Information
    - **English → Hindi**: Fine-tuned MarianMT model
    - **English → Kannada**: Fine-tuned MarianMT model
    - **OCR Engine**: EasyOCR with English support
    
    ### 💡 Tips
    - For best OCR results, use clear, well-lit images
    - Keep menu items simple and standard
    - One item per line when entering manually
    
    ### 📥 Download Results
    You can copy the translation table or export it using the download button in the dataframe.
    """)

if __name__ == "__main__":
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

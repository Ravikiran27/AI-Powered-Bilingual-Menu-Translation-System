# 🚀 Quick Start Guide

## Step-by-Step Instructions

### Option 1: Automated Setup (Recommended)

#### For Windows (PowerShell):
```powershell
# Run the setup script
.\setup.ps1
```

This will:
- Check Python installation
- Optionally create virtual environment
- Install all dependencies
- Check for trained models
- Guide you through next steps

---

### Option 2: Manual Setup

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Train the Models
```bash
# Launch Jupyter Notebook
jupyter notebook bilingual_menu_translation.ipynb

# Run all cells in the notebook (Ctrl+A, then Shift+Enter)
# This will take 15-30 minutes
```

**What happens during training:**
- Loads the 500-item menu dataset
- Creates beautiful visualizations
- Trains English → Hindi model
- Trains English → Kannada model
- Evaluates with BLEU scores
- Saves models to `saved_model/` folder

#### 3. Run the Web Application

**Option A: Streamlit (Recommended)**
```bash
streamlit run app.py
```
Opens at: http://localhost:8501

**Option B: Gradio**
```bash
python app_gradio.py
```
Opens at: http://localhost:7860

---

## 📱 Using the Application

### Method 1: Upload Menu Image
1. Click "Upload Image (OCR)" tab
2. Upload a clear photo of your menu
3. Click "Extract Text"
4. Review extracted items
5. Click "Translate All Items"
6. Download results as CSV

### Method 2: Enter Text Manually
1. Click "Enter Text Manually" tab
2. Type menu items (one per line)
3. Click "Translate"
4. View translations
5. Download CSV

---

## ⚡ Quick Commands Reference

```bash
# Install everything
pip install -r requirements.txt

# Train models (in Jupyter)
jupyter notebook bilingual_menu_translation.ipynb

# Run Streamlit app
streamlit run app.py

# Run Gradio app
python app_gradio.py

# Check installed packages
pip list

# Update packages
pip install --upgrade -r requirements.txt
```

---

## 🎯 What You Get

### In the Notebook:
✅ Dataset statistics and insights  
✅ Category distribution charts  
✅ Text length analysis  
✅ Translation coverage visualization  
✅ Training progress with loss curves  
✅ BLEU score evaluation  
✅ Sample translations  
✅ Model export and summary  

### In the Web App:
✅ Image upload with OCR  
✅ Manual text entry  
✅ Real-time translation  
✅ Results table  
✅ CSV download  
✅ Statistics dashboard  

---

## 📊 Expected Results

### Dataset:
- 500 menu items
- 3 languages (English, Hindi, Kannada)
- 10+ food categories

### Model Performance:
- Hindi BLEU Score: 60-80 (typical)
- Kannada BLEU Score: 55-75 (typical)
- Training time: 15-30 mins
- Inference: <1 second per item

### File Sizes:
- Dataset: ~100 KB
- Each trained model: ~300 MB
- Total project: ~700 MB

---

## 🐛 Troubleshooting

### "Models not found"
➜ Train models first using the Jupyter notebook

### "CUDA out of memory"
➜ In notebook, reduce BATCH_SIZE to 4 or 2

### "OCR not working"
➜ Check image quality and brightness
➜ Reinstall: `pip install easyocr pillow`

### "Import errors"
➜ Install missing packages: `pip install <package-name>`

### "Slow training"
➜ Normal on CPU (15-30 mins)
➜ Use GPU if available for faster training

---

## 💡 Pro Tips

1. **For best OCR results:**
   - Use high-resolution images
   - Ensure good lighting
   - Avoid shadows and glare
   - Keep text horizontal

2. **For better translations:**
   - Use standard menu terminology
   - Keep descriptions concise
   - Avoid special characters

3. **To save time:**
   - Train models once, reuse forever
   - Use manual text entry for quick tests
   - Batch process multiple items at once

---

## 📁 Project Files Overview

```
bilingual_menu_translation.ipynb  ← Train models here
app.py                           ← Streamlit UI
app_gradio.py                    ← Gradio UI (alternative)
menu_dataset.csv                 ← Your data
requirements.txt                 ← Dependencies
README.md                        ← Full documentation
QUICKSTART.md                    ← This file
setup.ps1                        ← Setup script
```

---

## 🎓 What to Learn Next

- Modify training parameters for better accuracy
- Add more languages (Tamil, Telugu, etc.)
- Increase dataset size
- Fine-tune on domain-specific data
- Deploy to cloud (Heroku, AWS, Azure)
- Create mobile app version

---

## ✅ Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Jupyter notebook opened
- [ ] All cells in notebook executed
- [ ] Models saved to `saved_model/` folder
- [ ] Web app launched (`streamlit run app.py`)
- [ ] Tested with sample images/text
- [ ] Results downloaded as CSV

---

## 🎉 You're Ready!

Your bilingual menu translation system is now set up and ready to use!

Need help? Check:
- README.md for detailed documentation
- Notebook comments for code explanations
- Error messages for specific issues

**Happy translating! 🍽️**

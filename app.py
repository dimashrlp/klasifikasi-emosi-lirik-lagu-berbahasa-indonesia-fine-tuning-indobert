#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indonesian Lyrics Emotion Detection - Thesis Application (FIXED)
Professional UI for Academic Presentation & Thesis Defense

Features:
🎯 Real-time Prediction with 12 Model Selection
📊 Enhanced Model Evaluation with Robust Metrics Detection  
🏆 Best Model Analysis & Performance Insights
🎓 Academic-focused Interface

Author: Thesis Project - Deteksi Emosi Lirik Lagu Indonesia
Usage: streamlit run app.py
"""

import json
import os
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ML imports
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    ML_AVAILABLE = True
    print("✅ Transformers loaded successfully")
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ Transformers not available - fallback mode")

warnings.filterwarnings('ignore')

# ========================================
# CONFIGURATION
# ========================================

st.set_page_config(
    page_title="Klasifikasi Emosi Lirik Lagu Indonesia - Skripsi", 
    page_icon="🎶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Path configuration - adjust based on your folder structure
BASE_PATH = Path.cwd()
if "streamlit_app" in str(BASE_PATH):
    # If running from streamlit_app folder, go up one level
    BASE_PATH = BASE_PATH.parent

MODELS_PATH = BASE_PATH / "models"
DATASETS_PATH = BASE_PATH / "datasets"

# Constants
EMOTIONS = ['bahagia', 'sedih', 'marah', 'takut']
EMOTION_COLORS = {
    'bahagia': '#28a745', 'sedih': '#007bff', 
    'marah': '#dc3545', 'takut': '#6f42c1'
}
EMOTION_EMOJIS = {
    'bahagia': '😊', 'sedih': '😢', 'marah': '😠', 'takut': '😨'
}

# ========================================
# CORE FUNCTIONS
# ========================================

@st.cache_data
def load_dataset_info():
    """Load dataset information from CSV files"""
    info = {'main_count': 0, 'additional_count': 0, 'test_count': 0, 'total_count': 0}
    
    try:
        # Main dataset
        if (DATASETS_PATH / "indosongemot_final_fiks_utf8.csv").exists():
            main_df = pd.read_csv(DATASETS_PATH / "indosongemot_final_fiks_utf8.csv")
            info['main_count'] = len(main_df)
        
        # Additional dataset
        if (DATASETS_PATH / "tambahandatatraincleaned  250.csv").exists():
            add_df = pd.read_csv(DATASETS_PATH / "tambahandatatraincleaned  250.csv")
            info['additional_count'] = len(add_df)
        
        # Test dataset
        if (DATASETS_PATH / "test_set.csv").exists():
            test_df = pd.read_csv(DATASETS_PATH / "test_set.csv")
            info['test_count'] = len(test_df)
        
        info['total_count'] = info['main_count'] + info['additional_count']
        
    except Exception as e:
        st.error(f"Error loading dataset info: {e}")
    
    return info

def get_available_models():
    """Get all available model configurations"""
    if not MODELS_PATH.exists():
        return []
    
    models = []
    for i in range(1, 13):
        model_name = f"model-config-{i}"
        model_path = MODELS_PATH / model_name
        
        if model_path.exists() and (model_path / 'config.json').exists():
            models.append(model_name)
    
    return sorted(models, key=lambda x: int(x.split('-')[-1]))

def load_model_performance(model_name):
    """Load model performance metrics and hyperparameters from results file"""
    results_file = MODELS_PATH / model_name / 'results_summary.txt'
    
    if not results_file.exists():
        return None
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract performance metrics
        performance_patterns = {
            'f1_macro': r'Test F1-Macro:\s*(\d+\.?\d*)',
            'accuracy': r'Test Accuracy:\s*(\d+\.?\d*)%?',
            'f1_weighted': r'Test F1-Weighted:\s*(\d+\.?\d*)',
            'training_duration': r'Training Duration:\s*(\d+\.?\d*)\s*minutes?'
        }
        
        # Extract hyperparameters
        hyperparameter_patterns = {
            'learning_rate': r'Learning Rate:\s*([0-9e\-\.]+)',
            'batch_size': r'Batch Size:\s*(\d+)',
            'weight_decay': r'Weight Decay:\s*(\d+\.?\d*)',
            'epochs': r'Epochs:\s*(\d+)',
            'max_length': r'Max Length:\s*(\d+)'
        }
        
        results = {'model': model_name}
        
        # Extract performance metrics
        for metric, pattern in performance_patterns.items():
            match = re.search(pattern, content)
            if match:
                value = float(match.group(1))
                if metric == 'accuracy' and value > 1:
                    value = value / 100  # Convert percentage to decimal
                results[metric] = value
            else:
                results[metric] = None
        
        # Extract hyperparameters
        for param, pattern in hyperparameter_patterns.items():
            match = re.search(pattern, content)
            if match:
                value = match.group(1)
                if param in ['batch_size', 'epochs', 'max_length']:
                    results[param] = int(value)
                elif param in ['weight_decay']:
                    results[param] = float(value)
                else:  # learning_rate
                    results[param] = value
            else:
                results[param] = None
        
        return results
    
    except Exception as e:
        print(f"Error loading performance for {model_name}: {e}")
        return None

def debug_results_file(model_name):
    """Debug function to show content of results file"""
    results_file = MODELS_PATH / model_name / 'results_summary.txt'
    
    if not results_file.exists():
        return None
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Show file content for debugging
        st.text_area("🔍 Debug: Results file content", content, height=200)
        return content
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def load_detailed_model_performance(model_name):
    """Load detailed performance metrics including per-emotion metrics - CLEAN VERSION"""
    results_file = MODELS_PATH / model_name / 'results_summary.txt'
    
    if not results_file.exists():
        return None
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Patterns specifically for your format
        overall_patterns = {
            'accuracy': [
                r'Test Accuracy:\s*(\d+\.?\d*)%',
                r'Test Accuracy:\s*(\d+\.?\d*)',
            ],
            'f1_macro': [
                r'Test F1-Macro:\s*(\d+\.?\d*)',
            ],
            'f1_weighted': [
                r'Test F1-Weighted:\s*(\d+\.?\d*)',
            ],
            'val_accuracy': [
                r'Validation Accuracy:\s*(\d+\.?\d*)%',
                r'Validation Accuracy:\s*(\d+\.?\d*)',
            ]
        }
        
        results = {'model': model_name}
        
        # Extract overall metrics
        for metric, patterns in overall_patterns.items():
            found_value = None
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                    # Convert percentage to decimal if needed
                    if 'accuracy' in metric and value > 1:
                        value = value / 100
                    found_value = value
                    break
            results[metric] = found_value
        
        # Extract per-emotion metrics and calculate macro averages
        emotions_indo = ['bahagia', 'sedih', 'marah', 'takut']
        per_emotion = {}
        all_precisions = []
        all_recalls = []
        
        for emotion in emotions_indo:
            # Pattern: "- Emotion: F1=0.xxx, Precision=0.xxx, Recall=0.xxx"
            pattern = rf'-\s*{emotion}:\s*F1=(\d+\.?\d*),\s*Precision=(\d+\.?\d*),\s*Recall=(\d+\.?\d*)'
            match = re.search(pattern, content, re.IGNORECASE)
            
            if match:
                f1_val = float(match.group(1))
                prec_val = float(match.group(2))
                rec_val = float(match.group(3))
                
                per_emotion[emotion] = {
                    'f1': f1_val,
                    'precision': prec_val,
                    'recall': rec_val
                }
                
                all_precisions.append(prec_val)
                all_recalls.append(rec_val)
        
        # Calculate macro averages
        if all_precisions:
            results['precision_macro'] = sum(all_precisions) / len(all_precisions)
        else:
            results['precision_macro'] = None
            
        if all_recalls:
            results['recall_macro'] = sum(all_recalls) / len(all_recalls)
        else:
            results['recall_macro'] = None
        
        results['per_emotion'] = per_emotion
        return results
    
    except Exception as e:
        print(f"Error loading detailed performance for {model_name}: {e}")
        return None

def get_all_model_performances():
    """Get performance data for all models"""
    available_models = get_available_models()
    performances = []
    
    for model in available_models:
        perf = load_model_performance(model)
        if perf:
            performances.append(perf)
        else:
            # Add placeholder for models without performance data
            config_id = int(model.split('-')[-1])
            performances.append({
                'model': model,
                'f1_macro': None,
                'accuracy': None,
                'f1_weighted': None,
                'training_duration': None
            })
    
    return performances

# ========================================
# EMOTION PREDICTOR CLASS
# ========================================

class ThesisEmotionPredictor:
    """Professional emotion predictor for thesis demonstration"""
    
    def __init__(self):
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        self.emotion_labels = {0: 'bahagia', 1: 'sedih', 2: 'marah', 3: 'takut'}
    
    def load_model(self, model_name):
        """Load selected model"""
        if not ML_AVAILABLE:
            st.warning("⚠️ Transformers library not available. Using fallback prediction.")
            return False
        
        model_path = MODELS_PATH / model_name
        if not model_path.exists():
            st.error(f"❌ Model path not found: {model_path}")
            return False
        
        try:
            # Clear previous model from memory
            if self.current_model:
                del self.current_model
                del self.current_tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Load new model
            self.current_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.current_model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            self.current_model.eval()
            self.current_model_name = model_name
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading {model_name}: {e}")
            return False
    
    def predict(self, text):
        """Predict emotion with comprehensive output"""
        if not text.strip():
            return None
        
        if self.current_model and self.current_tokenizer:
            return self._real_predict(text)
        else:
            return self._fallback_predict(text)
    
    def _real_predict(self, text):
        """Real IndoBERT prediction"""
        try:
            start_time = time.time()
            
            # Tokenize input
            inputs = self.current_tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True, 
                padding=True, 
                max_length=256
            )
            
            # Predict
            with torch.no_grad():
                outputs = self.current_model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).squeeze().numpy()
                predicted = torch.argmax(logits, dim=-1).item()
            
            # Format results
            return {
                'emotion': self.emotion_labels[predicted],
                'confidence': float(probs[predicted]),
                'probabilities': {self.emotion_labels[i]: float(probs[i]) for i in range(4)},
                'inference_time': time.time() - start_time,
                'model_used': self.current_model_name,
                'prediction_type': 'IndoBERT Real-time'
            }
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return self._fallback_predict(text)
    
    def _fallback_predict(self, text):
        """Fallback prediction when model not loaded"""
        text_lower = text.lower()
        
        # Enhanced keyword-based prediction
        if any(word in text_lower for word in ['bahagia', 'senang', 'gembira', 'cinta', 'indah', 'suka']):
            emotion, confidence = 'bahagia', 0.85
        elif any(word in text_lower for word in ['sedih', 'kecewa', 'menangis', 'hancur', 'luka']):
            emotion, confidence = 'sedih', 0.80
        elif any(word in text_lower for word in ['marah', 'benci', 'kesal', 'jengkel', 'muak']):
            emotion, confidence = 'marah', 0.75
        elif any(word in text_lower for word in ['takut', 'cemas', 'khawatir', 'gelisah', 'was-was']):
            emotion, confidence = 'takut', 0.70
        else:
            emotion, confidence = 'bahagia', 0.50
        
        # Generate probability distribution
        probs = {e: 0.1 for e in EMOTIONS}
        probs[emotion] = confidence
        remaining = (1.0 - confidence) / 3
        for e in EMOTIONS:
            if e != emotion:
                probs[e] = remaining
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': probs,
            'inference_time': 0.001,
            'model_used': 'Keyword-based fallback',
            'prediction_type': 'Keyword Matching'
        }
    
    def get_current_model(self):
        return self.current_model_name

# Initialize predictor
@st.cache_resource
def get_predictor():
    return ThesisEmotionPredictor()

# ========================================
# STYLING
# ========================================

st.markdown("""
<style>
    /* Professional Academic Styling */
    .main-title {
        text-align: center;
        color: #1e3d59;
        font-size: 2.8rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        font-family: 'Georgia', serif;
    }
    
    .subtitle {
        text-align: center;
        color: #5a5a5a;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .model-info {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .best-model {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .top-model {
        background: linear-gradient(135deg, #ffc107 0%, #ff8c00 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .metric-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .emotion-metric {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-left: 4px solid #673ab7;
    }
    
    .insights-box {
        background: #f8f9fa;
        border: 1px solid #28a745;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .stDataFrame {
        border: 1px solid #ddd;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# MAIN APPLICATION
# ========================================

def main():
    """Main thesis-focused application"""
    
    # Header
    st.markdown("""
    <h1 class="main-title">Klasifikasi Emosi Pada Lirik Lagu Berbahasa Indonesia</h1>
    """, unsafe_allow_html=True)
    
    # Initialize predictor
    predictor = get_predictor()
    
    # Sidebar Navigation - CLEAN VERSION FOR THESIS
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        
        page = st.selectbox(
            "Pilih Halaman:",
            [
                "🎯 Klasifikasi", 
                "📊 Model Evaluasi"
            ],
            index=0
        )
        
        st.markdown("---")
        
        # Quick project info
        dataset_info = load_dataset_info()
        available_models = get_available_models()
        
        # System status
        st.markdown("### 🔧 System Status")
        ml_status = "✅ Ready" if ML_AVAILABLE else "⚠️ Fallback"
        models_status = "✅ Found" if available_models else "❌ Missing"
        
        st.write(f"**Models**: {models_status}")
        
        current_model = predictor.get_current_model()
        if current_model:
            st.write(f"**Loaded**: {current_model}")
        else:
            st.write("**Loaded**: None")
    
    # Main content based on selected page - CLEAN VERSION
    if page == "🎯 Klasifikasi":
        show_prediction_interface(predictor)
    elif page == "📊 Model Evaluasi":
        show_model_evaluation()

def show_prediction_interface(predictor):
    """Klasifikasi interface"""
    
    # Model selection
    available_models = get_available_models()
    
    if not available_models:
        st.error("❌ No models found! Please check if models directory exists.")
        return
    
    # Model selection and loading
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create descriptive options for dropdown
        available_models = get_available_models()
        model_options = []
        model_mapping = {}
        
        for model in available_models:
            perf = load_model_performance(model)
            config_num = model.split('-')[-1]
            
            if perf and perf.get('learning_rate') and perf.get('batch_size') and perf.get('weight_decay'):
                # Create descriptive label
                descriptive_label = f"Konfigurasi {config_num} : LR {perf['learning_rate']} | Batch {int(perf['batch_size'])} | WD {perf['weight_decay']:.3f}"
            else:
                # Fallback to simple format
                descriptive_label = f"Konfigurasi {config_num}"
            
            model_options.append(descriptive_label)
            model_mapping[descriptive_label] = model
        
        if model_options:
            selected_display = st.selectbox(
                "**Pilih Model Konfigurasi:**",
                model_options,
                index=len(model_options)-1,  # Default to last model (usually best)
                key="prediction_model_selector"
            )
            selected_model = model_mapping[selected_display]
        else:
            selected_model = None
    
    with col2:
        if selected_model and st.button("🔄 Load Model", type="secondary"):
            with st.spinner(f"Loading {selected_model}..."):
                success = predictor.load_model(selected_model)
                if success:
                    st.success("✅ Model loaded!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Failed to load")
    
    st.markdown("---")
    
    # Input section
    st.markdown("### 📝 Input Lirik Lagu")
    
    # Sample texts for quick testing
    sample_texts = {
        "😊 Bahagia": "Aku merasa bahagia hari ini, semua impianku terwujud dengan sempurna dan hatiku dipenuhi kegembiraan yang luar biasa",
        "😢 Sedih": "Air mata mengalir di pipiku, hatiku hancur berkeping-keping karena kehilangan orang yang sangat kusayangi selamanya",
        "😠 Marah": "Aku muak dengan semua kebohongan ini, sangat kesal dan jengkel dengan sikap yang tidak bisa dipercaya sama sekali",
        "😨 Takut": "Aku takut kehilanganmu, cemas dan khawatir akan masa depan yang tidak pasti dan penuh dengan kegelapan"
    }
    
    # Create input interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        input_text = st.text_area(
            "Masukkan lirik lagu dalam bahasa Indonesia:",
            height=120,
            placeholder="Contoh: Aku merasa bahagia hari ini, matahari bersinar terang dan semua tampak indah di mataku..."
        )
    
    with col2:
        st.markdown("**Contoh Cepat:**")
        for emotion, text in sample_texts.items():
            if st.button(emotion, key=f"sample_{emotion}", use_container_width=True):
                # Store selected text in session state
                st.session_state.selected_sample = text
                st.rerun()
    
    # Use sample text if selected
    if hasattr(st.session_state, 'selected_sample'):
        input_text = st.session_state.selected_sample
        # Clear the selection
        del st.session_state.selected_sample
    
    # Prediction button and results
    if st.button("🔮 **PREDIKSI EMOSI**", type="primary", disabled=not input_text.strip(), use_container_width=True):
        if input_text.strip():
            with st.spinner("🧠 Analyzing emotion using IndoBERT..."):
                result = predictor.predict(input_text)
            
            if result:
                # Main prediction result
                emotion = result['emotion']
                confidence = result['confidence']
                emoji = EMOTION_EMOJIS[emotion]
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>{emoji} Detected Emotion: {emotion.upper()}</h2>
                    <h3>Confidence: {confidence:.1%}</h3>
                    <p>Model: {result['model_used']} | Type: {result['prediction_type']}</p>
                    <p>Inference Time: {result['inference_time']*1000:.1f}ms</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed analysis
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("#### 📊 Probability Distribution")
                    probs = result['probabilities']
                    
                    # Create probability chart
                    fig = px.bar(
                        x=list(probs.keys()),
                        y=list(probs.values()),
                        color=list(probs.keys()),
                        color_discrete_map=EMOTION_COLORS,
                        title="Probability for Each Emotion Class"
                    )
                    fig.update_layout(
                        showlegend=False,
                        yaxis_title="Probability",
                        xaxis_title="Emotion Class",
                        height=400
                    )
                    fig.update_traces(hovertemplate='<b>%{x}</b><br>Probability: %{y:.1%}<extra></extra>')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### 🎯 Confidence Score")
                    
                    # Confidence gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence,
                        number={'valueformat': '.1%'},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"{emotion.title()} Confidence"},
                        gauge={
                            'axis': {'range': [None, 1]},
                            'bar': {'color': EMOTION_COLORS[emotion]},
                            'steps': [
                                {'range': [0, 0.5], 'color': "lightgray"},
                                {'range': [0.5, 0.8], 'color': "gray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.9
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Detailed probabilities table
                    st.markdown("**Detail Probabilitas:**")
                    for emo, prob in probs.items():
                        emoji = EMOTION_EMOJIS[emo]
                        st.write(f"{emoji} **{emo.title()}**: {prob:.1%}")

def show_model_evaluation():
    """Comprehensive model evaluation and comparison - IMPROVED UI FOR THESIS"""
    st.markdown("## 📊 Model Evaluation & Comparison")
    st.markdown("*Evaluasi komprehensif performa 12 konfigurasi model IndoBERT*")
    
    # Load all performances
    performances = get_all_model_performances()
    
    if not performances:
        st.error("❌ No performance data available")
        return
    
    # Create DataFrame and filter valid performances
    df_perf = pd.DataFrame(performances)
    df_valid = df_perf[df_perf['f1_macro'].notna()].copy()
    
    if df_valid.empty:
        st.warning("⚠️ No valid performance data found. Please ensure models have results_summary.txt files.")
        return
    
    # Sort by F1-macro (descending)
    df_valid = df_valid.sort_values('f1_macro', ascending=False).reset_index(drop=True)
    
    # ========================================
    # BEST PERFORMING MODEL - ENHANCED DISPLAY
    # ========================================
    best_model = df_valid.iloc[0]
    
    st.markdown(f"""
    <div class="best-model">
        <h3>🏆 BEST PERFORMING MODEL</h3>
        <h2>{best_model['model']}</h2>
        <p><strong>F1-Macro Score:</strong> {best_model['f1_macro']:.4f} | 
           <strong>Accuracy:</strong> {best_model['accuracy']*100:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load detailed performance for best model
    detailed_perf = load_detailed_model_performance(best_model['model'])
    
    # ========================================
    # BEST MODEL DETAILED METRICS - IMPROVED LAYOUT
    # ========================================
    
    if detailed_perf:
        st.markdown("### 📈 Detailed Performance Analysis - Best Model")
        
        # Split into two main sections with better spacing
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📊 Overall Performance Metrics")
            
            # Overall metrics with improved styling
            overall_metrics = {
                'Test Accuracy': detailed_perf.get('accuracy', best_model['accuracy']),
                'F1-Macro Score': detailed_perf.get('f1_macro', best_model['f1_macro']),
                'Precision (Macro)': detailed_perf.get('precision_macro', 0.0),
                'Recall (Macro)': detailed_perf.get('recall_macro', 0.0)
            }
            
            # Create metric cards with better formatting
            for metric, value in overall_metrics.items():
                if value and value > 0:
                    percentage = value if value <= 1 else value / 100
                    
                    # Create styled metric display
                    st.markdown(f"""
                    <div style="background: #f8f9fa; border-left: 4px solid #007bff; padding: 15px; margin: 10px 0; border-radius: 5px;">
                        <h4 style="margin: 0; color: #495057;">{metric}</h4>
                        <h2 style="margin: 5px 0; color: #007bff;">{percentage:.1%}</h2>
                        <div style="background: #e9ecef; border-radius: 10px; height: 10px; margin-top: 10px;">
                            <div style="background: #007bff; height: 10px; border-radius: 10px; width: {percentage*100}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: #f8f9fa; border-left: 4px solid #6c757d; padding: 15px; margin: 10px 0; border-radius: 5px;">
                        <h4 style="margin: 0; color: #495057;">{metric}</h4>
                        <h2 style="margin: 5px 0; color: #6c757d;">N/A</h2>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 🎭 Per-Emotion Performance")
            
            # Per emotion metrics with enhanced display
            emotion_metrics = detailed_perf.get('per_emotion', {})
            emotions_indo = ['bahagia', 'sedih', 'marah', 'takut']
            emotion_names = {'bahagia': 'Happy', 'sedih': 'Sadness', 'marah': 'Anger', 'takut': 'Fear'}
            emotion_colors = {'bahagia': '#28a745', 'sedih': '#007bff', 'marah': '#dc3545', 'takut': '#6f42c1'}
            
            if emotion_metrics:
                for emotion in emotions_indo:
                    if emotion in emotion_metrics:
                        metrics = emotion_metrics[emotion]
                        color = emotion_colors[emotion]
                        emoji = EMOTION_EMOJIS[emotion]
                        
                        # Enhanced emotion metric display
                        f1_val = metrics.get('f1', 0)
                        prec_val = metrics.get('precision', 0)
                        rec_val = metrics.get('recall', 0)
                        
                        st.markdown(f"""
                        <div style="background: {color}15; border-left: 4px solid {color}; padding: 15px; margin: 10px 0; border-radius: 5px;">
                            <h4 style="margin: 0; color: {color};">{emoji} {emotion_names[emotion]} ({emotion.title()})</h4>
                            <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                                <div style="text-align: center;">
                                    <small style="color: #6c757d;">F1-Score</small><br>
                                    <strong style="color: {color};">{f1_val:.3f}</strong>
                                </div>
                                <div style="text-align: center;">
                                    <small style="color: #6c757d;">Precision</small><br>
                                    <strong style="color: {color};">{prec_val:.3f}</strong>
                                </div>
                                <div style="text-align: center;">
                                    <small style="color: #6c757d;">Recall</small><br>
                                    <strong style="color: {color};">{rec_val:.3f}</strong>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("📝 Detailed per-emotion metrics not available.")
    
    else:
        # Fallback display with available data
        st.markdown("### 📈 Available Performance Metrics - Best Model")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📊 Performance Metrics")
            st.metric("F1-Macro Score", f"{best_model['f1_macro']:.4f}")
            st.metric("Accuracy", f"{best_model['accuracy']*100:.2f}%")
            if best_model.get('f1_weighted'):
                st.metric("F1-Weighted", f"{best_model['f1_weighted']:.4f}")
        
        with col2:
            st.markdown("#### ⚙️ Model Configuration")
            if best_model.get('learning_rate'):
                st.metric("Learning Rate", best_model['learning_rate'])
            if best_model.get('batch_size'):
                st.metric("Batch Size", int(best_model['batch_size']))
            if best_model.get('weight_decay'):
                st.metric("Weight Decay", f"{best_model['weight_decay']:.3f}")
    
    # ========================================
    # COMPREHENSIVE PERFORMANCE TABLE - IMPROVED
    # ========================================
    st.markdown("---")
    st.markdown("### 📋 All Models Performance Comparison")
    st.markdown("*Tabel perbandingan lengkap dengan hyperparameters (diurutkan berdasarkan F1-Macro)*")
    
    # Create two separate tables for better readability
    
    # Table 1: Performance Metrics Only
    st.markdown("#### 🎯 Performance Metrics")
    
    # Prepare performance-only DataFrame
    perf_display_df = df_valid.copy()
    perf_display_df['Rank'] = range(1, len(perf_display_df) + 1)
    perf_display_df['Model'] = perf_display_df['model'].str.replace('model-config-', 'Config-')
    perf_display_df['F1-Macro'] = perf_display_df['f1_macro'].apply(lambda x: f"{x:.4f}")
    perf_display_df['Accuracy'] = perf_display_df['accuracy'].apply(lambda x: f"{x*100:.2f}%")
    perf_display_df['F1-Weighted'] = perf_display_df['f1_weighted'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A')
    
    # Performance table
    perf_table_df = perf_display_df[['Rank', 'Model', 'F1-Macro', 'Accuracy', 'F1-Weighted']]
    
    st.dataframe(
        perf_table_df, 
        hide_index=True, 
        use_container_width=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Model": st.column_config.TextColumn("Model", width="medium"),
            "F1-Macro": st.column_config.TextColumn("F1-Macro", width="medium"),
            "Accuracy": st.column_config.TextColumn("Accuracy", width="medium"),
            "F1-Weighted": st.column_config.TextColumn("F1-Weighted", width="medium")
        }
    )
    
    # Table 2: Hyperparameters Configuration
    st.markdown("#### ⚙️ Hyperparameters Configuration")
    
    # Prepare hyperparameters DataFrame
    hyper_display_df = df_valid.copy()
    hyper_display_df['Model'] = hyper_display_df['model'].str.replace('model-config-', 'Config-')
    
    # Add hyperparameter columns with better formatting
    hyper_display_df['Learning Rate'] = hyper_display_df.apply(lambda x: 
        f"{x['learning_rate']}" if pd.notna(x['learning_rate']) and x['learning_rate'] is not None 
        else 'N/A', axis=1)
    
    hyper_display_df['Batch Size'] = hyper_display_df.apply(lambda x: 
        f"{int(x['batch_size'])}" if pd.notna(x['batch_size']) and x['batch_size'] is not None 
        else 'N/A', axis=1)
    
    hyper_display_df['Weight Decay'] = hyper_display_df.apply(lambda x: 
        f"{x['weight_decay']:.3f}" if pd.notna(x['weight_decay']) and x['weight_decay'] is not None 
        else 'N/A', axis=1)
    
    hyper_display_df['Epochs'] = hyper_display_df.apply(lambda x: 
        f"{int(x['epochs'])}" if pd.notna(x['epochs']) and x['epochs'] is not None 
        else 'N/A', axis=1)
    
    # Hyperparameters table
    hyper_table_df = hyper_display_df[['Model', 'Learning Rate', 'Batch Size', 'Weight Decay', 'Epochs']]
    
    st.dataframe(
        hyper_table_df, 
        hide_index=True, 
        use_container_width=True,
        column_config={
            "Model": st.column_config.TextColumn("Model", width="medium"),
            "Learning Rate": st.column_config.TextColumn("Learning Rate", width="medium"),
            "Batch Size": st.column_config.TextColumn("Batch Size", width="small"),
            "Weight Decay": st.column_config.TextColumn("Weight Decay", width="medium"),
            "Epochs": st.column_config.TextColumn("Epochs", width="small")
        }
    )
    
    # # ========================================
    # # TOP 3 MODELS QUICK COMPARISON
    # # ========================================
    # st.markdown("---")
    # st.markdown("### 🏆 Top 3 Models Quick Comparison")
    
    # top_3 = df_valid.head(3)
    # cols = st.columns(3)
    
    # medals = ["🥇", "🥈", "🥉"]
    # colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
    
    # for i, (idx, model_data) in enumerate(top_3.iterrows()):
    #     with cols[i]:
    #         st.markdown(f"""
    #         <div style="background: linear-gradient(135deg, {colors[i]}20 0%, {colors[i]}10 100%); 
    #                     border: 2px solid {colors[i]}; border-radius: 10px; padding: 20px; text-align: center; margin: 10px 0;">
    #             <h3>{medals[i]} Rank {i+1}</h3>
    #             <h4>{model_data['model'].replace('model-config-', 'Config-')}</h4>
    #             <p><strong>F1-Macro:</strong> {model_data['f1_macro']:.4f}</p>
    #             <p><strong>Accuracy:</strong> {model_data['accuracy']*100:.2f}%</p>
    #             <small>LR: {model_data.get('learning_rate', 'N/A')} | 
    #                    Batch: {int(model_data['batch_size']) if pd.notna(model_data.get('batch_size')) else 'N/A'}</small>
    #         </div>
    #         """, unsafe_allow_html=True)
    
    # # ========================================
    # # PERFORMANCE INSIGHTS
    # # ========================================
    # st.markdown("---")
    # st.markdown("### 💡 Performance Insights")
    
    # col1, col2 = st.columns([1, 1])
    
    # with col1:
    #     # Calculate statistics safely
    #     best_f1 = df_valid['f1_macro'].max()
    #     avg_f1 = df_valid['f1_macro'].mean()
    #     best_acc = df_valid['accuracy'].max() * 100
    #     total_models = len(df_valid)
        
    #     st.markdown(f"""
    #     <div class="insights-box">
    #         <h4>📊 Key Statistics</h4>
    #         <ul>
    #             <li><strong>Best F1-Macro:</strong> {best_f1:.4f}</li>
    #             <li><strong>Average F1-Macro:</strong> {avg_f1:.4f}</li>
    #             <li><strong>Best Accuracy:</strong> {best_acc:.2f}%</li>
    #             <li><strong>Models Evaluated:</strong> {total_models}</li>
    #         </ul>
    #     </div>
    #     """, unsafe_allow_html=True)
    
    # with col2:
    #     # Find best hyperparameter patterns
    #     best_lr = best_model.get('learning_rate', 'N/A')
    #     best_batch = best_model.get('batch_size', 'N/A')
    #     best_wd = best_model.get('weight_decay', 'N/A')
        
    #     # Format values safely
    #     batch_display = f"{int(best_batch)}" if pd.notna(best_batch) and best_batch != 'N/A' else 'N/A'
    #     wd_display = f"{best_wd:.3f}" if pd.notna(best_wd) and best_wd != 'N/A' else 'N/A'
        
    #     st.markdown(f"""
    #     <div class="insights-box">
    #         <h4>🎯 Best Configuration Pattern</h4>
    #         <ul>
    #             <li><strong>Optimal Learning Rate:</strong> {best_lr}</li>
    #             <li><strong>Optimal Batch Size:</strong> {batch_display}</li>
    #             <li><strong>Optimal Weight Decay:</strong> {wd_display}</li>
    #             <li><strong>Class Focus:</strong> Multi-class emotion detection</li>
    #         </ul>
    #     </div>
    #     """, unsafe_allow_html=True)

# # Footer
# def show_footer():
#     st.markdown("---")
#     dataset_info = load_dataset_info()
#     available_models = get_available_models()

# ========================================
# RUN APPLICATION
# ========================================

if __name__ == "__main__":
    main()
    # show_footer()
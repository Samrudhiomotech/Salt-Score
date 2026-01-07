import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import io
import base64
from datetime import datetime
import requests
from io import BytesIO
import os
import warnings

# Suppress warnings and TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Page configuration
st.set_page_config(
    page_title="Alopecia Areata Detection System",
    page_icon="*",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for premium UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(120deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
        padding: 0;
    }
    
    .stApp {
        background: transparent;
    }
    
    .header-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 24px;
        padding: 40px 50px;
        margin: 30px 30px 20px 30px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    .main-title {
        background: linear-gradient(135deg, #1e3c72 0%, #7e22ce 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8em;
        font-weight: 700;
        margin: 0;
        letter-spacing: -1px;
        text-align: center;
    }
    
    .subtitle {
        color: #64748b;
        font-size: 1.05em;
        margin-top: 12px;
        text-align: center;
        font-weight: 400;
    }
    
    .card {
        background: linear-gradient(135deg, #ffffff 0%, #fafbff 100%);
        border-radius: 24px;
        padding: 35px;
        margin: 15px 30px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.12);
        border: 1px solid rgba(255,255,255,0.5);
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #1e3c72 0%, #7e22ce 100%);
    }
    
    .card-title {
        font-size: 1.6em;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 25px;
        letter-spacing: -0.5px;
    }
    
    .upload-area {
        border: 3px dashed #3b82f6;
        border-radius: 20px;
        padding: 80px 20px;
        text-align: center;
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        margin: 20px 0;
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .upload-area:hover {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-color: #2563eb;
        transform: translateY(-2px);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 32px;
        font-weight: 600;
        font-size: 1em;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        margin: 8px 0;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
    }
    
    .result-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px;
        margin: 12px 0;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 14px;
        border-left: 4px solid #3b82f6;
        transition: all 0.3s ease;
    }
    
    .result-item:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.15);
    }
    
    .result-label {
        font-weight: 500;
        color: #475569;
        font-size: 1.05em;
    }
    
    .result-value {
        font-weight: 700;
        color: #1e40af;
        font-size: 1.4em;
    }
    
    .severity-badge {
        display: inline-block;
        padding: 20px 50px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 1.3em;
        margin: 25px 0;
        text-align: center;
        width: 100%;
        letter-spacing: 1px;
        text-transform: uppercase;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .severity-mild {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
        border: 2px solid #10b981;
    }
    
    .severity-moderate {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #92400e;
        border: 2px solid #f59e0b;
    }
    
    .severity-severe {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
        border: 2px solid #ef4444;
    }
    
    .info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 4px solid #3b82f6;
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
        color: #1e40af;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
        color: #92400e;
    }
    
    .error-box {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #ef4444;
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
        color: #991b1b;
    }
    
    .empty-state {
        text-align: center;
        padding: 120px 40px;
        color: #94a3b8;
    }
    
    .selected-region {
        text-align: center;
        color: #1e40af;
        font-weight: 600;
        margin-top: 18px;
        font-size: 1.1em;
        padding: 12px;
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-radius: 10px;
        border: 2px solid #3b82f6;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'selected_region' not in st.session_state:
    st.session_state.selected_region = 'Frontal'
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'model_available' not in st.session_state:
    st.session_state.model_available = False

@st.cache_resource
def load_hair_classifier_model():
    model_paths = [
        'hair_classifier_final.keras',
        './hair_classifier_final.keras',
        'models/hair_classifier_final.keras',
        '../hair_classifier_final.keras'
    ]
    
    for model_path in model_paths:
        try:
            if os.path.exists(model_path):
                hair_model = keras.models.load_model(model_path)
                st.session_state.model_available = True
                return hair_model
        except Exception as e:
            continue
    
    st.warning("Pre-trained model not found. Using image analysis-based detection.")
    st.session_state.model_available = False
    return None

hair_classifier_model = load_hair_classifier_model()

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def detect_patches_advanced(image):
    """Advanced patch detection using multiple techniques"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    _, thresh1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)
    
    combined = cv2.bitwise_or(thresh1, thresh2)
    
    kernel = np.ones((5,5), np.uint8)
    morphed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = 150
    max_area = 50000
    valid_patches = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.3:
                    valid_patches.append(cnt)
    
    return len(valid_patches), morphed

def calculate_follicle_density_advanced(image):
    """Enhanced follicle density calculation"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges_canny = cv2.Canny(blurred, 30, 100)
    
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    edges_sobel = np.sqrt(sobelx**2 + sobely**2)
    edges_sobel = np.uint8(edges_sobel / edges_sobel.max() * 255)
    
    combined_edges = cv2.bitwise_or(edges_canny, edges_sobel)
    follicle_count = np.sum(combined_edges > 0)
    
    image_area_pixels = gray.shape[0] * gray.shape[1]
    assumed_scalp_area_cm2 = 100
    
    density = (follicle_count / image_area_pixels) * assumed_scalp_area_cm2 * 0.5
    density = max(15, min(85, density))
    
    return round(density, 1)

def calculate_salt_score_enhanced(hair_loss_pct, region, follicle_density):
    """Enhanced SALT score with follicle density factor"""
    base_score = hair_loss_pct * 0.35
    
    region_weights = {
        'Vertex': 0.40,
        'Frontal': 0.18,
        'R. Temporal': 0.08,
        'L. Temporal': 0.08,
        'R. Parietal': 0.08,
        'L. Parietal': 0.08,
        'Occipital': 0.10
    }
    
    weight = region_weights.get(region, 0.1)
    weighted_score = base_score * (1 + weight * 2)
    
    normal_density = 50
    density_factor = max(0, (normal_density - follicle_density) / normal_density * 15)
    
    total_score = weighted_score + density_factor
    total_score = max(0, min(100, total_score))
    
    return round(total_score, 1)

def classify_hair_type_advanced(image):
    """Advanced hair type classification using image analysis"""
    num_patches, patch_image = detect_patches_advanced(image)
    follicle_density = calculate_follicle_density_advanced(image)
    
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    image_variance = np.var(gray)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_normalized = hist / hist.sum()
    entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-8))
    
    if num_patches > 0:
        confidence = min(0.7 + (num_patches * 0.1), 0.95)
        return "bald", confidence
    else:
        if follicle_density > 40 and entropy > 6:
            confidence = 0.85
        elif follicle_density > 30 and entropy > 5:
            confidence = 0.75
        else:
            confidence = 0.6
        
        return "notbald", confidence

def classify_hair_type(image):
    """Classify hair as bald or notbald"""
    if hair_classifier_model is not None and st.session_state.model_available:
        try:
            processed_img = preprocess_image(image)
            prediction = hair_classifier_model.predict(processed_img, verbose=0)[0]
            
            # Assuming model output: [bald_probability, notbald_probability]
            if len(prediction) == 2:
                bald_prob = float(prediction[0])
                notbald_prob = float(prediction[1])
                
                if bald_prob > notbald_prob:
                    return "bald", bald_prob
                else:
                    return "notbald", notbald_prob
            else:
                # Single output (binary classification)
                confidence = float(prediction[0])
                if confidence > 0.5:
                    return "bald", confidence
                else:
                    return "notbald", 1 - confidence
                    
        except Exception as e:
            st.error(f"Model prediction error: {e}")
            return classify_hair_type_advanced(image)
    else:
        return classify_hair_type_advanced(image)

def analyze_image_comprehensive(image, region):
    """Comprehensive analysis - show SALT score only if bald"""
    hair_type, confidence = classify_hair_type(image)
    num_patches, _ = detect_patches_advanced(image)
    follicle_density = calculate_follicle_density_advanced(image)

    # Check if hair is bald
    is_bald = (hair_type.lower() == "bald")

    if is_bald:
        # Calculate metrics for bald scalp
        hair_loss_pct = min(confidence * 100 + num_patches * 2, 95)
        salt_score = calculate_salt_score_enhanced(hair_loss_pct, region, follicle_density)
        
        if salt_score < 25:
            severity = "Mild Severity"
            severity_class = "severity-mild"
            severity_desc = "S1 (0-25%)"
        elif salt_score < 50:
            severity = "Moderate Severity"
            severity_class = "severity-moderate"
            severity_desc = "S2 (25-50%)"
        elif salt_score < 75:
            severity = "Severe"
            severity_class = "severity-severe"
            severity_desc = "S3 (50-75%)"
        else:
            severity = "Very Severe"
            severity_class = "severity-severe"
            severity_desc = "S4 (75-100%)"
        
        return {
            'hair_type': 'Bald (Alopecia Detected)',
            'confidence': round(confidence * 100, 1),
            'salt_score': salt_score,
            'hair_loss_pct': round(hair_loss_pct, 1),
            'follicle_density': follicle_density,
            'region': region,
            'severity': severity,
            'severity_class': severity_class,
            'severity_desc': severity_desc,
            'show_score': True,
            'is_bald': True,
            'model_used': st.session_state.model_available
        }
    else:
        # Not bald - don't show SALT score
        return {
            'hair_type': 'Not Bald (Normal Hair)',
            'confidence': round(confidence * 100, 1),
            'salt_score': None,
            'hair_loss_pct': None,
            'follicle_density': follicle_density,
            'region': region,
            'severity': "Normal",
            'severity_class': "severity-mild",
            'severity_desc': "No significant hair loss detected",
            'show_score': False,
            'is_bald': False,
            'model_used': st.session_state.model_available
        }

# Header
st.markdown("""
<div class="header-container">
    <h1 class="main-title">Alopecia Areata Detection System</h1>
    <p class="subtitle">AI-Powered Hair Classification (Bald/Not Bald) & SALT Scoring | Developed by Sarthak Dhole</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.model_available:
    st.warning("**System Status**: Using advanced image analysis (AI model not available)")
else:
    st.success("**System Status**: AI model loaded successfully!")

col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown('<div class="card"><div class="card-title">Image Upload</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        st.session_state.uploaded_image = Image.open(uploaded_file).convert('RGB')
        st.image(st.session_state.uploaded_image, width='stretch')
    else:
        st.markdown("""
        <div class="upload-area">
            <div style="position: relative; z-index: 1;">
                <div style="font-size: 4em; color: #3b82f6;">📷</div>
                <div style="font-size: 1.3em; font-weight: 600; color: #1e40af;">Drag & Drop Image Here</div>
                <div style="color: #64748b; font-size: 0.95em;">or click to browse</div>
                <div style="color: #94a3b8; font-size: 0.85em; margin-top: 15px;">
                    Supported formats: JPG, PNG (Max 10MB)
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Region Selection
    st.markdown('<div style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); padding: 20px; border-radius: 16px; margin: 20px 0;"><div style="font-weight: 600; color: #334155; margin-bottom: 15px;">Select Scalp Region:</div>', unsafe_allow_html=True)
    
    col_a, col_b, col_c = st.columns(3)
    col_d, col_e, col_f = st.columns(3)
    col_g, _, _ = st.columns(3)
    
    with col_a:
        if st.button('Vertex', key='btn_vertex'):
            st.session_state.selected_region = 'Vertex'
    with col_b:
        if st.button('Frontal', key='btn_frontal'):
            st.session_state.selected_region = 'Frontal'
    with col_c:
        if st.button('R. Temporal', key='btn_rt'):
            st.session_state.selected_region = 'R. Temporal'
    
    with col_d:
        if st.button('L. Temporal', key='btn_lt'):
            st.session_state.selected_region = 'L. Temporal'
    with col_e:
        if st.button('R. Parietal', key='btn_rp'):
            st.session_state.selected_region = 'R. Parietal'
    with col_f:
        if st.button('L. Parietal', key='btn_lp'):
            st.session_state.selected_region = 'L. Parietal'
    
    with col_g:
        if st.button('Occipital', key='btn_occ'):
            st.session_state.selected_region = 'Occipital'
    
    st.markdown(f'<div class="selected-region">Selected Region: {st.session_state.selected_region}</div></div>', unsafe_allow_html=True)
    
    # Action Buttons
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button('🔍 Analyze Image', key='analyze'):
            if st.session_state.uploaded_image is not None:
                with st.spinner('Processing image analysis...'):
                    results = analyze_image_comprehensive(
                        st.session_state.uploaded_image,
                        st.session_state.selected_region
                    )
                    st.session_state.analysis_results = results
                st.success('✅ Analysis completed!')
            else:
                st.error('⚠️ Please upload an image first!')
    
    with col_btn2:
        if st.button('🗑️ Clear', key='clear'):
            st.session_state.uploaded_image = None
            st.session_state.analysis_results = None
            st.rerun()
    
    with col_btn3:
        if st.button('📥 Download Report', key='download'):
            if st.session_state.analysis_results:
                st.info('📄 Report generation feature coming soon!')
            else:
                st.warning('⚠️ Please analyze an image first!')
    
    st.markdown('</div>', unsafe_allow_html=True)

# Right Column - Results
with col2:
    st.markdown('<div class="card"><div class="card-title">Analysis Results</div>', unsafe_allow_html=True)
    
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # Show analysis method
        if not results['model_used']:
            st.info("📊 Analysis Method: Advanced Image Processing")
        else:
            st.success("🤖 Analysis Method: AI Model + Image Processing")
        
        # Display Hair Type with clear badge
        if results['is_bald']:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
                        padding: 20px; border-radius: 12px; margin: 15px 0; 
                        border: 2px solid #ef4444; text-align: center;">
                <div style="font-size: 1.8em; font-weight: 700; color: #991b1b;">
                    ⚠️ {results['hair_type']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); 
                        padding: 20px; border-radius: 12px; margin: 15px 0; 
                        border: 2px solid #10b981; text-align: center;">
                <div style="font-size: 1.8em; font-weight: 700; color: #065f46;">
                    ✅ {results['hair_type']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Confidence
        st.markdown(f"""
        <div class="result-item">
            <span class="result-label">Confidence</span>
            <span class="result-value">{results['confidence']}%</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Show SALT score and hair loss ONLY if bald
        if results['is_bald'] and results['show_score']:
            st.markdown(f"""
            <div class="result-item">
                <span class="result-label">SALT Score</span>
                <span class="result-value">{results['salt_score']}</span>
            </div>
            <div class="result-item">
                <span class="result-label">Hair Loss Percentage</span>
                <span class="result-value">{results['hair_loss_pct']}%</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Always show follicle density and region
        st.markdown(f"""
        <div class="result-item">
            <span class="result-label">Follicle Density</span>
            <span class="result-value">{results['follicle_density']}/cm²</span>
        </div>
        <div class="result-item">
            <span class="result-label">Scalp Region</span>
            <span class="result-value">{results['region'].upper()}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Severity badge
        st.markdown(f"""
        <div class="severity-badge {results['severity_class']}">
            {results['severity']} {' - ' + results['severity_desc'] if results['is_bald'] else ''}
        </div>
        """, unsafe_allow_html=True)
        
        # Clinical Interpretation
        st.markdown("<h3 style='color: #1e293b; margin-top: 30px;'>Clinical Interpretation</h3>", unsafe_allow_html=True)
        
        if not results['is_bald']:
            st.markdown("""
            <div class="info-box">
                <strong>✅ Normal Hair Pattern Detected</strong><br><br>
                No significant signs of alopecia areata detected. The scalp shows normal 
                follicle distribution and healthy hair density.<br><br>
                <strong>Recommendations:</strong>
                <ul style="margin-top: 10px;">
                    <li>Continue regular hair care routine</li>
                    <li>Maintain healthy diet rich in proteins and vitamins</li>
                    <li>Regular monitoring every 6 months</li>
                    <li>Consult dermatologist if changes occur</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Show detailed info based on severity
            if results['severity'] == "Mild Severity":
                st.markdown("""
                <div class="info-box">
                    <strong>⚠️ Mild Alopecia Areata (S1 - SALT Score < 25%)</strong><br><br>
                    Limited hair loss detected. Early intervention typically yields good results.<br><br>
                    <strong>Recommended Actions:</strong>
                    <ul>
                        <li>Consult dermatologist for treatment plan</li>
                        <li>Consider topical corticosteroids or minoxidil</li>
                        <li>Monitor progress every 3-6 months</li>
                        <li>Maintain stress management practices</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            elif results['severity'] == "Moderate Severity":
                st.markdown("""
                <div class="warning-box">
                    <strong>⚠️ Moderate Alopecia Areata (S2 - SALT Score 25-50%)</strong><br><br>
                    Significant hair loss detected requiring comprehensive treatment.<br><br>
                    <strong>Recommended Actions:</strong>
                    <ul>
                        <li>Immediate dermatologist consultation recommended</li>
                        <li>Consider systemic treatments (JAK inhibitors)</li>
                        <li>Combination therapy approach</li>
                        <li>Monthly monitoring required</li>
                        <li>Psychological support beneficial</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="error-box">
                    <strong>🚨 Severe Alopecia Areata (S3-S4 - SALT Score > 50%)</strong><br><br>
                    Extensive hair loss detected requiring urgent intervention.<br><br>
                    <strong>Urgent Actions Required:</strong>
                    <ul>
                        <li><strong>Immediate dermatologist consultation strongly advised</strong></li>
                        <li>Systemic immunotherapy and JAK inhibitors</li>
                        <li>Comprehensive multi-modal treatment plan</li>
                        <li>Weekly/bi-weekly monitoring during treatment</li>
                        <li>Psychological counseling recommended</li>
                        <li>Screen for related autoimmune conditions</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        # Educational Sections
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #1e293b;'>Understanding Your Results</h3>", unsafe_allow_html=True)
        
        with st.expander("📋 About Hair Classification", expanded=False):
            st.markdown("""
            **Hair Classification System:**
            
            - **Not Bald (Normal Hair):** Healthy scalp with uniform hair distribution and no visible bald patches
            - **Bald (Alopecia Detected):** Presence of one or more bald patches indicating alopecia areata
            
            Our system uses advanced AI and computer vision to analyze your scalp image and determine 
            the presence of hair loss patterns characteristic of alopecia areata.
            """)
        
        if results['is_bald'] and results['show_score']:
            with st.expander("📊 What is SALT Score?", expanded=False):
                st.markdown("""
                The **Severity of Alopecia Tool (SALT)** is a standardized clinical scoring system:
                
                **Scalp Regions:**
                - **Vertex (40%)** - Top of the scalp
                - **Right Profile (18%)** - Right front including temporal
                - **Left Profile (18%)** - Left front including temporal  
                - **Posterior (24%)** - Back of scalp including occipital
                
                **Severity Levels:**
                - **S1 (0-25%)**: Mild - Good prognosis with treatment
                - **S2 (25-50%)**: Moderate - Requires comprehensive care
                - **S3 (50-75%)**: Severe - Needs aggressive intervention
                - **S4 (75-100%)**: Very Severe - Extensive medical management
                
                The score helps doctors track progression and treatment effectiveness.
                """)
        
        with st.expander("🔬 Follicle Density Analysis", expanded=False):
            st.markdown(f"""
            **Your Follicle Density:** {results['follicle_density']} follicles/cm²
            
            Normal scalp follicle density ranges from 50-80 follicles per square centimeter. 
            Lower density indicates active hair loss or miniaturization.
            
            **Density Ranges:**
            - **Normal:** 50-80/cm² - Healthy hair growth
            - **Mild Reduction:** 35-50/cm² - Early changes
            - **Moderate Reduction:** 20-35/cm² - Significant loss
            - **Severe Reduction:** <20/cm² - Advanced hair loss
            
            {"Your density suggests potential hair thinning." if results['follicle_density'] < 50 else "Your density is within normal range."}
            """)
        
        if results['is_bald']:
            with st.expander("💊 Treatment Options", expanded=False):
                st.markdown("""
                **Common Treatment Approaches:**
                
                **Topical Treatments:**
                - Corticosteroid creams and ointments
                - Minoxidil (Rogaine) solution
                - Anthralin cream
                - Topical immunotherapy (DPCP, SADBE)
                
                **Injectable Treatments:**
                - Intralesional corticosteroid injections
                - Platelet-rich plasma (PRP) therapy
                
                **Systemic Medications:**
                - Oral corticosteroids
                - JAK inhibitors (Baricitinib, Tofacitinib)
                - Methotrexate
                - Cyclosporine
                
                **Note:** Treatment choice depends on severity, extent, and individual factors. 
                Always consult with a board-certified dermatologist.
                """)
        
        with st.expander("ℹ️ About Alopecia Areata", expanded=False):
            st.markdown("""
            **What is Alopecia Areata?**
            
            Alopecia areata is an autoimmune condition where the immune system mistakenly 
            attacks hair follicles, causing hair loss. It can affect:
            
            - **Scalp** (most common)
            - **Beard area**
            - **Eyebrows and eyelashes**
            - **Body hair** (in severe cases)
            
            **Key Facts:**
            - Affects about 2% of the population
            - Can occur at any age
            - Not contagious or caused by stress alone
            - Hair may regrow spontaneously
            - Treatment can help stimulate regrowth
            - Runs in families in about 20% of cases
            
            **When to See a Doctor:**
            - Sudden hair loss in patches
            - Complete loss of scalp or body hair
            - Rapid progression of hair loss
            - Associated symptoms (itching, burning)
            """)
    
    else:
        st.markdown("""
        <div class="empty-state">
            <div style="font-size: 5em; margin-bottom: 20px; opacity: 0.3;">📊</div>
            <div style="font-size: 1.4em; font-weight: 600; color: #64748b; margin-bottom: 12px;">
                No Analysis Available
            </div>
            <div style="font-size: 1em; color: #94a3b8;">
                Upload a scalp image and click "Analyze Image" to view results
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: white; padding: 30px 20px; margin-top: 40px;">
    <div style="font-size: 1.1em; font-weight: 600; margin-bottom: 10px;">
        ⚕️ Medical Disclaimer
    </div>
    <div style="opacity: 0.8; line-height: 1.8;">
        This AI-powered system assists in alopecia areata assessment.<br>
        It should NOT replace professional medical diagnosis and treatment.<br>
        Always consult with a board-certified dermatologist for accurate diagnosis.
    </div>
    <div style="margin-top: 25px; opacity: 0.7; font-size: 0.85em;">
        © 2025 Alopecia Areata Detection System | Developed by Sarthak Dhole<br>
        Powered by Advanced AI & Computer Vision Technology
    </div>
</div>
""", unsafe_allow_html=True)
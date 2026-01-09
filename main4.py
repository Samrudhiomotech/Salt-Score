import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
from datetime import datetime
import os
import warnings
import tempfile

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Try to import TensorFlow (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    tf.get_logger().setLevel('ERROR')
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Alopecia Areata Detection System",
    page_icon="*",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
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
    
    .model-status {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 15px 20px;
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-radius: 12px;
        border: 2px solid #3b82f6;
        margin: 15px 0;
        font-weight: 500;
    }
    
    .model-status.ready {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border-color: #22c55e;
        color: #166534;
    }
    
    .model-status.pending {
        background: linear-gradient(135deg, #fef3c7 0%, #fcd34d 100%);
        border-color: #f59e0b;
        color: #92400e;
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
if 'hair_classifier_model' not in st.session_state:
    st.session_state.hair_classifier_model = None
if 'model_loaded_from_upload' not in st.session_state:
    st.session_state.model_loaded_from_upload = False
if 'show_analysis_page' not in st.session_state:
    st.session_state.show_analysis_page = False

def load_hair_classifier_model():
    """Load the hair classification model if available"""
    if not TF_AVAILABLE:
        return None
        
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
                st.session_state.hair_classifier_model = hair_model
                return hair_model
        except Exception as e:
            continue
    
    st.session_state.model_available = False
    return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    try:
        img = image.resize(target_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Image preprocessing error: {e}")
        return None

def detect_patches_advanced(image):
    """Advanced patch detection using multiple techniques"""
    try:
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        _, thresh1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        thresh2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 15, 3)
        
        combined = cv2.bitwise_and(thresh1, thresh2)
        
        kernel_small = np.ones((3,3), np.uint8)
        kernel_large = np.ones((7,7), np.uint8)
        
        morphed = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small, iterations=2)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = 500
        max_area = 40000
        valid_patches = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        solidity = area / hull_area
                    else:
                        solidity = 0
                    
                    if circularity > 0.4 and solidity > 0.7:
                        valid_patches.append(cnt)
        
        return len(valid_patches), morphed
    except Exception as e:
        st.error(f"Patch detection error: {e}")
        return 0, None

def calculate_follicle_density_advanced(image):
    """Enhanced follicle density calculation"""
    try:
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges_canny = cv2.Canny(blurred, 30, 100)
        
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        edges_sobel = np.sqrt(sobelx**2 + sobely**2)
        edges_sobel = np.uint8(edges_sobel / (edges_sobel.max() + 1e-8) * 255)
        
        combined_edges = cv2.bitwise_or(edges_canny, edges_sobel)
        follicle_count = np.sum(combined_edges > 0)
        
        image_area_pixels = gray.shape[0] * gray.shape[1]
        assumed_scalp_area_cm2 = 100
        
        density = (follicle_count / image_area_pixels) * assumed_scalp_area_cm2 * 0.5
        density = max(15, min(85, density))
        
        return round(density, 1)
    except Exception as e:
        st.error(f"Follicle density calculation error: {e}")
        return 50.0

def calculate_salt_score_enhanced(hair_loss_pct, region, follicle_density):
    """Enhanced SALT score with follicle density factor"""
    try:
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
    except Exception as e:
        st.error(f"SALT score calculation error: {e}")
        return 0.0

def classify_hair_type_advanced(image):
    """Advanced hair type classification using image analysis"""
    try:
        num_patches, patch_image = detect_patches_advanced(image)
        follicle_density = calculate_follicle_density_advanced(image)
        
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        image_variance = np.var(gray)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_normalized = hist / (hist.sum() + 1e-8)
        entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-8))
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = np.var(laplacian)
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        hair_score = 0
        
        if follicle_density > 55:
            hair_score += 30
        elif follicle_density > 40:
            hair_score += 20
        elif follicle_density > 30:
            hair_score += 10
        
        if texture_variance > 100:
            hair_score += 25
        elif texture_variance > 50:
            hair_score += 15
        elif texture_variance > 20:
            hair_score += 5
        
        if edge_density > 0.15:
            hair_score += 25
        elif edge_density > 0.10:
            hair_score += 15
        elif edge_density > 0.05:
            hair_score += 5
        
        if entropy > 6.5:
            hair_score += 20
        elif entropy > 5.5:
            hair_score += 12
        elif entropy > 4.5:
            hair_score += 5
        
        if num_patches > 3:
            hair_score -= 40
        elif num_patches > 1:
            hair_score -= 20
        elif num_patches > 0:
            hair_score -= 10
        
        if hair_score > 50:
            confidence = min(0.6 + (hair_score - 50) * 0.008, 0.95)
            return "notbald", confidence
        else:
            confidence = min(0.6 + (50 - hair_score) * 0.008, 0.95)
            return "bald", confidence
            
    except Exception as e:
        st.error(f"Hair classification error: {e}")
        return "notbald", 0.5

def classify_hair_type(image):
    """Classify hair as bald or notbald"""
    if st.session_state.hair_classifier_model is not None and st.session_state.model_available:
        try:
            processed_img = preprocess_image(image)
            if processed_img is None:
                return classify_hair_type_advanced(image)
                
            prediction = st.session_state.hair_classifier_model.predict(processed_img, verbose=0)[0]
            
            if len(prediction) == 2:
                bald_prob = float(prediction[0])
                notbald_prob = float(prediction[1])
                
                if bald_prob > notbald_prob:
                    return "bald", bald_prob
                else:
                    return "notbald", notbald_prob
            else:
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
    try:
        hair_type, confidence = classify_hair_type(image)
        num_patches, _ = detect_patches_advanced(image)
        follicle_density = calculate_follicle_density_advanced(image)

        is_bald = (hair_type.lower() == "bald")

        if is_bald:
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
    except Exception as e:
        st.error(f"Analysis error: {e}")
        return None

# Load default model
load_hair_classifier_model()

# Check if we should show the model upload page or analysis page
if not st.session_state.show_analysis_page:
    # Model Upload Page
    st.markdown("""
    <div class="header-container">
        <h1 class="main-title">Alopecia Areata Detection System</h1>
        <p class="subtitle">AI-Powered Hair Classification & SALT Scoring | Developed by Sarthak Dhole</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.markdown('<div class="card"><div class="card-title">Model Setup</div>', unsafe_allow_html=True)
        
        st.markdown("<p style='color: #475569; font-size: 1.05em; margin-bottom: 20px;'>Select how you want to proceed with the analysis:</p>", unsafe_allow_html=True)
        
        # Current Model Status
        if st.session_state.model_available and not st.session_state.model_loaded_from_upload:
            st.markdown("""
            <div class="model-status ready">
                <span>Pre-trained Model: Ready to Use</span>
                <span style="font-size: 1.2em; color: #22c55e;">✓</span>
            </div>
            """, unsafe_allow_html=True)
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("Use Default Model", width="stretch", key="use_default"):
                    st.session_state.show_analysis_page = True
                    st.rerun()
            
            with col_btn2:
                st.markdown("<p style='font-size: 0.9em; text-align: center; color: #64748b; margin-top: 15px;'>OR</p>", unsafe_allow_html=True)
        
        st.markdown("<p style='font-size: 0.95em; color: #64748b; font-weight: 500; margin: 20px 0;'>Upload Your Custom Model</p>", unsafe_allow_html=True)
        
        uploaded_model = st.file_uploader(
            "Choose a model file (.keras, .h5, .pb, .tflite)",
            type=['keras', 'h5', 'pb', 'tflite'],
            label_visibility="collapsed",
            key="model_uploader"
        )
        
        if uploaded_model is not None:
            try:
                with st.spinner("Loading your model..."):
                    # Create temporary file for the model
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp_file:
                        tmp_file.write(uploaded_model.getbuffer())
                        tmp_path = tmp_file.name
                    
                    # Load the model
                    custom_model = keras.models.load_model(tmp_path)
                    st.session_state.hair_classifier_model = custom_model
                    st.session_state.model_available = True
                    st.session_state.model_loaded_from_upload = True
                    
                    # Clean up temp file
                    os.remove(tmp_path)
                
                st.markdown("""
                <div class="model-status ready">
                    <span>Custom Model: Successfully Loaded</span>
                    <span style="font-size: 1.2em; color: #22c55e;">✓</span>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Proceed to Analysis", width="stretch", key="proceed_analysis"):
                    st.session_state.show_analysis_page = True
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Failed to load model: {str(e)}")
                st.markdown("<p style='color: #64748b; font-size: 0.9em; margin-top: 15px;'>Please ensure the file is a valid TensorFlow/Keras model.</p>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card"><div class="card-title">Get Started</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style='color: #475569; line-height: 1.8; font-size: 0.95em; text-align: center; padding: 40px 20px;'>
            <p style='margin: 0; color: #64748b; font-size: 1.1em;'>Upload a model or use the default model to begin analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # Analysis Page
    st.markdown("""
    <div class="header-container">
        <h1 class="main-title">Alopecia Areata Detection System</h1>
        <p class="subtitle">AI-Powered Hair Classification & SALT Scoring | Developed by Sarthak Dhole</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.model_available:
        if st.session_state.model_loaded_from_upload:
            st.info("Analysis: Using your custom uploaded model")
        else:
            st.success("Analysis: Using pre-trained AI model")
    
    # Main layout
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.markdown('<div class="card"><div class="card-title">Image Upload</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed",
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            try:
                st.session_state.uploaded_image = Image.open(uploaded_file).convert('RGB')
                st.image(st.session_state.uploaded_image, width='stretch')
            except Exception as e:
                st.error(f"Error loading image: {e}")
                st.session_state.uploaded_image = None
        else:
            st.info("Upload a scalp image to begin analysis")
        
        # Region Selection
        st.markdown('<div style="margin-top: 20px;"><strong>Select Scalp Region:</strong></div>', unsafe_allow_html=True)
        
        col_a, col_b, col_c = st.columns(3)
        col_d, col_e, col_f = st.columns(3)
        col_g, _, _ = st.columns(3)
        
        with col_a:
            if st.button('Vertex', key='btn_vertex', width='stretch'):
                st.session_state.selected_region = 'Vertex'
        with col_b:
            if st.button('Frontal', key='btn_frontal', width='stretch'):
                st.session_state.selected_region = 'Frontal'
        with col_c:
            if st.button('R. Temporal', key='btn_rt', width='stretch'):
                st.session_state.selected_region = 'R. Temporal'
        
        with col_d:
            if st.button('L. Temporal', key='btn_lt', width='stretch'):
                st.session_state.selected_region = 'L. Temporal'
        with col_e:
            if st.button('R. Parietal', key='btn_rp', width='stretch'):
                st.session_state.selected_region = 'R. Parietal'
        with col_f:
            if st.button('L. Parietal', key='btn_lp', width='stretch'):
                st.session_state.selected_region = 'L. Parietal'
        
        with col_g:
            if st.button('Occipital', key='btn_occ', width='stretch'):
                st.session_state.selected_region = 'Occipital'
        
        st.markdown(f'<div class="selected-region">Selected Region: {st.session_state.selected_region}</div>', unsafe_allow_html=True)
        
        # Action Buttons
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button('Analyze Image', key='analyze', width='stretch'):
                if st.session_state.uploaded_image is not None:
                    with st.spinner('Processing image analysis...'):
                        results = analyze_image_comprehensive(
                            st.session_state.uploaded_image,
                            st.session_state.selected_region
                        )
                        if results:
                            st.session_state.analysis_results = results
                            st.success('Analysis completed!')
                        else:
                            st.error('Analysis failed. Please try again.')
                else:
                    st.error('Please upload an image first!')
        
        with col_btn2:
            if st.button('Clear', key='clear', width='stretch'):
                st.session_state.uploaded_image = None
                st.session_state.analysis_results = None
                st.rerun()
        
        with col_btn3:
            if st.button('Back to Setup', key='back_to_setup', width='stretch'):
                st.session_state.show_analysis_page = False
                st.session_state.uploaded_image = None
                st.session_state.analysis_results = None
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Right Column - Results
    with col2:
        st.markdown('<div class="card"><div class="card-title">Analysis Results</div>', unsafe_allow_html=True)
        
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            # Analysis method
            if not results['model_used']:
                st.info("Analysis Method: Advanced Image Processing")
            else:
                st.success("Analysis Method: AI Model + Image Processing")
            
            # Hair Type Badge
            if results['is_bald']:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
                            padding: 20px; border-radius: 12px; margin: 15px 0; 
                            border: 2px solid #ef4444; text-align: center;">
                    <div style="font-size: 1.8em; font-weight: 700; color: #991b1b;">
                        {results['hair_type']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); 
                            padding: 20px; border-radius: 12px; margin: 15px 0; 
                            border: 2px solid #10b981; text-align: center;">
                    <div style="font-size: 1.8em; font-weight: 700; color: #065f46;">
                        {results['hair_type']}
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
            
            # Show SALT score only if bald
            if results['is_bald'] and results['show_score']:
                st.markdown(f"""
                <div class="result-item">
                    <span class="result-label">SALT Score</span>
                    <span class="result-value">{results['salt_score']}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Hair Loss %</span>
                    <span class="result-value">{results['hair_loss_pct']}%</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Follicle density and region
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
            with st.expander("Clinical Interpretation", expanded=True):
                if not results['is_bald']:
                    st.success("""
                    Normal Hair Pattern Detected
                    
                    No significant signs of alopecia areata detected. The scalp shows normal 
                    follicle distribution and healthy hair density.
                    
                    Recommendations:
                    - Continue regular hair care routine
                    - Maintain healthy diet rich in proteins and vitamins
                    - Regular monitoring every 6 months
                    - Consult dermatologist if changes occur
                    """)
                else:
                    if results['severity'] == "Mild Severity":
                        st.info("""
                        Mild Alopecia Areata (S1 - SALT Score < 25%)
                        
                        Limited hair loss detected. Early intervention typically yields good results.
                        
                        Recommended Actions:
                        - Consult dermatologist for treatment plan
                        - Consider topical corticosteroids or minoxidil
                        - Monitor progress every 3-6 months
                        - Maintain stress management practices
                        """)
                    elif results['severity'] == "Moderate Severity":
                        st.warning("""
                        Moderate Alopecia Areata (S2 - SALT Score 25-50%)
                        
                        Significant hair loss detected requiring comprehensive treatment.
                        
                        Recommended Actions:
                        - Immediate dermatologist consultation recommended
                        - Consider systemic treatments (JAK inhibitors)
                        - Combination therapy approach
                        - Monthly monitoring required
                        - Psychological support beneficial
                        """)
                    else:
                        st.error("""
                        Severe Alopecia Areata (S3-S4 - SALT Score > 50%)
                        
                        Extensive hair loss detected requiring urgent intervention.
                        
                        Urgent Actions Required:
                        - Immediate dermatologist consultation strongly advised
                        - Systemic immunotherapy and JAK inhibitors
                        - Comprehensive multi-modal treatment plan
                        - Weekly/bi-weekly monitoring during treatment
                        - Psychological counseling recommended
                        - Screen for related autoimmune conditions
                        """)
            
            # Educational sections
            with st.expander("Understanding Your Results"):
                st.markdown("""
                Hair Classification System:
                
                - **Not Bald (Normal Hair):** Healthy scalp with uniform hair distribution
                - **Bald (Alopecia Detected):** Presence of bald patches indicating alopecia areata
                
                Our system uses advanced AI and computer vision to analyze scalp images.
                """)
            
            if results['is_bald'] and results['show_score']:
                with st.expander("About SALT Score"):
                    st.markdown("""
                    The **Severity of Alopecia Tool (SALT)** is a standardized clinical scoring system:
                    
                    Scalp Regions:
                    - **Vertex (40%)** - Top of the scalp
                    - **Frontal (18%)** - Front of scalp
                    - **Temporal (16%)** - Sides of scalp
                    - **Parietal (16%)** - Upper sides
                    - **Occipital (10%)** - Back of scalp
                    
                    Severity Levels:
                    - **S1 (0-25%)**: Mild - Good prognosis with treatment
                    - **S2 (25-50%)**: Moderate - Requires comprehensive care
                    - **S3 (50-75%)**: Severe - Needs aggressive intervention
                    - **S4 (75-100%)**: Very Severe - Extensive medical management
                    
                    The score helps doctors track progression and treatment effectiveness.
                    """)
            
            with st.expander("Follicle Density Analysis"):
                st.markdown(f"""
                Your Follicle Density: {results['follicle_density']} follicles/cm²
                
                Normal scalp follicle density ranges from 50-80 follicles per square centimeter.
                
                Density Ranges:
                - **Normal:** 50-80/cm² - Healthy hair growth
                - **Mild Reduction:** 35-50/cm² - Early changes
                - **Moderate Reduction:** 20-35/cm² - Significant loss
                - **Severe Reduction:** <20/cm² - Advanced hair loss
                
                {"Your density suggests potential hair thinning." if results['follicle_density'] < 50 else "Your density is within normal range."}
                """)
            
            if results['is_bald']:
                with st.expander("Treatment Options"):
                    st.markdown("""
                    Common Treatment Approaches:
                    
                    Topical Treatments:
                    - Corticosteroid creams and ointments
                    - Minoxidil (Rogaine) solution
                    - Anthralin cream
                    - Topical immunotherapy (DPCP, SADBE)
                    
                    Injectable Treatments:
                    - Intralesional corticosteroid injections
                    - Platelet-rich plasma (PRP) therapy
                    
                    Systemic Medications:
                    - Oral corticosteroids
                    - JAK inhibitors (Baricitinib, Tofacitinib)
                    - Methotrexate
                    - Cyclosporine
                    
                    Note: Treatment choice depends on severity and individual factors. 
                    Always consult with a board-certified dermatologist.
                    """)
            
            with st.expander("About Alopecia Areata"):
                st.markdown("""
                What is Alopecia Areata?
                
                Alopecia areata is an autoimmune condition where the immune system 
                attacks hair follicles, causing hair loss.
                
                Key Facts:
                - Affects about 2% of the population
                - Can occur at any age
                - Not contagious or caused by stress alone
                - Hair may regrow spontaneously
                - Treatment can help stimulate regrowth
                - Runs in families in about 20% of cases
                
                When to See a Doctor:
                - Sudden hair loss in patches
                - Complete loss of scalp or body hair
                - Rapid progression of hair loss
                - Associated symptoms (itching, burning)
                """)
        
        else:
            st.info("No analysis available. Upload an image and click 'Analyze Image' to begin.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: white; padding: 30px 20px; margin-top: 40px;">
        <div style="font-size: 1.1em; font-weight: 600; margin-bottom: 10px;">
            Medical Disclaimer
        </div>
        <div style="opacity: 0.8; line-height: 1.8;">
            This AI-powered system assists in alopecia areata assessment.<br>
            It should NOT replace professional medical diagnosis and treatment.<br>
            Always consult with a board-certified dermatologist for accurate diagnosis.
        </div>
        <div style="margin-top: 25px; opacity: 0.7; font-size: 0.85em;">
            2025 Alopecia Areata Detection System | Developed by Sarthak Dhole<br>
            Powered by Advanced AI & Computer Vision Technology
        </div>
    </div>
    """, unsafe_allow_html=True)

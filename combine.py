import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import os
import warnings
import tempfile
from datetime import datetime

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    tf.get_logger().setLevel('ERROR')
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Hair Health & Alopecia System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Unified Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(120deg, #0f766e 0%, #14b8a6 50%, #2dd4bf 100%);
        padding: 0;
    }
    
    .stApp {
        background: transparent;
    }
    
    .header-container {
        background: linear-gradient(135deg, #ffffff 0%, #f0fdfa 100%);
        border-radius: 24px;
        padding: 40px 50px;
        margin: 30px 30px 20px 30px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    .main-title {
        background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%);
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
        background: linear-gradient(135deg, #ffffff 0%, #f0fdfa 100%);
        border-radius: 24px;
        padding: 35px;
        margin: 15px 30px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.12);
        border: 1px solid rgba(255,255,255,0.5);
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
        background: linear-gradient(90deg, #0f766e 0%, #14b8a6 100%);
    }
    
    .card-title {
        font-size: 1.6em;
        font-weight: 600;
        color: #0f766e;
        margin-bottom: 25px;
        letter-spacing: -0.5px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%);
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
        box-shadow: 0 4px 15px rgba(20, 184, 166, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(20, 184, 166, 0.4);
    }
    
    .result-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px;
        margin: 12px 0;
        background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%);
        border-radius: 14px;
        border-left: 4px solid #14b8a6;
    }
    
    .result-label {
        font-weight: 500;
        color: #0f766e;
        font-size: 1.05em;
    }
    
    .result-value {
        font-weight: 700;
        color: #0d9488;
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
    
    .hair-strength-display {
        background: white;
        padding: 20px;
        border-radius: 14px;
        margin: 12px 0;
        border-left: 5px solid #14b8a6;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 12px rgba(20, 184, 166, 0.1);
    }
    
    .strength-value {
        font-weight: 700;
        font-size: 1.3em;
        min-width: 120px;
        text-align: right;
        padding: 10px 15px;
        border-radius: 8px;
    }
    
    .strength-weak {
        color: #991b1b;
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    }
    
    .strength-normal {
        color: #065f46;
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
    }
    
    .strength-strong {
        color: #0f766e;
        background: linear-gradient(135deg, #ccfbf1 0%, #99f6e4 100%);
    }
    
    .average-strength {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        padding: 25px;
        border-radius: 16px;
        margin: 20px 0;
        border: 3px solid #0284c7;
        text-align: center;
    }
    
    .average-strength-value {
        color: #0284c7;
        font-weight: 700;
        font-size: 3em;
        margin-top: 12px;
    }
    
    .info-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #3b82f6;
        margin: 15px 0;
        color: #1e40af;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #f59e0b;
        margin: 15px 0;
        color: #92400e;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #10b981;
        margin: 15px 0;
        color: #065f46;
    }

    .score-display {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        border: 3px solid #0284c7;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        margin: 30px 0;
    }

    .score-number {
        font-size: 4.5em;
        font-weight: 800;
        color: #0284c7;
        margin: 15px 0;
    }

    .analysis-card {
        background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%);
        padding: 25px;
        border-radius: 16px;
        margin: 20px 0;
        border-left: 6px solid #14b8a6;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    .grade-badge {
        background: white;
        padding: 8px 20px;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
        margin: 10px 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 'setup'
if 'model_available' not in st.session_state:
    st.session_state.model_available = False
if 'hair_classifier_model' not in st.session_state:
    st.session_state.hair_classifier_model = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'selected_region' not in st.session_state:
    st.session_state.selected_region = 'Frontal'
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'hair_strength' not in st.session_state:
    st.session_state.hair_strength = {
        'Vertex': 900, 'Frontal': 900, 'R. Temporal': 900,
        'L. Temporal': 900, 'R. Parietal': 900,
        'L. Parietal': 900, 'Occipital': 900
    }

def load_hair_classifier_model():
    """Load the hair classification model"""
    if not TF_AVAILABLE:
        return None
    
    model_paths = [
        'hair_classifier_final.keras', './hair_classifier_final.keras',
        'models/hair_classifier_final.keras', '../hair_classifier_final.keras'
    ]
    
    for model_path in model_paths:
        try:
            if os.path.exists(model_path):
                model = keras.models.load_model(model_path)
                st.session_state.model_available = True
                st.session_state.hair_classifier_model = model
                return model
        except Exception:
            continue
    
    st.session_state.model_available = False
    return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model"""
    try:
        img = image.resize(target_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Image preprocessing error: {e}")
        return None

def detect_patches_advanced(image):
    """Detect bald patches using advanced image processing"""
    try:
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        _, thresh1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
        combined = cv2.bitwise_and(thresh1, thresh2)
        
        kernel_small = np.ones((3,3), np.uint8)
        kernel_large = np.ones((7,7), np.uint8)
        morphed = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small, iterations=2)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_patches = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 500 < area < 40000:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    
                    if circularity > 0.4 and solidity > 0.7:
                        valid_patches.append(cnt)
        
        return len(valid_patches), morphed
    except Exception as e:
        st.error(f"Patch detection error: {e}")
        return 0, None

def calculate_follicle_density_advanced(image):
    """Calculate follicle density using edge detection"""
    try:
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges_canny = cv2.Canny(blurred, 30, 100)
        
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        edges_sobel = np.sqrt(sobelx**2 + sobely**2)
        edges_sobel = np.uint8(edges_sobel / (edges_sobel.max() + 1e-8) * 255)
        
        combined_edges = cv2.bitwise_or(edges_canny, edges_sobel)
        follicle_count = np.sum(combined_edges > 0)
        
        image_area_pixels = gray.shape[0] * gray.shape[1]
        density = (follicle_count / image_area_pixels) * 100 * 0.5
        density = max(15, min(85, density))
        
        return round(density, 1)
    except Exception as e:
        st.error(f"Follicle density error: {e}")
        return 50.0

def classify_hair_type(image):
    """Classify hair as bald or notbald using the model"""
    if st.session_state.hair_classifier_model is not None:
        try:
            processed_img = preprocess_image(image)
            if processed_img is None:
                return "notbald", 0.5
            
            prediction = st.session_state.hair_classifier_model.predict(processed_img, verbose=0)[0]
            
            if len(prediction) == 2:
                bald_prob = float(prediction[0])
                notbald_prob = float(prediction[1])
                return ("bald", bald_prob) if bald_prob > notbald_prob else ("notbald", notbald_prob)
            else:
                confidence = float(prediction[0])
                return ("bald", confidence) if confidence > 0.5 else ("notbald", 1 - confidence)
        except Exception as e:
            st.error(f"Classification error: {e}")
            return "notbald", 0.5
    return "notbald", 0.5

def calculate_salt_score_enhanced(hair_loss_pct, region, follicle_density):
    """Calculate SALT score with region weighting"""
    try:
        base_score = hair_loss_pct * 0.35
        region_weights = {
            'Vertex': 0.40, 'Frontal': 0.18, 'R. Temporal': 0.08,
            'L. Temporal': 0.08, 'R. Parietal': 0.08, 'L. Parietal': 0.08, 'Occipital': 0.10
        }
        weight = region_weights.get(region, 0.1)
        weighted_score = base_score * (1 + weight * 2)
        
        normal_density = 50
        density_factor = max(0, (normal_density - follicle_density) / normal_density * 15)
        
        total_score = weighted_score + density_factor
        return max(0, min(100, round(total_score, 1)))
    except Exception as e:
        st.error(f"SALT score error: {e}")
        return 0.0

def analyze_image_comprehensive(image, region):
    """Perform comprehensive image analysis"""
    try:
        hair_type, confidence = classify_hair_type(image)
        num_patches, _ = detect_patches_advanced(image)
        follicle_density = calculate_follicle_density_advanced(image)
        
        is_bald = (hair_type.lower() == "bald")
        
        if is_bald:
            hair_loss_pct = min(confidence * 100 + num_patches * 2, 95)
            salt_score = calculate_salt_score_enhanced(hair_loss_pct, region, follicle_density)
            
            if salt_score < 25:
                severity, severity_class, desc = "Mild Severity", "severity-mild", "S1 (0-25%)"
            elif salt_score < 50:
                severity, severity_class, desc = "Moderate Severity", "severity-moderate", "S2 (25-50%)"
            elif salt_score < 75:
                severity, severity_class, desc = "Severe", "severity-severe", "S3 (50-75%)"
            else:
                severity, severity_class, desc = "Very Severe", "severity-severe", "S4 (75-100%)"
            
            return {
                'hair_type': 'Bald (Alopecia Detected)',
                'confidence': round(confidence * 100, 1),
                'salt_score': salt_score,
                'hair_loss_pct': round(hair_loss_pct, 1),
                'follicle_density': follicle_density,
                'region': region,
                'severity': severity,
                'severity_class': severity_class,
                'severity_desc': desc,
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

def get_strength_category(value):
    """Determine hair strength category"""
    if value < 601:
        return "Weak", "strength-weak"
    elif value < 1201:
        return "Normal", "strength-normal"
    else:
        return "Strong", "strength-strong"

def get_recommendations(strength_value):
    """Get personalized recommendations based on hair strength"""
    if strength_value < 601:
        return {
            'status': 'Critical Attention Required',
            'description': 'Hair strength is in the weak range and requires immediate care.',
            'recommendations': [
                'Consult a dermatologist immediately for professional assessment',
                'Consider biotin and vitamin D supplements after medical consultation',
                'Avoid harsh chemical treatments and excessive heat styling',
                'Use protein-enriched deep conditioning treatments weekly',
                'Reduce stress through meditation, yoga, or other relaxation techniques',
                'Improve diet with iron-rich foods, zinc, and omega-3 fatty acids',
                'Get adequate sleep (7-9 hours) to support hair health',
                'Avoid tight hairstyles that cause tension on hair follicles'
            ]
        }
    elif strength_value < 1201:
        return {
            'status': 'Normal Hair Strength',
            'description': 'Hair strength is within normal range. Maintain current care routine.',
            'recommendations': [
                'Continue regular hair care routine with gentle products',
                'Maintain balanced diet rich in proteins, vitamins, and minerals',
                'Use deep conditioning treatments bi-weekly',
                'Minimize heat styling and use heat protectant when necessary',
                'Get regular trims every 6-8 weeks to prevent split ends',
                'Manage stress levels through healthy lifestyle choices',
                'Stay hydrated by drinking adequate water daily',
                'Protect hair from environmental damage (sun, pollution)'
            ]
        }
    else:
        return {
            'status': 'Excellent Hair Strength',
            'description': 'Hair strength is in the strong range. Keep up the great work!',
            'recommendations': [
                'Continue your current successful hair care routine',
                'Maintain healthy lifestyle and balanced nutrition',
                'Stay consistent with beneficial hair care habits',
                'Schedule annual check-ups to monitor hair health',
                'Protect hair from environmental stressors when possible',
                'Share your routine to help others improve their hair health'
            ]
        }

def generate_detailed_analysis(overall_score, alopecia_component, strength_score, analysis_results):
    """Generate detailed analysis based on scores"""
    analysis = []
    
    # Overall Health Assessment
    if overall_score >= 85:
        analysis.append({
            'category': 'Overall Health Status',
            'score': overall_score,
            'grade': 'A+',
            'assessment': 'Excellent',
            'details': 'Your hair health is in excellent condition. All indicators show optimal hair health with strong follicles and minimal hair loss. Continue your current hair care routine and lifestyle habits.'
        })
    elif overall_score >= 75:
        analysis.append({
            'category': 'Overall Health Status',
            'score': overall_score,
            'grade': 'A',
            'assessment': 'Very Good',
            'details': 'Your hair health is very good with minor areas that could benefit from attention. Most metrics are within healthy ranges. Minor improvements in care routine could elevate your hair health to excellent status.'
        })
    elif overall_score >= 65:
        analysis.append({
            'category': 'Overall Health Status',
            'score': overall_score,
            'grade': 'B+',
            'assessment': 'Good',
            'details': 'Your hair health is good but shows some areas requiring attention. Consider enhancing your hair care routine with targeted treatments and consulting with a professional for personalized advice.'
        })
    elif overall_score >= 55:
        analysis.append({
            'category': 'Overall Health Status',
            'score': overall_score,
            'grade': 'B',
            'assessment': 'Satisfactory',
            'details': 'Your hair health is satisfactory but has room for improvement. Several metrics indicate the need for enhanced care. Consider professional consultation to address specific concerns.'
        })
    elif overall_score >= 45:
        analysis.append({
            'category': 'Overall Health Status',
            'score': overall_score,
            'grade': 'C',
            'assessment': 'Fair',
            'details': 'Your hair health needs attention. Multiple indicators show suboptimal conditions. Professional consultation is recommended to develop a comprehensive treatment plan.'
        })
    elif overall_score >= 35:
        analysis.append({
            'category': 'Overall Health Status',
            'score': overall_score,
            'grade': 'D',
            'assessment': 'Poor',
            'details': 'Your hair health requires immediate attention. Several critical indicators are below healthy thresholds. Urgent professional consultation with a dermatologist is strongly recommended.'
        })
    else:
        analysis.append({
            'category': 'Overall Health Status',
            'score': overall_score,
            'grade': 'F',
            'assessment': 'Critical',
            'details': 'Your hair health is in critical condition requiring immediate medical intervention. Do not delay in seeking professional help from a board-certified dermatologist or trichologist.'
        })
    
    # Alopecia Analysis Component
    if analysis_results:
        if alopecia_component >= 85:
            analysis.append({
                'category': 'Alopecia Assessment',
                'score': alopecia_component,
                'grade': 'A+',
                'assessment': 'Excellent - No Significant Hair Loss',
                'details': f"Follicle density is {analysis_results['follicle_density']}/cm² which is optimal. No signs of alopecia detected. Hair coverage is healthy and uniform across the analyzed region."
            })
        elif alopecia_component >= 70:
            analysis.append({
                'category': 'Alopecia Assessment',
                'score': alopecia_component,
                'grade': 'B+',
                'assessment': 'Good - Minimal Hair Thinning',
                'details': f"Follicle density is {analysis_results['follicle_density']}/cm². Minor hair thinning detected but within acceptable range. Monitor for changes and maintain preventive care."
            })
        elif alopecia_component >= 55:
            analysis.append({
                'category': 'Alopecia Assessment',
                'score': alopecia_component,
                'grade': 'C+',
                'assessment': 'Moderate - Early Stage Hair Loss',
                'details': f"Follicle density is {analysis_results['follicle_density']}/cm². Early stage hair loss detected. Consider early intervention treatments and professional consultation."
            })
        elif alopecia_component >= 40:
            analysis.append({
                'category': 'Alopecia Assessment',
                'score': alopecia_component,
                'grade': 'D',
                'assessment': 'Concerning - Moderate Hair Loss',
                'details': f"Follicle density is {analysis_results['follicle_density']}/cm². Moderate hair loss detected. Professional treatment is recommended to prevent progression."
            })
        else:
            analysis.append({
                'category': 'Alopecia Assessment',
                'score': alopecia_component,
                'grade': 'F',
                'assessment': 'Severe - Advanced Hair Loss',
                'details': f"Follicle density is {analysis_results['follicle_density']}/cm². Severe hair loss detected. Immediate medical intervention required."
            })
    
    # Hair Strength Analysis Component
    avg_strength = np.mean(list(st.session_state.hair_strength.values()))
    
    if strength_score >= 85:
        analysis.append({
            'category': 'Hair Strength Assessment',
            'score': strength_score,
            'grade': 'A+',
            'assessment': 'Excellent - Strong Hair',
            'details': f'Average hair force is {round(avg_strength, 1)} N, indicating excellent hair strength. Hair follicles are robust and resilient. Protein structure is optimal with minimal breakage risk.'
        })
    elif strength_score >= 70:
        analysis.append({
            'category': 'Hair Strength Assessment',
            'score': strength_score,
            'grade': 'B+',
            'assessment': 'Good - Above Average Strength',
            'details': f'Average hair force is {round(avg_strength, 1)} N, showing good hair strength. Hair is healthy but could benefit from protein treatments to reach optimal strength levels.'
        })
    elif strength_score >= 55:
        analysis.append({
            'category': 'Hair Strength Assessment',
            'score': strength_score,
            'grade': 'C',
            'assessment': 'Fair - Normal Strength',
            'details': f'Average hair force is {round(avg_strength, 1)} N, indicating normal but not optimal strength. Consider protein-rich treatments to strengthen hair.'
        })
    elif strength_score >= 40:
        analysis.append({
            'category': 'Hair Strength Assessment',
            'score': strength_score,
            'grade': 'D',
            'assessment': 'Poor - Weak Hair',
            'details': f'Average hair force is {round(avg_strength, 1)} N, showing weak hair structure. Hair is prone to breakage and damage. Urgent need for strengthening treatments.'
        })
    else:
        analysis.append({
            'category': 'Hair Strength Assessment',
            'score': strength_score,
            'grade': 'F',
            'assessment': 'Critical - Very Weak Hair',
            'details': f'Average hair force is {round(avg_strength, 1)} N, indicating critically weak hair. Severe risk of breakage and damage. Immediate intervention required including deep conditioning treatments, avoiding all heat and chemicals, and dermatologist consultation.'
        })
    
    # Regional Variance Analysis
    variance = np.std(list(st.session_state.hair_strength.values()))
    if variance < 100:
        analysis.append({
            'category': 'Regional Hair Uniformity',
            'score': 95,
            'grade': 'A+',
            'assessment': 'Excellent - Uniform Strength',
            'details': f'Hair strength variance is {round(variance, 1)} N across regions, indicating excellent uniformity. All scalp regions show consistent hair health with minimal variation.'
        })
    elif variance < 200:
        analysis.append({
            'category': 'Regional Hair Uniformity',
            'score': 80,
            'grade': 'B+',
            'assessment': 'Good - Minor Variations',
            'details': f'Hair strength variance is {round(variance, 1)} N, showing minor regional differences. This is within normal range but monitor weaker regions for any decline.'
        })
    elif variance < 300:
        analysis.append({
            'category': 'Regional Hair Uniformity',
            'score': 65,
            'grade': 'C',
            'assessment': 'Moderate - Notable Variations',
            'details': f'Hair strength variance is {round(variance, 1)} N, indicating notable regional differences. Some areas are significantly weaker and may require targeted treatment.'
        })
    else:
        analysis.append({
            'category': 'Regional Hair Uniformity',
            'score': 40,
            'grade': 'D',
            'assessment': 'Poor - High Variations',
            'details': f'Hair strength variance is {round(variance, 1)} N, showing significant regional disparities. Some areas are critically weak compared to others. Focused treatment on weak regions is essential.'
        })
    
    return analysis

# Load default model
load_hair_classifier_model()

# Header
st.markdown("""
<div class="header-container">
    <h1 class="main-title">Hair Health & Alopecia Detection System</h1>
    <p class="subtitle">AI-Powered Analysis | Hair Strength Assessment | Alopecia Detection | SALT Scoring</p>
</div>
""", unsafe_allow_html=True)

# Navigation
col_nav1, col_nav2, col_nav3, col_nav4 = st.columns(4)

with col_nav1:
    if st.button("Setup", use_container_width=True):
        st.session_state.current_tab = 'setup'
        st.rerun()

with col_nav2:
    if st.button("Alopecia Analysis", use_container_width=True):
        st.session_state.current_tab = 'alopecia'
        st.rerun()

with col_nav3:
    if st.button("Hair Strength", use_container_width=True):
        st.session_state.current_tab = 'strength'
        st.rerun()

with col_nav4:
    if st.button("Report", use_container_width=True):
        st.session_state.current_tab = 'report'
        st.rerun()

# SETUP TAB
if st.session_state.current_tab == 'setup':
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card"><div class="card-title">Model Setup</div>', unsafe_allow_html=True)
        
        if st.session_state.model_available:
            st.markdown('<div class="success-box"><strong>Status:</strong> AI model is active and ready for analysis</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box"><strong>Warning:</strong> No model loaded. Upload a model file to enable AI-powered analysis.</div>', unsafe_allow_html=True)
        
        st.markdown("<h4 style='color: #0f766e; margin-top: 20px;'>Upload Custom Model</h4>", unsafe_allow_html=True)
        st.markdown("<p style='color: #64748b; font-size: 0.9em;'>Supported formats: .keras, .h5, .pb</p>", unsafe_allow_html=True)
        
        uploaded_model = st.file_uploader("Choose model file", type=['keras', 'h5', 'pb'], label_visibility="collapsed")
        
        if uploaded_model is not None:
            try:
                with st.spinner("Loading model..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp:
                        tmp.write(uploaded_model.getbuffer())
                        tmp_path = tmp.name
                    custom_model = keras.models.load_model(tmp_path)
                    st.session_state.hair_classifier_model = custom_model
                    st.session_state.model_available = True
                    os.remove(tmp_path)
                st.success("Model loaded successfully")
                st.rerun()
            except Exception as e:
                st.error(f"Model loading failed: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card"><div class="card-title">Quick Guide</div>', unsafe_allow_html=True)
        st.markdown("""
        **System Features:**
        - AI-powered alopecia detection with deep learning models
        - SALT score calculation for standardized assessment
        - Hair strength measurement across multiple scalp regions
        - Follicle density analysis using computer vision
        - Comprehensive health reports and recommendations
        
        **How to Use:**
        1. **Setup:** Configure your AI model (optional)
        2. **Alopecia Analysis:** Upload scalp images for detection
        3. **Hair Strength:** Input force measurements by region
        4. **Report:** View integrated analysis and download results
        
        **Recommended Workflow:**
        - Start with model setup if using custom AI model
        - Capture clear, well-lit scalp images
        - Analyze multiple regions for complete assessment
        - Review recommendations and consult professionals
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# ALOPECIA ANALYSIS TAB
elif st.session_state.current_tab == 'alopecia':
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card"><div class="card-title">Image Analysis</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload scalp image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
        
        if uploaded_file:
            st.session_state.uploaded_image = Image.open(uploaded_file).convert('RGB')
            st.image(st.session_state.uploaded_image, use_container_width=True, caption="Uploaded Scalp Image")
        else:
            st.markdown('<div class="info-box">Please upload a clear scalp image for analysis. Supported formats: JPG, JPEG, PNG</div>', unsafe_allow_html=True)
        
        st.markdown('<h4 style="color: #0f766e; margin-top: 20px;">Select Scalp Region</h4>', unsafe_allow_html=True)
        st.markdown('<p style="color: #64748b; font-size: 0.9em; margin-bottom: 15px;">Choose the region that matches your uploaded image</p>', unsafe_allow_html=True)
        
        regions = ['Vertex', 'Frontal', 'R. Temporal', 'L. Temporal', 'R. Parietal', 'L. Parietal', 'Occipital']
        cols = st.columns(3)
        for idx, region in enumerate(regions[:-1]):
            with cols[idx % 3]:
                if st.button(region, use_container_width=True, key=f"region_{region}"):
                    st.session_state.selected_region = region
        with st.columns(3)[0]:
            if st.button(regions[-1], use_container_width=True, key=f"region_{regions[-1]}"):
                st.session_state.selected_region = regions[-1]
        
        st.markdown(f'<div style="text-align: center; color: #0f766e; font-weight: 600; margin-top: 15px; padding: 12px; background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%); border-radius: 10px;">Selected Region: {st.session_state.selected_region}</div>', unsafe_allow_html=True)
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button('Analyze Image', use_container_width=True, key="analyze_btn"):
                if st.session_state.uploaded_image:
                    with st.spinner('Analyzing image... Please wait'):
                        results = analyze_image_comprehensive(st.session_state.uploaded_image, st.session_state.selected_region)
                        if results:
                            st.session_state.analysis_results = results
                            st.success('Analysis complete')
                            st.rerun()
                else:
                    st.error('Please upload an image first')
        with col_btn2:
            if st.button('Clear All', use_container_width=True, key="clear_btn"):
                st.session_state.uploaded_image = None
                st.session_state.analysis_results = None
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card"><div class="card-title">Analysis Results</div>', unsafe_allow_html=True)
        
        if st.session_state.analysis_results:
            r = st.session_state.analysis_results
            
            color = "#fee2e2" if r['is_bald'] else "#d1fae5"
            bord = "#ef4444" if r['is_bald'] else "#10b981"
            txt = "#991b1b" if r['is_bald'] else "#065f46"
            
            st.markdown(f'<div style="background: {color}; padding: 20px; border-radius: 12px; border: 2px solid {bord}; text-align: center; margin-bottom: 20px;"><div style="font-size: 1.6em; font-weight: 700; color: {txt};">{r["hair_type"]}</div></div>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="result-item"><span class="result-label">Confidence Level</span><span class="result-value">{r["confidence"]}%</span></div>', unsafe_allow_html=True)
            
            if r['is_bald'] and r['show_score']:
                st.markdown(f'<div class="result-item"><span class="result-label">SALT Score</span><span class="result-value">{r["salt_score"]}</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="result-item"><span class="result-label">Hair Loss Percentage</span><span class="result-value">{r["hair_loss_pct"]}%</span></div>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="result-item"><span class="result-label">Follicle Density</span><span class="result-value">{r["follicle_density"]}/cm²</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-item"><span class="result-label">Region Analyzed</span><span class="result-value">{r["region"].upper()}</span></div>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="severity-badge {r["severity_class"]}">{r["severity"]}<br><small style="font-size: 0.7em; opacity: 0.8;">{r["severity_desc"]}</small></div>', unsafe_allow_html=True)
            
            if r['is_bald']:
                st.markdown('<div class="warning-box"><strong>Important:</strong> This analysis is for informational purposes only. Please consult a dermatologist for professional diagnosis and treatment.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">No analysis results available yet. Upload an image and click "Analyze Image" to begin.</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# HAIR STRENGTH TAB
elif st.session_state.current_tab == 'strength':
    st.markdown('<div class="card"><div class="card-title">Hair Strength Assessment (100-1800 N)</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748b; margin-bottom: 20px;">Enter force measurements in Newtons (N) for each scalp region</p>', unsafe_allow_html=True)
    
    regions = ['Vertex', 'Frontal', 'R. Temporal', 'L. Temporal', 'R. Parietal', 'L. Parietal', 'Occipital']
    
    col1, col2 = st.columns(2)
    for idx, region in enumerate(regions):
        with col1 if idx < 4 else col2:
            val = st.number_input(f"{region} Region (N)", min_value=100, max_value=1800, 
                                 value=st.session_state.hair_strength[region], step=10, key=f"str_{region}")
            st.session_state.hair_strength[region] = val
    
    avg = np.mean(list(st.session_state.hair_strength.values()))
    cat, css = get_strength_category(avg)
    
    st.markdown(f"""
    <div class="average-strength">
        <div style="color: #0c4a6e; font-weight: 700; font-size: 1.2em;">Average Hair Force</div>
        <div class="average-strength-value">{round(avg, 1)} N</div>
        <div style="color: #0c4a6e; font-weight: 600; margin-top: 10px; font-size: 1.1em;">Status: {cat}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h4 style="color: #0f766e; margin-top: 30px; margin-bottom: 15px;">Force Measurements by Region</h4>', unsafe_allow_html=True)
    
    for region in regions:
        strength_val = st.session_state.hair_strength[region]
        category, css_class = get_strength_category(strength_val)
        st.markdown(f"""
        <div class="hair-strength-display">
            <span style="font-weight: 600; color: #0f766e;">{region}</span>
            <span class="strength-value {css_class}">{strength_val} N - {category}</span>
        </div>
        """, unsafe_allow_html=True)
    
    rec = get_recommendations(avg)
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); padding: 25px; border-radius: 12px; margin-top: 30px; border-left: 5px solid #10b981;">
        <h3 style="color: #065f46; margin-top: 0;">{rec['status']}</h3>
        <p style="color: #047857; font-size: 1.05em; margin: 15px 0;">{rec['description']}</p>
        <h4 style="color: #065f46; margin-top: 20px;">Personalized Recommendations:</h4>
        <ul style="color: #047857; margin-top: 10px; line-height: 1.8;">
    """, unsafe_allow_html=True)
    
    for rec_item in rec['recommendations']:
        st.markdown(f"<li>{rec_item}</li>", unsafe_allow_html=True)
    
    st.markdown("</ul></div>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# COMBINED REPORT TAB
elif st.session_state.current_tab == 'report':
    st.markdown('<div class="card"><div class="card-title">Comprehensive Hair Health Report</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div style="background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%); padding: 20px; border-radius: 12px; border-left: 5px solid #14b8a6;">', unsafe_allow_html=True)
        st.subheader("Alopecia Analysis Summary")
        if st.session_state.analysis_results:
            r = st.session_state.analysis_results
            st.write(f"**Status:** {r['hair_type']}")
            st.write(f"**Confidence:** {r['confidence']}%")
            st.write(f"**Region:** {r['region']}")
            st.write(f"**Follicle Density:** {r['follicle_density']}/cm²")
            if r['is_bald']:
                st.write(f"**SALT Score:** {r['salt_score']}")
                st.write(f"**Hair Loss:** {r['hair_loss_pct']}%")
                st.write(f"**Severity:** {r['severity']}")
        else:
            st.info("No alopecia analysis available. Please complete analysis in the Alopecia Analysis tab.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 20px; border-radius: 12px; border-left: 5px solid #f59e0b;">', unsafe_allow_html=True)
        st.subheader("Hair Strength Summary")
        avg_str = np.mean(list(st.session_state.hair_strength.values()))
        cat_str, _ = get_strength_category(avg_str)
        st.write(f"**Average Force:** {round(avg_str, 1)} N")
        st.write(f"**Overall Status:** {cat_str}")
        st.write(f"**Strongest Region:** {max(st.session_state.hair_strength, key=st.session_state.hair_strength.get)} ({max(st.session_state.hair_strength.values())} N)")
        st.write(f"**Weakest Region:** {min(st.session_state.hair_strength, key=st.session_state.hair_strength.get)} ({min(st.session_state.hair_strength.values())} N)")
        
        variance = np.std(list(st.session_state.hair_strength.values()))
        st.write(f"**Strength Variance:** {round(variance, 1)} N")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Calculate overall score
    overall_score = 0
    score_components = []
    
    # Alopecia component (50% weight)
    if st.session_state.analysis_results:
        r = st.session_state.analysis_results
        if r['is_bald']:
            alopecia_score = max(0, 100 - r['salt_score'])
        else:
            alopecia_score = 95
        
        density_score = min(100, (r['follicle_density'] / 80) * 100)
        alopecia_component = (alopecia_score * 0.7 + density_score * 0.3)
        score_components.append(('Alopecia Analysis', alopecia_component, 50))
        overall_score += alopecia_component * 0.5
    else:
        score_components.append(('Alopecia Analysis', 0, 50))
        alopecia_component = 0
    
    # Hair Strength component (50% weight)
    avg_str = np.mean(list(st.session_state.hair_strength.values()))
    strength_score = min(100, max(0, ((avg_str - 100) / (1800 - 100)) * 100))
    score_components.append(('Hair Strength', strength_score, 50))
    overall_score += strength_score * 0.5
    
    # Overall Health Score Display
    st.markdown('<h3 style="color: #0f766e; margin: 30px 0 20px 0; text-align: center;">Overall Hair Health Score</h3>', unsafe_allow_html=True)
    
    if overall_score >= 80:
        score_color = "#10b981"
        score_bg = "linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%)"
        score_status = "Excellent"
        score_desc = "Your hair health is in excellent condition"
    elif overall_score >= 60:
        score_color = "#f59e0b"
        score_bg = "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)"
        score_status = "Good"
        score_desc = "Your hair health is good with room for improvement"
    elif overall_score >= 40:
        score_color = "#f97316"
        score_bg = "linear-gradient(135deg, #fed7aa 0%, #fdba74 100%)"
        score_status = "Fair"
        score_desc = "Your hair health needs attention and care"
    else:
        score_color = "#ef4444"
        score_bg = "linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)"
        score_status = "Poor"
        score_desc = "Your hair health requires immediate professional attention"
    
    st.markdown(f"""
    <div style="background: {score_bg}; padding: 40px; border-radius: 20px; text-align: center; border: 3px solid {score_color}; box-shadow: 0 10px 30px rgba(0,0,0,0.15);">
        <div style="font-size: 1.2em; color: {score_color}; font-weight: 600; margin-bottom: 10px;">Overall Hair Health Score</div>
        <div class="score-number" style="color: {score_color};">{round(overall_score)}<span style="font-size: 0.5em;">/100</span></div>
        <div style="font-size: 1.5em; font-weight: 700; color: {score_color}; margin: 15px 0;">{score_status}</div>
        <div style="font-size: 1.1em; color: {score_color}; opacity: 0.9; margin-top: 10px;">{score_desc}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Score breakdown
    st.markdown('<h4 style="color: #0f766e; margin: 25px 0 15px 0;">Score Breakdown</h4>', unsafe_allow_html=True)
    
    col_break1, col_break2 = st.columns(2)
    
    with col_break1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%); padding: 25px; border-radius: 12px; border-left: 5px solid #14b8a6;">
            <div style="font-weight: 600; color: #0f766e; font-size: 1.1em; margin-bottom: 10px;">Alopecia Analysis</div>
            <div style="font-size: 2.5em; font-weight: 700; color: #0d9488; margin: 10px 0;">{round(score_components[0][1])}<span style="font-size: 0.5em;">/100</span></div>
            <div style="color: #64748b; font-size: 0.9em;">Weight: {score_components[0][2]}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_break2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 25px; border-radius: 12px; border-left: 5px solid #f59e0b;">
            <div style="font-weight: 600; color: #92400e; font-size: 1.1em; margin-bottom: 10px;">Hair Strength</div>
            <div style="font-size: 2.5em; font-weight: 700; color: #d97706; margin: 10px 0;">{round(score_components[1][1])}<span style="font-size: 0.5em;">/100</span></div>
            <div style="color: #64748b; font-size: 0.9em;">Weight: {score_components[1][2]}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Progress visualization
    st.markdown('<h4 style="color: #0f766e; margin: 25px 0 15px 0;">Visual Progress</h4>', unsafe_allow_html=True)
    
    progress_html = f"""
    <div style="background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <div style="margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="font-weight: 600; color: #0f766e;">Alopecia Analysis</span>
                <span style="font-weight: 700; color: #0d9488;">{round(score_components[0][1])}%</span>
            </div>
            <div style="background: #e5e7eb; height: 30px; border-radius: 15px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, #14b8a6 0%, #0d9488 100%); height: 100%; width: {score_components[0][1]}%; transition: width 0.5s ease;"></div>
            </div>
        </div>
        <div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="font-weight: 600; color: #92400e;">Hair Strength</span>
                <span style="font-weight: 700; color: #d97706;">{round(score_components[1][1])}%</span>
            </div>
            <div style="background: #e5e7eb; height: 30px; border-radius: 15px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%); height: 100%; width: {score_components[1][1]}%; transition: width 0.5s ease;"></div>
            </div>
        </div>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # DETAILED ANALYSIS SECTION
    st.markdown('<h3 style="color: #0f766e; margin: 30px 0 20px 0; text-align: center;">Detailed Analysis Report</h3>', unsafe_allow_html=True)
    
    detailed_analysis = generate_detailed_analysis(overall_score, alopecia_component, strength_score, st.session_state.analysis_results)
    
    for item in detailed_analysis:
        # Color coding based on grade
        if item['grade'] in ['A+', 'A']:
            card_bg = "linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%)"
            border_color = "#10b981"
            text_color = "#065f46"
        elif item['grade'] in ['B+', 'B']:
            card_bg = "linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%)"
            border_color = "#3b82f6"
            text_color = "#1e40af"
        elif item['grade'] in ['C+', 'C']:
            card_bg = "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)"
            border_color = "#f59e0b"
            text_color = "#92400e"
        elif item['grade'] == 'D':
            card_bg = "linear-gradient(135deg, #fed7aa 0%, #fdba74 100%)"
            border_color = "#f97316"
            text_color = "#9a3412"
        else:  # F
            card_bg = "linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)"
            border_color = "#ef4444"
            text_color = "#991b1b"
        
        st.markdown(f"""
        <div style="background: {card_bg}; padding: 25px; border-radius: 16px; margin: 20px 0; border-left: 6px solid {border_color}; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h4 style="color: {text_color}; margin: 0; font-size: 1.3em;">{item['category']}</h4>
                <div style="display: flex; gap: 15px; align-items: center;">
                    <span class="grade-badge" style="color: {text_color};">Grade: {item['grade']}</span>
                    <span class="grade-badge" style="color: {text_color};">{round(item['score'])}/100</span>
                </div>
            </div>
            <div style="background: white; padding: 15px; border-radius: 10px; margin-bottom: 12px;">
                <p style="margin: 0; font-weight: 600; color: {text_color}; font-size: 1.1em;">{item['assessment']}</p>
            </div>
            <p style="color: {text_color}; margin: 0; line-height: 1.7; font-size: 1em;">{item['details']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Export options
    st.markdown('<h4 style="color: #0f766e; margin: 25px 0 15px 0;">Export Report</h4>', unsafe_allow_html=True)
    
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        if st.button("Download PDF Report", use_container_width=True):
            st.info("PDF export feature coming soon")
    
    with col_export2:
        if st.button("Download CSV Data", use_container_width=True):
            # Create CSV data
            export_data = {
                'Metric': [
                    'Overall Score',
                    'Overall Status',
                    'Alopecia Score',
                    'Hair Strength Score',
                    'Average Hair Force (N)',
                    'Hair Type',
                    'Follicle Density',
                    'SALT Score',
                    'Hair Loss %',
                    'Analysis Date'
                ],
                'Value': [
                    round(overall_score),
                    score_status,
                    round(score_components[0][1]),
                    round(score_components[1][1]),
                    round(avg_str, 1),
                    st.session_state.analysis_results['hair_type'] if st.session_state.analysis_results else 'N/A',
                    st.session_state.analysis_results['follicle_density'] if st.session_state.analysis_results else 'N/A',
                    st.session_state.analysis_results['salt_score'] if st.session_state.analysis_results and st.session_state.analysis_results['salt_score'] else 'N/A',
                    st.session_state.analysis_results['hair_loss_pct'] if st.session_state.analysis_results and st.session_state.analysis_results['hair_loss_pct'] else 'N/A',
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ]
            }
            
            df_export = pd.DataFrame(export_data)
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"hair_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<style>
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)
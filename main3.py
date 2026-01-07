import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Hair Strength Assessment",
    page_icon="💪",
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
        background: linear-gradient(120deg, #0f766e 0%, #14b8a6 50%, #2dd4bf 100%);
        padding: 0;
    }
    
    .stApp {
        background: transparent;
    }
    
    /* Header Styling */
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
    
    /* Card Styling */
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
    
    /* Input Container */
    .input-container {
        background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%);
        padding: 25px;
        border-radius: 16px;
        margin: 20px 0;
        border: 2px solid #14b8a6;
    }
    
    .input-label {
        font-weight: 700;
        color: #0f766e;
        margin-bottom: 20px;
        font-size: 1.1em;
    }
    
    /* Hair Strength Display */
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
        transition: all 0.3s ease;
    }
    
    .hair-strength-display:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 16px rgba(20, 184, 166, 0.2);
    }
    
    .strength-region-name {
        font-weight: 600;
        color: #0f766e;
        font-size: 1.05em;
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
    
    /* Average Strength Display */
    .average-strength {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        padding: 25px;
        border-radius: 16px;
        margin: 20px 0;
        border: 3px solid #0284c7;
        text-align: center;
    }
    
    .average-strength-label {
        color: #0c4a6e;
        font-weight: 700;
        font-size: 1.1em;
    }
    
    .average-strength-value {
        color: #0284c7;
        font-weight: 700;
        font-size: 3em;
        margin-top: 12px;
    }
    
    .average-strength-category {
        color: #0c4a6e;
        font-weight: 600;
        font-size: 1.05em;
        margin-top: 10px;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 20px 40px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 1.2em;
        margin: 25px 0;
        text-align: center;
        width: 100%;
        letter-spacing: 0.5px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        animation: fadeInUp 0.6s ease;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .status-weak {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
        border: 2px solid #dc2626;
    }
    
    .status-normal {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
        border: 2px solid #059669;
    }
    
    .status-strong {
        background: linear-gradient(135deg, #ccfbf1 0%, #99f6e4 100%);
        color: #0f766e;
        border: 2px solid #14b8a6;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #f0f4f8 0%, #d9e8f5 100%);
        border-left: 5px solid #0284c7;
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
        color: #0c4a6e;
        font-size: 0.95em;
        line-height: 1.8;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 5px solid #d97706;
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
        color: #92400e;
        font-size: 0.95em;
        line-height: 1.8;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 5px solid #059669;
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
        color: #065f46;
        font-size: 0.95em;
        line-height: 1.8;
    }
    
    /* Button Styling */
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
        letter-spacing: 0.3px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(20, 184, 166, 0.4);
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: white;
        padding: 30px 20px;
        font-size: 0.9em;
        margin-top: 40px;
    }
    
    .footer-title {
        font-size: 1.1em;
        font-weight: 600;
        margin-bottom: 10px;
        opacity: 0.95;
    }
    
    .footer-text {
        opacity: 0.8;
        line-height: 1.8;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'hair_strength' not in st.session_state:
    st.session_state.hair_strength = {
        'Vertex': 900,
        'Frontal': 900,
        'R. Temporal': 900,
        'L. Temporal': 900,
        'R. Parietal': 900,
        'L. Parietal': 900,
        'Occipital': 900
    }

# Helper function to get strength category (Updated scale: 100-1800 N)
def get_strength_category(value):
    """Determine strength category based on value (Newton force)"""
    if value < 601:
        return "Weak", "strength-weak", "status-weak"
    elif value < 1201:
        return "Normal", "strength-normal", "status-normal"
    else:
        return "Strong", "strength-strong", "status-strong"

# Get recommendations based on strength
def get_recommendations(strength_value):
    """Get health recommendations based on strength value"""
    category, _, _ = get_strength_category(strength_value)
    
    if strength_value < 601:
        return {
            'status': 'Critical Attention Required',
            'description': 'Hair strength is in the weak range and requires immediate care and professional guidance.',
            'recommendations': [
                'Consult a trichologist or dermatologist immediately',
                'Consider biotin (10,000 mcg), iron, and vitamin D supplements',
                'Avoid harsh treatments, heat styling, and tight hairstyles',
                'Use protein-enriched hair treatments weekly',
                'Reduce stress through meditation, yoga, or counseling',
                'Improve diet with iron-rich foods (spinach, red meat), zinc, and amino acids',
                'Use gentle, sulfate-free shampoos and conditioners',
                'Get thyroid function tested'
            ],
            'box_type': 'warning-box'
        }
    elif strength_value < 1201:
        return {
            'status': 'Normal Hair Strength',
            'description': 'Hair strength is within normal range. Continue with good hair care practices.',
            'recommendations': [
                'Maintain regular hair care routine',
                'Continue balanced diet rich in proteins and vitamins',
                'Use deep conditioning treatments bi-weekly',
                'Minimize heat styling or use heat protectants',
                'Protect hair from UV damage with hats or UV-protectant sprays',
                'Get regular trims every 6-8 weeks',
                'Manage stress levels through healthy activities',
                'Monitor hair health with periodic self-assessments'
            ],
            'box_type': 'success-box'
        }
    else:
        return {
            'status': 'Excellent Hair Strength',
            'description': 'Hair strength is in the strong range. Maintain your excellent hair care practices.',
            'recommendations': [
                'Continue your current care routine',
                'Maintain healthy lifestyle and nutrition',
                'Use moisturizing treatments monthly',
                'Consider professional treatments for enhancement',
                'Protect from environmental damage (pollution, chlorine)',
                'Stay consistent with healthy habits',
                'Annual dermatology or trichology check-ups',
                'Share your routine to help others'
            ],
            'box_type': 'success-box'
        }

# Header
st.markdown("""
<div class="header-container">
    <h1 class="main-title">Hair Strength Assessment System</h1>
    <p class="subtitle">Comprehensive Hair Force Evaluation Across All Scalp Regions (100-1800 N)</p>
</div>
""", unsafe_allow_html=True)

# Main content
st.markdown('<div class="card"><div class="card-title">Hair Strength Measurement (Force Scale: 100-1800 N)</div>', unsafe_allow_html=True)

st.markdown("""
<div class="input-container">
    <div class="input-label">Enter Hair Force Values for Each Region (in Newtons)</div>
</div>
""", unsafe_allow_html=True)

regions = ['Vertex', 'Frontal', 'R. Temporal', 'L. Temporal', 'R. Parietal', 'L. Parietal', 'Occipital']

# Create two columns for inputs
col1, col2 = st.columns(2)

for idx, region in enumerate(regions):
    if idx < 4:
        with col1:
            input_value = st.number_input(
                f"{region} Hair Force (N)",
                min_value=100,
                max_value=1800,
                value=st.session_state.hair_strength[region],
                step=10,
                key=f"input_{region}"
            )
            st.session_state.hair_strength[region] = input_value
    else:
        with col2:
            input_value = st.number_input(
                f"{region} Hair Force (N)",
                min_value=100,
                max_value=1800,
                value=st.session_state.hair_strength[region],
                step=10,
                key=f"input_{region}"
            )
            st.session_state.hair_strength[region] = input_value

# Calculate average strength
average_strength = np.mean(list(st.session_state.hair_strength.values()))

# Display Average Strength
avg_cat, avg_css, avg_stat = get_strength_category(average_strength)

st.markdown(f"""
<div class="average-strength">
    <div class="average-strength-label">Overall Average Hair Force</div>
    <div class="average-strength-value">{round(average_strength, 1)} N</div>
    <div class="average-strength-category">Status: {avg_cat}</div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Display individual strength values
st.markdown('<div class="card"><div class="card-title">Hair Force by Region</div>', unsafe_allow_html=True)

for region in regions:
    strength_val = st.session_state.hair_strength[region]
    category, css_class, _ = get_strength_category(strength_val)
    
    st.markdown(f"""
    <div class="hair-strength-display">
        <span class="strength-region-name">{region}</span>
        <span class="strength-value {css_class}">{strength_val} N - {category}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Overall Status Badge
st.markdown('<div class="card">', unsafe_allow_html=True)

cat, css, stat = get_strength_category(average_strength)

st.markdown(f"""
<div class="status-badge {stat}">
    Overall Hair Strength Status: {cat.upper()}
</div>
""", unsafe_allow_html=True)

# Get recommendations based on average strength
recommendations = get_recommendations(average_strength)

st.markdown(f"""
<div class="{recommendations['box_type']}">
    <strong>{recommendations['status']}</strong><br><br>
    {recommendations['description']}<br><br>
    <strong>Recommended Actions:</strong>
    <ul style="margin-top: 10px; margin-bottom: 0;">
""", unsafe_allow_html=True)

for rec in recommendations['recommendations']:
    st.markdown(f"<li>{rec}</li>", unsafe_allow_html=True)

st.markdown("""
    </ul>
</div>
""", unsafe_allow_html=True)

# Information Section
st.markdown("<h3 style='color: #0f766e; margin-top: 30px; font-size: 1.3em;'>Understanding Hair Force Scale</h3>", unsafe_allow_html=True)

with st.expander("Hair Force Categories & What They Mean", expanded=True):
    st.markdown("""
    **Hair Force Scale (100-1800 Newtons):**
    
    **Weak Hair (100-600 N):** 
    - Hair has low tensile strength and is prone to breakage
    - May indicate severe nutritional deficiency, hormonal issues, or damage
    - Requires immediate professional attention and intensive care
    - Signs: Excessive breakage, dull appearance, weak roots, hair shedding
    - Common causes: Poor nutrition, stress, hormonal imbalance, chemical damage
    
    **Normal Hair (601-1200 N):**
    - Hair demonstrates healthy strength and normal elasticity
    - Good protein structure and adequate nutrient levels
    - Maintenance of current care practices recommended
    - Signs: Minimal breakage, good shine, normal growth rate, manageable texture
    - Common causes: Balanced diet, proper care, healthy lifestyle
    
    **Strong Hair (1201-1800 N):**
    - Hair has superior tensile strength and maximum resilience
    - Excellent protein structure and optimal nutrient levels
    - Continue healthy lifestyle and care practices
    - Signs: No breakage, high shine, strong growth, thick texture
    - Common causes: Excellent nutrition, genetics, optimal care routine
    
    **Note:** Hair force is measured in Newtons (N), representing the pulling force required to break a single hair strand.
    """)

with st.expander("Factors Affecting Hair Strength", expanded=False):
    st.markdown("""
    **Nutritional Factors:**
    - Protein: Primary component of hair (keratin is a protein)
    - Biotin (Vitamin B7): Essential for hair growth and strength
    - Iron: Prevents hair loss and supports oxygen delivery
    - Zinc: Supports hair tissue growth and repair
    - Vitamins A, C, D, E: Antioxidant protection and collagen production
    - Omega-3 fatty acids: Nourish hair and support scalp health
    
    **Lifestyle Factors:**
    - Stress levels: High cortisol can weaken hair
    - Sleep quality: 7-9 hours recommended for cell regeneration
    - Exercise: Improves blood circulation to scalp
    - Hydration: Essential for hair moisture and flexibility
    - Smoking & alcohol: Can damage hair structure
    
    **Hair Care Practices:**
    - Heat damage: From styling tools (straighteners, curlers)
    - Chemical treatments: Coloring, perming, relaxing
    - Product quality: Sulfates, parabens can damage hair
    - Washing frequency: Over-washing strips natural oils
    - Brushing technique: Aggressive brushing causes breakage
    
    **Health Conditions:**
    - Thyroid disorders: Hypothyroidism weakens hair
    - Autoimmune diseases: Alopecia areata, lupus
    - Hormonal imbalances: PCOS, menopause
    - Nutritional deficiencies: Anemia, malabsorption
    - Medications: Chemotherapy, beta-blockers, anticoagulants
    """)

with st.expander("Tips to Improve Hair Strength", expanded=False):
    st.markdown("""
    **Nutrition:**
    - Eat protein-rich foods: eggs, fish, chicken, beans, lentils
    - Include biotin sources: nuts, seeds, whole grains, avocado
    - Take iron and zinc supplements if deficient (consult doctor)
    - Consume fruits and vegetables for vitamins and antioxidants
    - Include omega-3 rich foods: salmon, walnuts, flaxseeds
    - Stay hydrated: 8-10 glasses of water daily
    
    **Hair Care:**
    - Minimize heat styling (use lower temperatures below 350°F)
    - Avoid harsh chemical treatments or space them out
    - Use sulfate-free, gentle shampoos and conditioners
    - Apply deep conditioning treatments weekly
    - Trim hair every 6-8 weeks to prevent split ends
    - Use wide-tooth combs when hair is wet
    - Sleep on silk or satin pillowcases to reduce friction
    
    **Lifestyle:**
    - Manage stress through meditation, yoga, or therapy
    - Sleep 7-9 hours daily for optimal cell regeneration
    - Exercise regularly (30 minutes, 5 days/week)
    - Avoid smoking and limit alcohol consumption
    - Protect hair from sun, chlorine, and pollution
    
    **Professional Help:**
    - Consult dermatologist or trichologist for persistent issues
    - Consider professional scalp treatments and analysis
    - Get regular hair health assessments (annually)
    - Use prescription treatments if needed (minoxidil, finasteride)
    - Blood tests to check for deficiencies (iron, vitamin D, thyroid)
    """)

with st.expander("Hair Force Measurement Methods", expanded=False):
    st.markdown("""
    **How Hair Strength is Measured:**
    
    Hair strength is typically measured using specialized equipment that applies tension to individual hair strands until they break. The force required to break the hair is recorded in Newtons (N).
    
    **Common Testing Methods:**
    - Tensile Testing: Pulling a hair strand until it breaks
    - Trichometer: Device that measures hair tensile strength
    - Specialized Labs: Professional hair analysis facilities
    
    **What Affects Test Results:**
    - Hair moisture content
    - Hair diameter and thickness
    - Testing conditions (temperature, humidity)
    - Hair chemical treatments
    - Natural hair color (melanin content)
    
    **Professional Assessment:**
    For accurate measurements, consult a trichologist or dermatologist who can perform standardized hair strength tests using calibrated equipment.
    """)

st.markdown('</div>', unsafe_allow_html=True)

# Action Buttons
col_btn1, col_btn2, col_btn3 = st.columns(3)

with col_btn1:
    if st.button('Reset All Values', use_container_width=True):
        for region in regions:
            st.session_state.hair_strength[region] = 900
        st.success('All values reset to 900 N!')
        st.rerun()

with col_btn2:
    if st.button('Generate Report', use_container_width=True):
        # Create report data
        report_data = {
            'Region': regions,
            'Force (N)': [st.session_state.hair_strength[r] for r in regions],
            'Category': [get_strength_category(st.session_state.hair_strength[r])[0] for r in regions]
        }
        df = pd.DataFrame(report_data)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Hair Strength Report</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)
        
        # Summary statistics
        st.markdown(f"""
        **Summary Statistics:**
        - Average Force: {average_strength:.1f} N
        - Minimum Force: {min(st.session_state.hair_strength.values())} N ({min(st.session_state.hair_strength, key=st.session_state.hair_strength.get)})
        - Maximum Force: {max(st.session_state.hair_strength.values())} N ({max(st.session_state.hair_strength, key=st.session_state.hair_strength.get)})
        - Overall Status: {cat}
        - Date: {datetime.now().strftime("%B %d, %Y %I:%M %p")}
        """)
        st.markdown('</div>', unsafe_allow_html=True)

with col_btn3:
    if st.button('Export Data (CSV)', use_container_width=True):
        report_data = {
            'Region': regions,
            'Force_N': [st.session_state.hair_strength[r] for r in regions],
            'Category': [get_strength_category(st.session_state.hair_strength[r])[0] for r in regions]
        }
        df = pd.DataFrame(report_data)
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"hair_strength_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# Footer
st.markdown("""
<div class="footer">
    <div class="footer-title">Medical Disclaimer</div>
    <div class="footer-text">
        This hair strength assessment system is designed to provide general guidance on hair health.<br>
        It should not replace professional medical diagnosis and treatment.<br>
        Always consult with a board-certified dermatologist or trichologist for persistent hair issues or concerns.
    </div>
    <div style="margin-top: 25px; opacity: 0.7; font-size: 0.85em;">
        © 2025 Hair Strength Assessment System<br>
        Powered by Advanced Hair Health Analysis | Force Scale: 100-1800 N
    </div>
</div>
""", unsafe_allow_html=True)
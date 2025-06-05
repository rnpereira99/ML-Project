import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb


st.set_page_config(
    layout="wide",
    page_title="Workers Compensation Claim Predictor",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_model_and_mappings():
    model = pickle.load(open('tuned_XGB.pkl', 'rb'))
    
    district_mapping = {
        'ALBANY': 0, 'BINGHAMTON': 1, 'BINGHAMTON': 2,
        'BUFFALO': 3, 'ROCHESTER': 4, 'SYRACUSE': 5, 'STATEWITE': 6
    }
    
    carrier_mapping = {
    'PRIVATE': 0,                               
    'SIF': 1,                                  
    'SELF PUBLIC': 2,                      
    'SELF PRIVATE': 3,                               
    'SPECIAL FUND - CONS. COMM.': 4,            
    'SPECIAL FUND - POI CARRIER': 5,                 
    'SPECIAL FUND - UNKNOWN': 6,                     
    'UNKNOWN': 7                                
}
    
    medical_fee_mapping = {
        '1': 0, '2': 1, '3': 2, '4': 3, 'UK': 4
    }
    
    return model, district_mapping, carrier_mapping, medical_fee_mapping

try:
    model, district_mapping, carrier_mapping, medical_fee_mapping = load_model_and_mappings()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()


st.title("Workers Compensation Claim Predictor")


# Create tabs for the diffrent sections
tab1, tab2, tab3, tab4 = st.tabs([
    "Basic Information",
    "Timeline Details",
    "Additional Factors",
    "Medical Information"
])

# Store form data in session state
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}

# Tab 1: Basic Information
with tab1:
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown('<div class="step-header">Step 1: Enter Basic Information</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.form_data['birth_year'] = st.number_input(
            "Birth Year",
            min_value=1940,
            max_value=2010,
            value=1980,
            help="Enter the claimant's birth year"
        )
        
        st.session_state.form_data['accident_month'] = st.selectbox(
            "Accident Month",
            options=list(range(1, 13)),
            format_func=lambda x: pd.Timestamp(2024, x, 1).strftime('%B'),
            help="Month when the accident occurred"
        )
        
        st.session_state.form_data['district_name'] = st.selectbox(
            "District Name",
            options=list(district_mapping.keys()),
            help="Select the district office handling the claim"
        )
    
    with col2:
        st.session_state.form_data['medical_fee_region'] = st.selectbox(
            "Medical Fee Region",
            options=list(medical_fee_mapping.keys()),
            help="Medical fee region code"
        )
        
        st.session_state.form_data['carrier_name'] = st.selectbox(
            "Insurance Carrier Type",
            options=list(carrier_mapping.keys()),
            help="Select the insurance carrier type"
        )
        
        st.session_state.form_data['attorney_representative'] = st.checkbox(
            "Attorney/Representative Present",
            help="Is the claimant represented by an attorney?"
        )

# Tab 2: Timeline Details
with tab2:
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown('<div class="step-header">Step 2: Timeline Information</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Form Submission Timeline")
        st.session_state.form_data['c2_after_c3_flag'] = st.checkbox(
            "C2 Form submitted after C3",
            help="Was Form C2 submitted after Form C3?"
        )
        
        st.session_state.form_data['c3_after_assembly_flag'] = st.checkbox(
            "C3 Form submitted after assembly",
            help="Was Form C3 submitted after claim assembly?"
        )
        
        st.session_state.form_data['c3_date_converted'] = st.checkbox(
            "C3 Date Converted",
            help="C3 date conversion status"
        )
    
    with col2:
        st.markdown("##### Processing Durations (Days)")
        st.session_state.form_data['days_accident_to_c3'] = st.number_input(
            "Days: Accident to C3 Form",
            min_value=0,
            max_value=1000,
            value=0,
            help="Number of days between accident and C3 form submission"
        )
        
        st.session_state.form_data['days_accident_to_assembly'] = st.number_input(
            "Days: Accident to Assembly",
            min_value=0,
            max_value=1000,
            value=0,
            help="Number of days between accident and claim assembly"
        )
        
        st.session_state.form_data['days_accident_to_first_hearing'] = st.number_input(
            "Days: Accident to First Hearing",
            min_value=0,
            max_value=1000,
            value=0,
            help="Number of days between accident and first hearing"
        )

# Tab 3: Additional Factors
with tab3:
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown('<div class="step-header">Step 3: Risk Factors and Demographics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Risk Assessment")
        st.session_state.form_data['region_risk_score'] = st.slider(
            "Region Risk Score",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            help="Risk score associated with the region"
        )
        
        st.session_state.form_data['county_claims_normalized'] = st.slider(
            "County Claims (Normalized)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            help="Normalized claim count for the county"
        )
    
    with col2:
        st.markdown("##### Additional Details")
        st.session_state.form_data['age_outlier_flag'] = st.checkbox(
            "Age is Statistical Outlier",
            help="Is the claimant's age considered an outlier?"
        )
        
        st.session_state.form_data['average_weekly_wage'] = st.number_input(
            "Average Weekly Wage ($)",
            min_value=0,
            max_value=10000,
            value=0,
            help="Claimant's average weekly wage"
        )
        
        st.session_state.form_data['ime_4_count'] = st.number_input(
            "Number of IME-4 Forms",
            min_value=0,
            max_value=10,
            value=0,
            help="Number of IME-4 forms received"
        )

# Tab 4: Medical Information
with tab4:
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown('<div class="step-header">Step 4: Injury Details</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.form_data['wcio_part_body_code'] = st.number_input(
            "Body Part Code (WCIO)",
            min_value=0,
            max_value=99,
            value=30,
            help="Code representing the injured body part"
        )
    
    with col2:
        st.session_state.form_data['wcio_cause_injury_code'] = st.number_input(
            "Injury Cause Code (WCIO)",
            min_value=0,
            max_value=99,
            value=1,
            help="Code representing the cause of injury"
        )

# Prediction Section
st.markdown("### Generate Prediction")
if st.button("Predict Claim Type", use_container_width=True):
    with st.spinner("Analyzing claim data..."):
        # Create input DataFrame
        input_data = pd.DataFrame({
            'Days_Accident_to_First_Hearing': [st.session_state.form_data['days_accident_to_first_hearing']],
            'IME-4 Count': [st.session_state.form_data['ime_4_count']],
            'Average Weekly Wage': [st.session_state.form_data['average_weekly_wage']],
            'Attorney/Representative': [int(st.session_state.form_data['attorney_representative'])],
            'WCIO Cause of Injury Code': [st.session_state.form_data['wcio_cause_injury_code']],
            'WCIO Part Of Body Code': [st.session_state.form_data['wcio_part_body_code']],
            'Region_Risk_Score': [st.session_state.form_data['region_risk_score']],
            'C2_After_C3_Flag': [int(st.session_state.form_data['c2_after_c3_flag'])],
            'C3_After_Assembly_Flag': [int(st.session_state.form_data['c3_after_assembly_flag'])],
            'C-3 Date Converted': [int(st.session_state.form_data['c3_date_converted'])],
            'Days_Accident_to_C3': [st.session_state.form_data['days_accident_to_c3']],
            'Carrier Name': [carrier_mapping[st.session_state.form_data['carrier_name']]],
            'Birth Year': [st.session_state.form_data['birth_year']],
            'Days_Accident_to_Assembly': [st.session_state.form_data['days_accident_to_assembly']],
            'District Name': [district_mapping[st.session_state.form_data['district_name']]],
            'Medical Fee Region': [medical_fee_mapping[st.session_state.form_data['medical_fee_region']]],
            'Accident_Month': [st.session_state.form_data['accident_month']],
            'Age_Outlier_Flag': [int(st.session_state.form_data['age_outlier_flag'])],
            'County_Claims_Normalized': [st.session_state.form_data['county_claims_normalized']]
        })

        try:
            prediction = model.predict(input_data)[0]
            probas = model.predict_proba(input_data)[0]
            
            claim_types = {
                0: 'CANCELLED',
                1: 'NON-COMP',
                2: 'MED ONLY',
                3: 'TEMPORARY',
                4: 'PPD SCH LOSS',
                5: 'PPD NSL',
                6: 'PTD',
                7: 'DEATH'
            }
            
            # Primary prediction
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric(
                    label="Predicted Claim Type",
                    value=claim_types[prediction]
                )


            
            # Probability distribution
            with col2:
                st.markdown("##### Probability Distribution")
                prob_df = pd.DataFrame({
                    'Claim Type': claim_types.values(),
                    'Probability': probas
                })
                prob_df = prob_df.sort_values('Probability', ascending=False)
                st.bar_chart(prob_df.set_index('Claim Type'))
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            with st.expander("View Debug Information"):
                st.write("Input data types:")
                st.write(input_data.dtypes)
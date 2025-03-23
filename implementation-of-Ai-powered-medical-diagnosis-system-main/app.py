import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from streamlit_option_menu import option_menu

# Set up Streamlit page
st.set_page_config(page_title="Disease Prediction", page_icon="⚕️")

# Hide Streamlit UI elements
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Add background image
background_image_url = "https://www.strategyand.pwc.com/m1/en/strategic-foresight/sector-strategies/healthcare/ai-powered-healthcare-solutions/img01-section1.jpg"
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
background-image: url({"https://cdn.analyticsvidhya.com/wp-content/uploads/2022/01/30738medtec-futuristic-650-672c56a896ab7.webp"});
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load models
models = {
    'diabetes': pickle.load(open('Models/diabetes_model.sav', 'rb')),
    'heart_disease': pickle.load(open('Models/heart_disease_model.sav', 'rb')),
    'parkinsons': pickle.load(open('Models/parkinsons_model.sav', 'rb')),
    'lung_cancer': pickle.load(open('Models/lungs_disease_model.sav', 'rb')),
    'thyroid': pickle.load(open('Models/Thyroid_model.sav', 'rb'))
}

# Disease selection
disease_options = {
    'Diabetes Prediction': 'diabetes',
    'Heart Disease Prediction': 'heart_disease',
    'Parkinsons Prediction': 'parkinsons',
    'Lung Cancer Prediction': 'lung_cancer',
    'Hypo-Thyroid Prediction': 'thyroid'
}
selected = st.selectbox("Select a Disease to Predict", list(disease_options.keys()))

#for visualizing the features
def plot_feature_distribution(data, feature_name):
    fig, ax = plt.subplots()
    sns.histplot(data, bins=20, kde=True, ax=ax)
    ax.set_title(f'Distribution of {feature_name}')
    ax.set_xlabel(feature_name)
    st.pyplot(fig)

# Function to collect inputs
def collect_inputs(features):
    user_inputs = []
    for feature in features:
        user_inputs.append(st.number_input(feature, key=feature))
    return np.array(user_inputs).reshape(1, -1)

# Prediction logic
selected_model = disease_options[selected]
features_map = {
    'diabetes': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
    'heart_disease': ['Age', 'Sex', 'ChestPain', 'RestBP', 'Cholesterol', 'FBS', 'RestECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'Slope', 'CA', 'Thal'],
    'parkinsons': ['Fo', 'Fhi', 'Flo', 'Jitter_percent', 'Jitter_Abs', 'RAP', 'PPQ', 'DDP', 'Shimmer', 'Shimmer_dB'],
    'lung_cancer': ['Gender', 'Age', 'Smoking', 'YellowFingers', 'Anxiety', 'PeerPressure', 'ChronicDisease', 'Fatigue', 'Allergy', 'Wheezing'],
    'thyroid': ['Age', 'Sex', 'OnThyroxine', 'TSH', 'T3Measured', 'T3', 'TT4']
}

st.title(f'{selected} Test')
st.write("Enter the required details below:")
user_input = collect_inputs(features_map[selected_model])

if st.button(f'Predict {selected}'):
    with st.spinner('Processing...'):
        time.sleep(2)  # Simulate a delay
    prediction = models[selected_model].predict(user_input)
    result = 'Positive' if prediction[0] == 1 else 'Negative'
    st.success(f'Test Result: {result}')
    
    # Animation effect using progress bar
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        progress_bar.progress(percent_complete + 1)
    
    # Visualization: Feature distribution
    for idx, feature in enumerate(features_map[selected_model]):
        plot_feature_distribution(user_input[:, idx], feature)



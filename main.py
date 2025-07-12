import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# Custom Styling
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #E8F5E9, #A5D6A7, #66BB6A);
        background-attachment: fixed;
        color: #1B5E20;
    }
    .stButton>button {
        background-color:rgb(150, 201, 153);
        color: white;
        font-size: 18px;
        padding: 10px 25px;
        border-radius: 12px;
        font-family: 'Arial', sans-serif;
        transition: 0.3s;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color:rgb(130, 144, 131);
        transform: scale(1.05);
    }
    .title-text {
        font-size: 42px;
        font-weight: bold;
        color:rgb(27, 28, 27);
        text-align: center;
        font-family: 'Trebuchet MS', sans-serif;
    }
    .subtitle-text {
        font-size: 24px;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        font-family: 'Lucida Console', monospace;
    }
    .quote-text {
        font-size: 18px;
        font-style: italic;
        color:rgb(48, 65, 48);
        text-align: center;
        font-family: 'Georgia', serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load Model
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = Image.open(test_image).resize((128, 128))
    input_arr = np.expand_dims(np.array(image), axis=0)  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar Navigation
st.sidebar.title("🌿 Dashboard")
st.sidebar.markdown("### Navigation 🌍")
app_mode = st.sidebar.radio("Select Page", ["🏡 Home", "📖 About", "🔍 Disease Recognition"])

# Home Page
if app_mode == "🏡 Home":
    st.markdown("<h3><b><u><p class='title-text'>🌱 Plant Disease Recognition System</p></u></b></h3>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Healthy Plants, Healthy Future! 🍀</p>", unsafe_allow_html=True)
    st.image("home_page.jpeg", use_container_width=True)
    st.markdown("<p class='quote-text'>\"Look deep into nature, and then you will understand everything better.\"</p>", unsafe_allow_html=True)
    st.markdown("""
    ### 🌾 Welcome to the Plant Disease Recognition System!
    Upload an image of a plant leaf, and our AI-powered system will analyze it to detect any diseases.\n
    ✅ **Fast & Accurate** disease detection.\n
    ✅ **User-Friendly Interface**.\n
    🚀 Click on the **Disease Recognition** page in the sidebar to get started!
    """)

# About Page
elif app_mode == "📖 About":
    st.title("📚 About the Project")
    st.markdown("<p class='quote-text'>\"The ultimate goal of farming is not the growing of crops, but the cultivation of human beings.\"</p>", unsafe_allow_html=True)
    st.markdown("""
    Our dataset consists of **87K+ images** categorized into **38 different classes** of plant diseases.
    - 🌿 **Train Set:** 70,295 images
    - 🌾 **Validation Set:** 17,572 images
    - 🌱 **Test Set:** 33 images
    Using **deep learning** and **TensorFlow**, we trained a model to classify plant diseases with high accuracy.
    """)

# Disease Recognition Page
elif app_mode == "🔍 Disease Recognition":
    st.title("🔍 Disease Recognition")
    st.subheader("Upload an image of a plant leaf")
    test_image = st.file_uploader("📸 Choose an Image:", type=["jpg", "png", "jpeg"])
    
    if test_image is not None:
        st.image(test_image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🌟 Predict Disease"):
            with st.spinner('Analyzing Image... ⏳'):
                time.sleep(2)  # Simulate processing delay
                result_index = model_prediction(test_image)
                
            class_names = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]
            
            predicted_class = class_names[result_index]
            st.markdown(
                f"""
                <div style="
                background-color:#DFF2BF;
                color:#4F8A10;
                font-size:24px;
                font-weight:bold;
                padding:15px;
                border-radius:10px;
                text-align:center;
                font-family: 'Arial', sans-serif;">
                🍃 Our AI Predicts: <span style="color:#276749;">{predicted_class}</span>
                </div>
                """, 
                unsafe_allow_html=True
                )


# Footer
st.markdown("""
    ---
    © 0705 Vishal L Shettigar | Plant Disease Recognition System 🌱
""")
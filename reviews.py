import streamlit as st
import pickle

# Set page configuration
st.set_page_config(page_title="Kalki2898MoviesReviewsIMDB", page_icon=":package:", layout="centered")

# Display images
st.image("Screenshot 2024-07-25 185502.png")
st.image("Kalki_2898AD_logo-jpg.webp")
st.image("logo-imdb.png")

# Load the vectorizer and model using pickle
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Title of the app
st.title("Sentiment Analysis App")

# Instructions
st.write("Enter a movie review to analyze its sentiment.")

# Text input
text_input = st.text_area("Enter review text:")

# Button to trigger prediction
if st.button("Analyze"):
    if text_input:
        # Transform the input text
        text_vector = vectorizer.transform([text_input])
        # Predict sentiment
        prediction = model.predict(text_vector)[0]
        
        # Display the corresponding image based on the prediction
        if prediction == 'Positive':
            st.image("KalkiPOSTIVE9758.jpg", caption="Positive Review")
        else:
            st.image("KalkiNegative3783.jpg", caption="Negative Review")
        
        # Display the prediction text
        st.write(f"Sentiment: {prediction}")
    else:
        st.write("Please enter some text to analyze.")

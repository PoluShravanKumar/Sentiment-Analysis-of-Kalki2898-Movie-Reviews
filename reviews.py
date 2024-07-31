import streamlit as st
import pickle

st.set_page_config(page_title="Kalki2898MoviesReviewsIMDB", page_icon=":😊:", layout="centered")

st.image("Screenshot 2024-07-25 185502.png")
st.image("Kalki_2898AD_logo-jpg.webp")
st.image("logo-imdb.png")

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Sentiment Analysis App")
st.subheader("By Shravan Kumar Polu")

st.write("Enter a movie review to analyze its sentiment.")

text_input = st.text_area("Enter review text:")

if st.button("Analyze"):
    if text_input:
        text_vector = vectorizer.transform([text_input])
        prediction = model.predict(text_vector)[0]
        
        if prediction == 'Positive':
            st.image("KalkiPOSTIVE9758.jpg", caption="Positive Review")
        else:
            st.image("KalkiNegative3783.jpg", caption="Negative Review")
        
        st.write(f"Sentiment: {prediction}")
    else:
        st.write("Please enter some text to analyze.")
st.markdown("---")
st.markdown("© 2024 Shravan Kumar Polu")

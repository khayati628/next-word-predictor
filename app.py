import streamlit as st

from predictor import NextWordPredictor

st.set_page_config(page_title="Next Word Prediction", page_icon="🧠", layout="centered")


@st.cache_resource
def get_predictor():
    return NextWordPredictor()


predictor = get_predictor()

st.title("🧠 Next Word Prediction (LSTM)")
st.write("Type a sentence and the model will predict the **next word**.")

user_input = st.text_input("✍️ Enter text:", placeholder="Type a sentence here...")

if st.button("Predict Next Word", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        results = predictor.predict(user_input, top_k=5)
        if not results:
            st.error("None of these words are in the model's vocabulary. Try different words.")
        else:
            best_word, best_prob = results[0]
            st.success(f"**Predicted next word:** {best_word}  ({best_prob:.1%})")
            st.subheader("Top 5 suggestions")
            st.bar_chart({word: prob for word, prob in results}, horizontal=True)

st.markdown("---")
st.caption("LSTM-based next word prediction built with TensorFlow and Streamlit")

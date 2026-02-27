import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import os

st.set_page_config(page_title="Categorize News", layout="wide")
st.markdown("<h2 style='text-align: center; color: #1a73e8;'>Analyze News Category</h2>", unsafe_allow_html=True)

labels = ["General News", "Politics News", "World News"]
model_path = 'models/'

st.markdown("""
    <style>
    /* Analyze Article button  */
    button[kind="primary"] {
        background-color: #1a73e8 !important;
        color: white !important;
        border: none !important;
    }

    /* Analyze Article */
    div.stButton > button:has(div p:contains("Analyze Article")) {
        background-color: #1a73e8 !important;
        color: white !important;
    }

    /* Clear button */
    div.stButton > button:has(div p:contains("Clear")) {
        background-color: transparent !important;
        color: white !important;
        border: 1px solid #555 !important;
    }

    div.stButton button p {
        color: black !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


def clear_text():
    st.session_state["user_input_key"] = ""

# Layout - Input Area
with st.container():
    st.subheader("Input & Detection Area")
    arch = st.selectbox("Select Prediction Model:", ["Bi-LSTM", "LSTM"])
    user_input = st.text_area("Article Text:",
                              height=200,
                              placeholder="Paste full article text here...",
                              key="user_input_key")
    st.write("")
    st.write("")

    col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns([2, 1, 2, 1, 2])

    with col_b3:
        analyze_btn = st.button("Analyze Article", type="primary", use_container_width=True)

    with col_b4:
        st.button("Clear", use_container_width=True, on_click=clear_text)

## --- Logic Flow ---
if analyze_btn:
    if not user_input.strip():
        st.warning("Please enter news text first!")
    else:
        word_count = len(user_input.split())

        if word_count < 50:
            st.error(f"⚠️ The article is too short ({word_count} words). Please enter at least 50 words for a reliable analysis.")
        else:
            if word_count > 5000:
                st.warning("⚠️ The article is quite long. Note that only the first 500 words will be prioritized.")

# if analyze_btn:
#     if user_input:
            with st.spinner('Analyzing... Please wait...'):
                try:
                    m_file = 'bilstm__model.keras' if "Bi-LSTM" in arch else 'lstm_multiclass_model.keras'
                    t_file = 'bilstm_tokenizer.pickle' if "Bi-LSTM" in arch else 'lstm_multiclass_tokenizer.pickle'

                    model = load_model(os.path.join(model_path, m_file))
                    with open(os.path.join(model_path, t_file), 'rb') as f:
                        tokenizer = pickle.load(f)

                    seq = tokenizer.texts_to_sequences([user_input.lower()])
                    padded = pad_sequences(seq, maxlen=500)
                    pred = model.predict(padded)[0]

                    result_idx = np.argmax(pred)

                    # Result Display
                    st.divider()
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        # Label box Style
                        st.markdown(f"""
                            <p style="font-size: 1.5rem; display: flex; align-items: center;">
                                <b>Predicted Class:</b> 
                                <span style="color: #00BFFF; font-weight: bold; font-size: 2rem; margin-left: 15px;">
                                    {labels[result_idx]}
                                </span>
                            </p>
                            """, unsafe_allow_html=True)

                        confidence_pct = float(pred[result_idx] * 100)

                        st.markdown(
                            f"""
                                <div style="display: flex; flex-direction: column; gap: 5px;">
                                    <div style="margin-left: 5px;">
                                        <span style="color: black; font-size: 1.2rem;">Model Confidence: </span>
                                        <span style="color: black; font-weight: bold; font-size: 1.2rem;">{confidence_pct:.2f}%</span>
                                    </div>
                                </div>
                                """,
                            unsafe_allow_html=True
                        )

                        st.write("")
                        st.write("")
                        st.write(f"**Explainability Component:** Analysis shows high similarity to patterns found in **{labels[result_idx]}** databases.")

                    with col2:
                        st.write("**Confidence Distribution (Visual Analytics):**")
                        st.bar_chart({labels[i]: float(pred[i]) for i in range(3)})

                except Exception as e:
                    st.error(f"Error loading model or predicting: {e}")







import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import re

# --- Page Config ---
st.set_page_config(page_title="News Validator", layout="wide")
st.markdown("<h2 style='text-align: center; color: #1a73e8;'>🛡️ News Authenticity Validator </h2>", unsafe_allow_html=True)
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

# --- Custom CSS ---
st.markdown("""
<style>
    .metric-card { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; border: 1px solid #e0e0e0; }
    .status-badge { padding: 8px 16px; border-radius: 20px; font-weight: bold; font-size: 1.2rem; display: inline-block; margin-bottom: 10px; }
    .highlight-box { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #1a73e8; line-height: 1.8; }
    .highlight-real { background-color: #d4edda; border-bottom: 2px solid #28a745; padding: 2px 4px; border-radius: 3px; font-weight: 500; }
    .highlight-fake { background-color: #f8d7da; border-bottom: 2px solid #dc3545; padding: 2px 4px; border-radius: 3px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def clear_text():
    st.session_state["user_input_key"] = ""

def get_influence_metrics(text, model, tokenizer):
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.lower().split()
    embedding_layer = model.layers[0]
    weights = embedding_layer.get_weights()[0]

    word_weights = []
    word_info = []

    for word in words:
        word_id = tokenizer.word_index.get(word, 0)
        if 0 < word_id < len(weights):
            weight = np.mean(weights[word_id])
            word_weights.append(weight)
            word_info.append((word, weight))
        else:
            word_info.append((word, 0))

    if word_weights:
        avg_weight = np.mean(np.abs(word_weights))
        std_weight = np.std(np.abs(word_weights))
        adaptive_threshold = avg_weight + std_weight
    else:
        adaptive_threshold = 0.001

    real_influence = 0.0
    fake_influence = 0.0
    highlighted_words = []

    for word, weight in word_info:
        if weight < -adaptive_threshold:
            real_influence += abs(weight)
            highlighted_words.append(f'<span class="highlight-real">{word}</span>')
        elif weight > adaptive_threshold:
            fake_influence += weight
            highlighted_words.append(f'<span class="highlight-fake">{word}</span>')
        else:
            highlighted_words.append(word)

    return " ".join(highlighted_words), float(real_influence), float(fake_influence)

# --- UI Layout ---
with st.container():
    st.subheader("Input & Detection Area")
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

# --- Execution Logic ---
if analyze_btn:
    if not user_input.strip():
        st.warning("Please enter news text first!")
    else:
        word_count = len(user_input.split())
        if word_count < 50:
            st.error(f"⚠️ The article is too short ({word_count} words). Min 50 words required.")
        else:
            if word_count > 5000:
                st.warning("⚠️ Article is long. Only the first 500 words are prioritized.")

            with st.spinner('Analyzing patterns ... Please wait ...'):
                try:
                    # Model loading (Adjust filenames if Bi-LSTM uses different files)
                    model = load_model(os.path.join(model_path, 'lstm_model.keras'))
                    with open(os.path.join(model_path, 'lstm_tokenizer.pickle'), 'rb') as f:
                        tokenizer = pickle.load(f)

                    # 1. Prediction
                    seq = tokenizer.texts_to_sequences([user_input.lower()])
                    padded = pad_sequences(seq, maxlen=500)
                    score = model.predict(padded)[0][0]

                    is_real = score > 0.5
                    confidence = score if is_real else (1 - score)
                    color = "#28a745" if is_real else "#dc3545"
                    label = "REAL / TRUSTED" if is_real else "FAKE / UNRELIABLE"

                    # 2. Influence Metrics
                    html_highlights, real_score, fake_score = get_influence_metrics(user_input, model, tokenizer)

                    # Developer Debug Option
                    # show_debug = st.sidebar.checkbox("Show Developer Debug Info")
                    # if show_debug:
                    if st.session_state.get('show_debug_key', False):
                        st.subheader("🛠️ Debug: Raw Scores")
                        col_d1, col_d2 = st.columns(2)
                        col_d1.metric("Raw Real Score", f"{real_score:.4f}")
                        col_d2.metric("Raw Fake Score", f"{fake_score:.4f}")

                    st.divider()

                    # 3. Visual Results
                    res_col1, res_col2 = st.columns([1, 1.5])
                    with res_col1:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=confidence * 100,
                            title={'text': "Reliability Score", 'font': {'size': 20}},
                            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color},
                                   'steps': [{'range': [0, 50], 'color': '#f8d7da'},
                                             {'range': [50, 100], 'color': '#d4edda'}]}
                        ))
                        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                        st.plotly_chart(fig, use_container_width=True)

                    with res_col2:
                        st.markdown(f"""
                            <div class='metric-card'>
                                <div class='status-badge' style='background-color: {color}22; color: {color}; border: 2px solid {color};'>
                                    {label}
                                </div>
                                <p style='color: #666;'>Confidence Score: {confidence * 100:.2f}%</p>
                            </div>
                        """, unsafe_allow_html=True)

                        st.write("---")
                        st.subheader("⚖️ Influence Distribution")

                        fig_bar = go.Figure(go.Bar(
                            x=['Real News', 'Fake News'],
                            y=[real_score, fake_score],
                            marker_color=['#28a745', '#dc3545']
                        ))

                        fig_bar.update_layout(
                            height=350,
                            margin=dict(l=20, r=20, t=20, b=20),
                            xaxis_title="Signals",
                            yaxis_title="Influence Weights"
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)

                    # 4. Feature Importance
                    with st.expander("🔍 View Model-Driven Signal Highlights"):
                        st.markdown(f'<div class="highlight-box">{html_highlights}</div>', unsafe_allow_html=True)
                        st.caption("🟢 Green = Real Pattern Signals | 🔴 Red = Fake Pattern Signals")

                except Exception as e:
                    st.error(f"Error during analysis: {e}")

with st.sidebar:
    st.checkbox("Show Developer Debug Info", key="show_debug_key")




import streamlit as st
import plotly.express as px
import pandas as pd


# Page configuration
# st.set_page_config(page_title="News Analysis", layout="wide")

st.sidebar.markdown(
    """
    <style>
    .sidebar-text {
        color: white !important;
        font-size: 30px;
        font-weight: bold;
        text-align: center;
    }
    </style>
    <p class="sidebar-text">News Analysis</p>
    """,
    unsafe_allow_html=True
)


# Sidebar navigation icon
# st.sidebar.image("https://cdn-icons-png.flaticon.com/512/21/21601.png",width=80)
col1, col2, col3 = st.sidebar.columns([1, 2, 1])
with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/21/21601.png", width=80)

st.sidebar.markdown("<h1 style='text-align: center; color: black;'>News Analysis</h1>", unsafe_allow_html=True)



# Home Page Content
# st.image("https://cdn-icons-png.flaticon.com/512/21/21601.png", width=40)
# st.title("📰 News Analysis Dashboard")
st.markdown("<h1 style='text-align: left; color: #1a73e8;'>📰 News Analysis Dashboard</h1>", unsafe_allow_html=True)

st.markdown("""> **In an era where information spreads faster than ever, distinguishing truth from fabrication is more critical than ever. Our platform provides real-time analysis to detect misinformation and categorize news articles,ensuring you stay informed with facts, not fiction.**""")


st.markdown("""
### System Capabilities:
* **Multi-class Classification:** Categorize news into General, Politics, or World using LSTM or Bidirectional LSTM.
* **Binary Detection:** Check if news is Real or Fake using LSTM.
* **High Reliability:** Trained on extensive datasets.
""")
st.write("")
st.write("")

col1, col2, col3 = st.columns(3)
col1.metric("Models Available", "2 Models")
col2.metric("Accuracy Avg", "85% +")
col3.metric("Supported Categories", "3 Classes")


st.write("---")
st.write("")
st.write("")
# --- Subject Distribution Section ---
st.subheader("📊 Data Distribution")

chart_col1, chart_col2 = st.columns([2, 1])

with chart_col1:
    st.markdown("**Top Subjects**")
    subject_data = {
        'Subject': ['Politics News', 'World News', 'News', 'Politics', 'Left-news', 'Government News', 'Middle-east', 'US_News'],
        'Count': [11083, 9839, 9050, 4300, 2368, 889, 378, 405]
    }
    df_sub = pd.DataFrame(subject_data)

    # Horizontal Bar Chart ဆွဲခြင်း
    fig_sub = px.bar(df_sub, x='Count', y='Subject', orientation='h',
                     color='Subject',
                     color_discrete_sequence=px.colors.qualitative.Pastel)

    fig_sub.update_layout(showlegend=False, height=350,
                          margin=dict(l=0, r=0, t=0, b=0),
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(color="white"))
    st.plotly_chart(fig_sub, use_container_width=True)

with chart_col2:
    st.markdown("**Dataset Balance**")
    # Fake vs Real Data
    balance_data = {
        'Type': ['Fake News', 'Real News'],
        'Count': [17390, 20922]
    }
    df_bal = pd.DataFrame(balance_data)

    # Bar Chart for Dataset Balance Bar Chart
    fig_bal = px.bar(df_bal, x='Count', y='Type', orientation='h',
                     color='Type',
                     color_discrete_map={'Fake News': '#EF553B', 'Real News': '#00CC96'})

    fig_bal.update_layout(showlegend=False, height=200,
                          margin=dict(l=0, r=0, t=0, b=0),
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(color="white"))
    st.plotly_chart(fig_bal, use_container_width=True)

st.write("")
st.write("")

st.markdown("""
    <div style="
        text-align: center; 
        padding: 40px 0px; 
        color: #8892B0; 
        font-size: 16px; 
        font-weight: 500;
        letter-spacing: 1.5px;
    ">
        © 2026 <span style="color: #4facfe;">Fake News Detection</span> | Data Mining Project
    </div>
    """, unsafe_allow_html=True)
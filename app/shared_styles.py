SHARED_STYLES = """
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        max-width: 1000px !important;
    }
    
    /* App background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        background-attachment: fixed;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        padding-top: 0.5rem !important;
        background: linear-gradient(180deg, #1e3a8a 0%, #1e40af 50%, #1e3a8a 100%) !important;
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.3);
    }
    
    /* Sidebar layout */
    section[data-testid="stSidebar"] > div:first-child {
        display: flex;
        flex-direction: column;
    }
    
    section[data-testid="stSidebar"] .sidebar-header {
        order: -1;
    }
    
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
        order: 1;
    }
    
    /* Sidebar navigation */
    [data-testid="stSidebarNav"] {
        margin-top: 0.1rem !important;
        padding-top: 0 !important;
    }
    
    .sidebar-header {
        position: relative;
        z-index: 10;
    }
    
    [data-testid="stSidebarNav"] ul {
        list-style: none !important;
        padding-left: 0 !important;
        margin: 0 !important;
    }
    
    [data-testid="stSidebarNav"] li {
        margin: 0.3rem 0 !important;
        padding: 0.25rem 0 !important;
    }
    
    [data-testid="stSidebarNav"] a {
        color: #e2e8f0 !important;
        text-decoration: none !important;
        padding: 0.6rem 0.9rem !important;
        border-radius: 8px !important;
        display: block !important;
        transition: all 0.3s ease !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stSidebarNav"] a:hover {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.15) 0%, rgba(255, 255, 255, 0.1) 100%) !important;
        color: #ffffff !important;
        transform: translateX(5px);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.25) 0%, rgba(255, 255, 255, 0.15) 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    
    /* Typography */
    h1 {
        text-align: center;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-top: 0.25rem !important;
        padding-top: 0 !important;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem !important;
    }
    
    .prediction-card h1 {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        background: none !important;
        -webkit-background-clip: unset !important;
        background-clip: unset !important;
    }
    
    h2 {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    h3 {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.25rem !important;
        font-size: 1.15rem !important;
    }
    
    h4 {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        margin-top: 0.25rem !important;
        margin-bottom: 0.15rem !important;
    }
    
    [data-testid="stMarkdownContainer"] p {
        color: #cbd5e1;
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }
    
    /* Form elements */
    [data-testid="stForm"] {
        background: rgba(30, 41, 59, 0.6) !important;
        padding: 1rem !important;
        border-radius: 16px !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stNumberInput"] > div > div {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stNumberInput"] > div > div:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    [data-testid="stSelectbox"] > div > div {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stSelectbox"] > div > div:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    [data-baseweb="checkbox"] {
        accent-color: #3b82f6 !important;
    }
    
    [data-testid="stCheckbox"] label {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
    }
    
    /* Buttons */
    [data-testid="stButton"] > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4) !important;
    }
    
    [data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6) !important;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.6) !important;
        padding: 0.5rem !important;
        border-radius: 12px !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* DataFrames */
    [data-testid="stDataFrame"] {
        background: rgba(30, 41, 59, 0.6) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
    }
    
    /* Custom components */
    .element-container {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(226, 232, 240, 0.95) 0%, rgba(203, 213, 225, 0.95) 100%);
        color: #1e293b;
        padding: 0.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(148, 163, 184, 0.3);
        font-weight: 500;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
        text-align: center;
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .card {
        background: rgba(30, 41, 59, 0.6) !important;
        padding: 1rem !important;
        border-radius: 12px !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        backdrop-filter: blur(10px);
        margin: 0.5rem 0;
    }
    
    /* Horizontal rules */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.3), transparent);
        margin: 0.75rem 0;
    }
    
    /* Labels */
    label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
</style>
"""

def apply_shared_styles():
    import streamlit as st
    st.markdown(SHARED_STYLES, unsafe_allow_html=True)

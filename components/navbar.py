import streamlit as st

def render_header():
    st.markdown("""
        <div style="background-color:#1E3A8A;padding:15px;border-radius:8px;margin-bottom:15px">
            <h2 style="color:white;margin:0;text-align:center;">🔬 Machine Learning Virtual Laboratory</h2>
            <p style="color:#E0E7FF;margin:0;text-align:center;font-size:14px;">Interactive Hands-On Environment for Clustering & Quality Evaluation</p>
        </div>
    """, unsafe_allow_html=True)

def render_navigation():
    cols = st.columns(5)
    pages = [
        ("Overview", "📖 Theory"),
        ("Pre-Lab", "📚 Pre-Lab"),
        ("Procedure", "🗺️ Guided Steps"),
        ("Workbench", "⚡ Workbench"),
        ("Analysis", "📊 Post-Test & Report")
    ]
    
    for idx, (page_key, display_name) in enumerate(pages):
        with cols[idx]:
            is_active = (st.session_state.current_page == page_key)
            btn_type = "primary" if is_active else "secondary"
            if st.button(display_name, key=f"nav_{page_key}", type=btn_type, width="stretch"):
                st.session_state.current_page = page_key
                st.rerun()

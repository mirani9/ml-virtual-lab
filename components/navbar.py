import streamlit as st

def render_header():
    st.markdown("""
        <div style="background-color: #1E3A8A; padding: 15px; border-radius: 8px; margin-bottom: 20px; color: white;">
            <h2 style="margin: 0; color: white; font-size: 24px;">🔬 ML Virtual Lab Platform</h2>
            <p style="margin: 0; opacity: 0.9; font-size: 14px;">Interactive Machine Learning Virtual Laboratory — Academic Edition</p>
        </div>
    """, unsafe_allow_html=True)

def render_navigation():
    cols = st.columns(7)
    pages = [("🏠 Home", "Home"), ("📋 Overview", "Overview"), ("📚 Pre-Lab", "Pre-Lab"), ("🗺️ Guided Procedure", "Procedure"), ("⚡ Workbench", "Workbench"), ("📊 Analysis & Quiz", "Analysis"), ("📄 Lab Report", "Report")]
    for idx, (label, page_key) in enumerate(pages):
        with cols[idx]:
            if st.button(label, use_container_width=True, type="primary" if st.session_state.current_page == page_key else "secondary"):
                st.session_state.current_page = page_key
                st.rerun()

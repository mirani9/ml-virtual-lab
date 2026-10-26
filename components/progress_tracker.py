import streamlit as st

def render_step_tracker(steps: list, current_step: int):
    st.markdown("### 📍 Guided Practical Workflow")
    cols = st.columns(len(steps))
    for idx, step in enumerate(steps, 1):
        with cols[idx-1]:
            is_active = (idx == current_step)
            is_done = idx in st.session_state.completed_steps
            badge = "✅" if is_done else ("🔵" if is_active else "⚪")
            st.markdown(f"**Step {idx}** {badge}\n\n*{step['title']}*")

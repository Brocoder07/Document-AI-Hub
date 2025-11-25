import streamlit as st

def render_landing_page(go_login, go_register):
    st.markdown("""
        <div style="text-align: center; margin-bottom: 3rem;">
            <h1 style="font-size: 3rem;">
                Chat with your <span style="color: #FF4B4B;">Knowledge Base</span>
            </h1>
            <p style="color: #666; font-size: 1.2rem;">
                The intelligent hub for your documents. Upload PDFs, ask questions, 
                and get AI-cited answers instantly.
            </p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🚀 Get Started", use_container_width=True, type="primary"):
                go_register()
        with c2:
            if st.button("🔐 Login", use_container_width=True):
                go_login()
    
    st.divider()
    
    # Feature Grid
    c1, c2, c3 = st.columns(3)
    c1.info("📄 **Upload Anything**\n\nPDFs, Images, Audio supported.")
    c2.info("🧠 **Smart Context**\n\nRAG pipeline retrieves exact evidence.")
    c3.info("🛡️ **Secure**\n\nRBAC and JWT authentication built-in.")
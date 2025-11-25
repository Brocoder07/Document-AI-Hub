import streamlit as st
from session_state import SessionState
from api_client import APIClient

# Components
from components.landing import render_landing_page
from components.auth import render_login, render_register
from components.documents import render_documents_page
from components.chat import render_chat_page
from components.settings import render_settings_page    # <--- NEW
from components.utilities import render_utilities_page  # <--- NEW

st.set_page_config(page_title="Document AI Hub", page_icon="🤖", layout="wide")

SessionState.initialize()
api = APIClient()

def render_dashboard():
    # Header
    c1, c2 = st.columns([8, 1])
    with c1: st.title("🤖 AI Hub")
    with c2: 
        if st.button("Logout"): SessionState.logout()
    
    st.divider()

    # Navigation
    with st.sidebar:
        st.header("Menu")
        if "nav_section" not in st.session_state:
            st.session_state.nav_section = "Chat"
        
        # Extended Menu
        selection = st.radio(
            "Go to:", 
            ["Chat", "Documents", "Utilities", "Settings"], 
            key="nav_radio"
        )
        st.session_state.nav_section = selection

    # Routing
    sec = st.session_state.nav_section
    if sec == "Chat":
        render_chat_page(api)
    elif sec == "Documents":
        render_documents_page(api)
    elif sec == "Utilities":
        render_utilities_page(api)
    elif sec == "Settings":
        render_settings_page(api)

def main():
    if SessionState.is_authenticated():
        render_dashboard()
    else:
        view = st.session_state.current_view
        if view == "landing":
            render_landing_page(
                lambda: setattr(st.session_state, 'current_view', 'login') or st.rerun(),
                lambda: setattr(st.session_state, 'current_view', 'register') or st.rerun()
            )
        elif view == "login": render_login(api)
        elif view == "register": render_register(api)

if __name__ == "__main__":
    main()
import streamlit as st
import time
from session_state import SessionState
from api_client import APIClient

# Components
from components.landing import render_landing_page
from components.auth import render_login, render_register
from components.documents import render_documents_page
from components.chat import render_chat_page
from components.utilities import render_utilities
from components.settings import render_settings_page
from components.sidebar import render_sidebar

# Import Session Manager
from utils.session_manager import get_token_from_cookie, set_token_cookie, init_cookie_manager

# Page Config
st.set_page_config(
    page_title="Document AI Hub", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize State & API
SessionState.initialize()
api = APIClient()

# --- CRITICAL FIX: Initialize Cookie Manager Once ---
# This must happen in the main flow, before any logic that uses cookies
init_cookie_manager()
# --------------------------------------------------

def render_dashboard():
    token = st.session_state.access_token
    
    if "user_profile" not in st.session_state or st.session_state.user_profile is None:
        with st.spinner("Loading profile..."):
            user_data = api.get_current_user(token)
            if "error" in user_data:
                st.error("Session expired. Please login again.")
                time.sleep(1)
                SessionState.logout()
                return
            st.session_state.user_profile = user_data
    
    user_data = st.session_state.user_profile
    selected_nav = render_sidebar(api, token, user_data)

    if selected_nav == "Chat":
        render_chat_page(api)
    elif selected_nav == "Documents":
        render_documents_page(api)
    elif selected_nav == "Utilities":
        render_utilities(api)
    elif selected_nav == "Settings":
        render_settings_page(api)

def main():
    # --- 1. SESSION RESTORATION LOGIC ---
    if not SessionState.is_authenticated():
        # Now safe to call because init_cookie_manager() ran above
        token = get_token_from_cookie()
        if token:
            user_info = api.get_current_user(token)
            if "error" not in user_info:
                SessionState.set_user(token, user_info["email"])
                st.rerun()

    # --- 2. ROUTING ---
    if SessionState.is_authenticated():
        render_dashboard()
    else:
        view = st.session_state.current_view
        if view == "landing":
            render_landing_page(
                lambda: setattr(st.session_state, 'current_view', 'login') or st.rerun(),
                lambda: setattr(st.session_state, 'current_view', 'register') or st.rerun()
            )
        elif view == "login": 
            render_login(api)
        elif view == "register": 
            render_register(api)

if __name__ == "__main__":
    main()
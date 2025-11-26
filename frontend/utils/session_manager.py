import extra_streamlit_components as stx
import streamlit as st
import time

def init_cookie_manager():
    """
    Initializes the CookieManager and stores it in session state.
    MUST be called exactly once at the start of the app.
    """
    # Create the manager with a stable key
    # This renders the component (iframe) in the frontend
    manager = stx.CookieManager(key="auth_cookie_manager")
    
    # Store reference in session state for access elsewhere
    st.session_state["cookie_manager_instance"] = manager
    
    # Small delay to ensure component sync (common fix for this library)
    time.sleep(0.05) 
    return manager

def get_manager():
    """
    Retrieves the existing manager from session state.
    If not initialized, warns the developer (should be init in main.py).
    """
    if "cookie_manager_instance" not in st.session_state:
        # Fallback: Initialize if missing (e.g., during a partial rerun or edge case)
        # But ideally, init_cookie_manager() should be called in main.py
        return init_cookie_manager()
        
    return st.session_state["cookie_manager_instance"]

def get_token_from_cookie():
    """
    Attempts to retrieve the access token from browser cookies.
    """
    cookie_manager = get_manager()
    cookies = cookie_manager.get_all()
    return cookies.get("access_token")

def set_token_cookie(token, expires_at=None):
    """
    Saves the access token to a browser cookie.
    """
    cookie_manager = get_manager()
    cookie_manager.set("access_token", token, key="set_auth_token")

def delete_token_cookie():
    """
    Removes the access token from browser cookies.
    """
    cookie_manager = get_manager()
    cookie_manager.delete("access_token", key="del_auth_token")
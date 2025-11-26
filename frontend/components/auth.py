import streamlit as st
from session_state import SessionState
from utils.session_manager import set_token_cookie
import time

def render_login(api):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("← Back", type="secondary"):
            st.session_state.current_view = "landing"
            st.rerun()

        st.markdown("## 🔐 Login")
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", type="primary")

            if submitted:
                if not email or not password:
                    st.error("Missing credentials")
                else:
                    with st.spinner("Verifying..."):
                        res = api.login(email, password)
                    
                    if "access_token" in res:
                        token = res["access_token"]
                        
                        # 1. Save to Session State (Memory)
                        SessionState.set_user(token, email)
                        
                        # 2. Save to Cookie (Disk) - Expires in 7 days
                        set_token_cookie(token)
                        
                        st.success("Logged in!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(res.get("error", "Login failed"))

def render_register(api):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("← Back", type="secondary"):
            st.session_state.current_view = "landing"
            st.rerun()
            
        st.markdown("## 📝 New Account")
        with st.form("reg_form"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            name = st.text_input("Full Name")
            
            # UPDATED: Specific Role Selection
            role = st.selectbox("I am a...", [
                "Student", 
                "Researcher", 
                "Doctor", 
                "Lawyer", 
                "Banker", 
                "Business Man", 
                "Employee"
            ])
            
            pass1 = st.text_input("Password", type="password")
            pass2 = st.text_input("Confirm Password", type="password")
            
            if st.form_submit_button("Create Account", type="primary"):
                if pass1 != pass2:
                    st.error("Passwords do not match")
                elif not username or not email:
                    st.error("Please fill in all fields")
                else:
                    with st.spinner("Registering..."):
                        # Role is passed to backend here
                        res = api.register(username, email, name, pass1, role)
                    if "error" not in res:
                        st.success("Success! Account created.")
                        time.sleep(1)
                        st.session_state.current_view = "login"
                        st.rerun()
                    else:
                        st.error(res.get("error"))
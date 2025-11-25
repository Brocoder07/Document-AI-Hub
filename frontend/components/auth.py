import streamlit as st
from session_state import SessionState

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
                        SessionState.set_user(res["access_token"], email)
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
            role = st.selectbox("Role", ["student", "researcher", "lawyer", "doctor", "banker", "employee"])
            pass1 = st.text_input("Password", type="password")
            pass2 = st.text_input("Confirm Password", type="password")
            
            if st.form_submit_button("Create Account", type="primary"):
                if pass1 != pass2:
                    st.error("Passwords do not match")
                else:
                    with st.spinner("Registering..."):
                        res = api.register(username, email, name, pass1, role)
                    if "error" not in res:
                        st.success("Success! Please login.")
                        st.session_state.current_view = "login"
                        st.rerun()
                    else:
                        st.error(res.get("error"))
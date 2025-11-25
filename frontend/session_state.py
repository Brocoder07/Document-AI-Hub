import streamlit as st

class SessionState:
    @staticmethod
    def initialize():
        if "access_token" not in st.session_state:
            st.session_state.access_token = None
        if "user_info" not in st.session_state:
            st.session_state.user_info = None
        if "current_view" not in st.session_state:
            st.session_state.current_view = "landing"
        # Chat specific state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "chat_session_id" not in st.session_state:
            st.session_state.chat_session_id = None

    @staticmethod
    def set_user(token, email):
        st.session_state.access_token = token
        st.session_state.user_info = {"email": email}
        st.session_state.current_view = "dashboard"

    @staticmethod
    def logout():
        st.session_state.access_token = None
        st.session_state.user_info = None
        st.session_state.chat_history = []
        st.session_state.chat_session_id = None
        st.session_state.current_view = "landing"
        st.rerun()

    @staticmethod
    def is_authenticated():
        return st.session_state.access_token is not None
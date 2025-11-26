import streamlit as st
import streamlit_antd_components as sac
from utils.session_manager import delete_token_cookie

def render_sidebar(api, token, user_data):
    """
    Renders the sidebar with Navigation (Top) and User Profile (Bottom).
    """
    with st.sidebar:
        # --- CSS to Push Profile to Bottom ---
        # This flex-box trick forces the last container in the sidebar to the bottom
        st.markdown(
            """
            <style>
                [data-testid="stSidebar"] > div:first-child {
                    display: flex;
                    flex-direction: column;
                    height: 100vh;
                }
                [data-testid="stSidebar"] > div:first-child > div:nth-child(2) {
                    flex-grow: 1; /* Pushes the next element (profile) to bottom */
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # --- NAVIGATION ---
        nav_selected = sac.menu([
            sac.MenuItem('Chat', icon='chat-dots'),
            sac.MenuItem('Documents', icon='file-earmark-text'), # Restored
            sac.MenuItem('Utilities', icon='tools'),
            sac.MenuItem('Settings', icon='gear'), # Restored
        ], index=0, format_func='title', size='middle', indent=30, open_index=None, key='nav_menu')

        # --- SPACER (The CSS above makes this expand) ---
        st.write("") 

        # --- USER PROFILE (Fixed at Bottom) ---
        with st.container():
            st.markdown("---")
            
            full_name = user_data.get("full_name", "User")
            email = user_data.get("email", "")
            role = user_data.get("role", "employee").capitalize()
            avatar_letter = full_name[0].upper() if full_name else "U"

            st.markdown(f"""
            <div style="
                display: flex; 
                align-items: center; 
                padding: 10px; 
                background-color: #f8f9fa; 
                border-radius: 8px;
                border: 1px solid #dee2e6;
                margin-bottom: 10px;
            ">
                <div style="
                    min-width: 35px; 
                    height: 35px; 
                    background-color: #2196F3; 
                    color: white; 
                    border-radius: 50%; 
                    display: flex; 
                    justify-content: center; 
                    align-items: center; 
                    font-weight: 600; 
                    margin-right: 10px;
                ">
                    {avatar_letter}
                </div>
                <div style="flex-grow: 1; overflow: hidden;">
                    <div style="font-size: 14px; font-weight: 600; color: #333;">{full_name}</div>
                    <div style="font-size: 11px; color: #666;">{email}</div>
                    <div style="font-size: 10px; color: #888; text-transform: uppercase; margin-top: 2px;">{role}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚪 Log out", use_container_width=True):
                delete_token_cookie()
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

    return nav_selected
import streamlit as st
from session_state import SessionState

def render_settings_page(api):
    st.title("⚙️ User Settings")
    token = st.session_state.access_token

    # Fetch User Info
    user = api.get_current_user(token)
    if "error" in user:
        st.error("Failed to load profile.")
        return

    st.subheader("My Profile")
    
    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        new_name = col1.text_input("Full Name", value=user.get("full_name", ""))
        new_email = col2.text_input("Email", value=user.get("email", ""))
        
        if st.form_submit_button("Update Profile"):
            res = api.update_user(token, {"full_name": new_name, "email": new_email})
            if "error" in res:
                st.error(res["error"])
            else:
                st.success("Profile updated!")
                st.session_state.user_info["email"] = new_email # Update local state
                st.rerun()

    st.divider()
    
    st.subheader("Danger Zone")
    if st.button("❌ Delete Account", type="primary"):
        st.warning("Are you sure? This action cannot be undone.")
        if st.button("Yes, permanently delete my account"):
            if api.delete_user(token):
                st.success("Account deleted.")
                SessionState.logout()
            else:
                st.error("Failed to delete account.")
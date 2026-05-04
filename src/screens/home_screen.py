



import streamlit as st
from src.components.header import header_home
from src.components.footer import footer_home
from src.ui.base_layout import style_base_layout, style_background_home

def home_screen():
   

    
    style_background_home()
    style_base_layout()
    header_home()

    col1, col2 = st.columns([1,1], gap="large")

    with col1:
        st.markdown("""
        <h2 style='text-align:center; font-weight:60;'>I'm Student</h2>
        """,    unsafe_allow_html=True)

        st.image("student_png.png", width=200)
        if st.button('Student Portal', type='secondary', icon=':material/arrow_outward:', icon_position='right',width='stretch'):
            st.session_state['login_type']='student'
            st.rerun()
       

    with col2:
        st.markdown("""
        <h2 style='text-align:center; font-weight:60;'>I'm Teacher</h2>
        """,    unsafe_allow_html=True)

        
        st.image("teacher_png.png", width=200)
        if st.button('Teacher Portal', type='secondary', icon=':material/arrow_outward:', icon_position='right',width='stretch'):
            st.session_state['login_type']='teacher'
            st.rerun()


    footer_home()

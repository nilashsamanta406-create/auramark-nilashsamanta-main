import streamlit as st

import base64
from pathlib import Path

def get_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def header_home():

    img_base64 = get_image_base64("logo.png")
    
    st.markdown(f"""
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; margin-bottom:30px; margin-top:30px">
            <img src='data:image/png;base64,{img_base64}' style='height:200px;' />
            <h1 style='text-align:center; color:#E0E3FF; font-weight:100; letter-spacing:4px;'>AuraMark</h1>
        </div>   
                
                """, unsafe_allow_html=True)


def header_dashboard():

    img_base64 = get_image_base64("black_logo.png")
    
    st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:center; gap:10px">
            <img src='data:image/png;base64,{img_base64}' style='height:100px;margin-bottom:20px;' />
            <h2 style='text-align:left; color:#5865F2; font-weight:100; letter-spacing:2px; '>AuraMark</h2>
        </div>   
                
                """, unsafe_allow_html=True)
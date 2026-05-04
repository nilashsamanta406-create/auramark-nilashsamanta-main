import streamlit as st
from src.database.db import enroll_student_to_subject
from src.database.config import supabase
from PIL import Image
import time


@st.dialog("Capture or upload photos")
def add_photos_dialog():

    st.write('Add classroom photos to scan for attendance')

    if 'photo_tab' not in st.session_state:
        st.session_state.photo_tab = 'camera'
    if 'added_cam_id' not in st.session_state:
        st.session_state.added_cam_id = None
    if 'added_upload_names' not in st.session_state:
        st.session_state.added_upload_names = set()

    t1, t2 = st.columns(2)

    with t1:
        type_camera = "primary" if st.session_state.photo_tab == 'camera' else 'tertiary'
        if st.button('Camera', type=type_camera, width='stretch'):
            st.session_state.photo_tab = 'camera'



    with t2:
        type_upload = "primary" if st.session_state.photo_tab == 'upload' else 'tertiary'
        if st.button('Upload photos', type=type_upload, width='stretch'):
            st.session_state.photo_tab = 'upload'

    if st.session_state.photo_tab == 'camera':
        cam_photo = st.camera_input('Take Snapshot', key='dialog_cam')
        if cam_photo:
            cam_id = cam_photo.file_id
            if cam_id != st.session_state.added_cam_id:
                st.session_state.attendance_images.append(Image.open(cam_photo))
                st.session_state.added_cam_id = cam_id
                st.toast('Photo Captured')
            


    if st.session_state.photo_tab == 'upload':
        uploaded_files = st.file_uploader( 'choose image files', type=['jpg', 'png', 'jpeg' ], accept_multiple_files=True, key='dialog_upload')

        if uploaded_files:
            added_any = False
            for f in uploaded_files:
                if f.name not in st.session_state.added_upload_names:
                    st.session_state.attendance_images.append(Image.open(f))
                    st.session_state.added_upload_names.add(f.name)
                    added_any = True
            if added_any:
                st.toast('Photos uploaded successfully')

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric('Photos added', len(st.session_state.attendance_images))
    with col2:

        if st.button('Done', type='primary', width='stretch'):
            st.session_state.show_add_photos = False  # close the flag
            st.session_state.photo_tab = 'camera'     # reset tab for next open
            st.rerun() 
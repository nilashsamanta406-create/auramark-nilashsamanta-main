import dlib
import numpy as np
import face_recognition_models
from sklearn.svm import SVC
import streamlit as st

from src.database.db import get_all_students


@st.cache_resource
def load_dlib_models():
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(
        face_recognition_models.pose_predictor_model_location()
    )
    facerec = dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )
    return detector, sp, facerec


def get_face_embeddings(image_np):
    detector, sp, facerec = load_dlib_models()

    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    elif image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]

    faces = detector(image_np, 1)
    encodings = []

    for face in faces:
        shape = sp(image_np, face)
        face_descriptor = facerec.compute_face_descriptor(image_np, shape, 1)
        encodings.append(np.array(face_descriptor))

    return encodings


def get_student_count():
    student_db = get_all_students()
    return len([s for s in student_db if s.get('face_embedding')])


@st.cache_resource
def get_trained_model(_student_count):
    X = []
    y = []

    student_db = get_all_students()

    if not student_db:
        return None

    for student in student_db:
        embedding = student.get('face_embedding')
        if embedding:
            X.append(np.array(embedding))
            y.append(student.get('student_id'))

    if len(X) == 0:
        return None

    if len(set(y)) == 1:
        return {'clf': None, 'X': X, 'y': y, 'single_student': True}

    clf = SVC(kernel='linear', probability=True, class_weight='balanced')

    try:
        clf.fit(X, y)
    except ValueError as e:
        return None

    return {'clf': clf, 'X': X, 'y': y, 'single_student': False}


def train_classifier():
    st.cache_resource.clear()
    count = get_student_count()
    model_data = get_trained_model(count)
    return bool(model_data)


def predict_attendance(class_image_np):
    encodings = get_face_embeddings(class_image_np)
    detected_students = {}

    if not encodings:
        return detected_students, [], 0

    count = get_student_count()
    model_data = get_trained_model(count)

    if not model_data:
        return detected_students, [], len(encodings)

    X_train = model_data['X']
    y_train = model_data['y']
    single_student = model_data.get('single_student', False)
    clf = model_data['clf']

    all_students = sorted(list(set(y_train)))

    DISTANCE_THRESHOLD = 0.8

    for encoding in encodings:
        # Find closest match by distance
        best_score = float('inf')
        best_id = None

        for idx, trained_id in enumerate(y_train):
            dist = np.linalg.norm(X_train[idx] - encoding)
            if dist < best_score:
                best_score = dist
                best_id = trained_id

        # ✅ Fixed: reject if distance too HIGH (was wrong logic before)
        if best_score > DISTANCE_THRESHOLD:
            continue  # unknown face, skip

        # ✅ Fixed: use best_id directly for single student
        if single_student:
            predicted_id = int(best_id)
        else:
            predicted_id = int(clf.predict([encoding])[0])

        detected_students[predicted_id] = True

    return detected_students, all_students, len(encodings)


def predict_login(image_np):
    encodings = get_face_embeddings(image_np)

    if not encodings:
        return None, "no_face"

    if len(encodings) > 1:
        return None, "multiple_faces"

    encoding = encodings[0]

    count = get_student_count()
    model_data = get_trained_model(count)

    if not model_data:
        return None, "no_model"

    clf = model_data['clf']
    X_train = model_data['X']
    y_train = model_data['y']
    single_student = model_data.get('single_student', False)

    # ✅ Defined before use
    all_students = sorted(list(set(y_train)))

    DISTANCE_THRESHOLD = 0.8
    CONFIDENCE_THRESHOLD = 0.4

    # Find closest match by distance
    best_score = float('inf')
    best_id = None

    for idx, trained_id in enumerate(y_train):
        dist = np.linalg.norm(X_train[idx] - encoding)
        if dist < best_score:
            best_score = dist
            best_id = trained_id

    # Reject unknown face
    if best_score > DISTANCE_THRESHOLD:
        return None, "unknown"

    if single_student:
        # ✅ Fixed: use best_id directly
        predicted_id = int(best_id)
    else:
        predicted_id = int(clf.predict([encoding])[0])
        proba = clf.predict_proba([encoding])[0]
        confidence = max(proba)

        if confidence < CONFIDENCE_THRESHOLD:
            return None, "unknown"

        # ✅ Both must agree
        if predicted_id != int(best_id):
            return None, "unknown"

    return predicted_id, "success"
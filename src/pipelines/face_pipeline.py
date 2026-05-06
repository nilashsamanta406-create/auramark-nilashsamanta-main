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

    # ✅ Convert to RGB if needed
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
    """Returns number of students with face embeddings — used as cache key"""
    student_db = get_all_students()
    return len([s for s in student_db if s.get('face_embedding')])


@st.cache_resource
def get_trained_model(_student_count):
    """
    Trains SVM classifier on all students with face embeddings.
    Cache invalidates automatically when student count changes.
    """
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

    # ✅ Handle single student — skip SVM
    if len(set(y)) == 1:
        return {'clf': None, 'X': X, 'y': y, 'single_student': True}

    clf = SVC(kernel='linear', probability=True, class_weight='balanced')

    try:
        clf.fit(X, y)
    except ValueError as e:
        st.error(f"Classifier training failed: {e}")
        return None

    return {'clf': clf, 'X': X, 'y': y, 'single_student': False}


def train_classifier():
    """Force clears cache and retrains model"""
    st.cache_resource.clear()
    count = get_student_count()
    model_data = get_trained_model(count)
    return bool(model_data)


def predict_attendance(class_image_np):
    """
    Used by teacher for bulk attendance from class photo.
    Returns detected students, all student IDs, and face count.
    """
    encodings = get_face_embeddings(class_image_np)
    detected_students = {}

    if not encodings:
        return detected_students, [], 0

    # ✅ Cache invalidates when new student registers
    count = get_student_count()
    model_data = get_trained_model(count)

    if not model_data:
        return detected_students, [], len(encodings)

    clf = model_data['clf']
    X_train = model_data['X']
    y_train = model_data['y']
    single_student = model_data.get('single_student', False)

    all_students = sorted(list(set(y_train)))

    # ✅ Tunable thresholds
    DISTANCE_THRESHOLD = 0.55   # increase if too strict, decrease if too lenient
    CONFIDENCE_THRESHOLD = 0.65  # increase if too strict, decrease if too lenient

    for encoding in encodings:
        # ✅ Step 1: Find closest match by distance across ALL students
        best_score = float('inf')
        best_id = None

        for idx, trained_id in enumerate(y_train):
            dist = np.linalg.norm(X_train[idx] - encoding)
            if dist < best_score:
                best_score = dist
                best_id = trained_id

        # ✅ Step 2: Reject if distance too high (unknown face)
        if best_score > DISTANCE_THRESHOLD:
            continue

        # ✅ Step 3: SVM confidence check
        if not single_student:
            predicted_id = int(clf.predict([encoding])[0])
            proba = clf.predict_proba([encoding])[0]
            confidence = max(proba)

            # ✅ Reject if not confident enough
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            # ✅ Step 4: Both SVM and distance must agree
            if predicted_id != best_id:
                continue
        else:
            predicted_id = int(all_students[0])

        detected_students[predicted_id] = True

    return detected_students, all_students, len(encodings)


def predict_login(image_np):
    """
    Used for single student face login.
    Returns (student_id, status) where status is:
    - 'success' → face matched
    - 'unknown' → face not recognized
    - 'no_face' → no face detected
    - 'multiple_faces' → more than one face
    - 'no_model' → no students registered
    """
    encodings = get_face_embeddings(image_np)

    if not encodings:
        return None, "no_face"

    if len(encodings) > 1:
        return None, "multiple_faces"

    encoding = encodings[0]

    # ✅ Cache invalidates when new student registers
    count = get_student_count()
    model_data = get_trained_model(count)

    if not model_data:
        return None, "no_model"

    clf = model_data['clf']
    X_train = model_data['X']
    y_train = model_data['y']
    single_student = model_data.get('single_student', False)

    all_students = sorted(list(set(y_train)))

    # ✅ Tunable thresholds
    DISTANCE_THRESHOLD = 0.55
    CONFIDENCE_THRESHOLD = 0.65

    # ✅ Step 1: Find closest match by distance
    best_score = float('inf')
    best_id = None

    for idx, trained_id in enumerate(y_train):
        dist = np.linalg.norm(X_train[idx] - encoding)
        if dist < best_score:
            best_score = dist
            best_id = trained_id

    # ✅ Step 2: Reject unknown face
    if best_score > DISTANCE_THRESHOLD:
        return None, "unknown"

    # ✅ Step 3: SVM confidence check
    if not single_student:
        predicted_id = int(clf.predict([encoding])[0])
        proba = clf.predict_proba([encoding])[0]
        confidence = max(proba)

        # ✅ Reject if not confident
        if confidence < CONFIDENCE_THRESHOLD:
            return None, "unknown"

        # ✅ Both must agree
        if predicted_id != best_id:
            return None, "unknown"
    else:
        all_students = sorted(list(set(y_train)))
        predicted_id = int(all_students[0])

    return predicted_id, "success"
import streamlit as st
import uuid
import copy
import time

def init_session_state():
    default_states = {
        "current_page": "Home",
        "current_step": 1,
        "completed_steps": set(),
        "dataset_name": "Blobs (Separable)",
        "noise_level": 0.05,
        "dataset_params": {"n_samples": 300, "noise_level": 0.05, "random_state": 42},
        "apply_scaling": True,
        "dataset_key": None,
        "raw_df": None,
        "X_prep": None,
        "y_true": None,
        "active_run": None,
        "kmeans_results": None,
        "em_results": None,
        "dbscan_results": None,
        "observations": [],
        "student_notes": "",
        "student_conclusion": "",
        "quiz_answers": {},
        "quiz_score": None,
        "quiz_attempted": False,
        "quiz_passed": False,
        "dbscan_noise_policy": "Exclude Noise",
        "student_name": "Student User",
        "student_id": "ML-2026-LAB",
    }
    for key, val in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = val

def add_observation(module_name: str, algorithm: str, params: dict, metrics: dict, notes: str = ""):
    run_record = {
        "run_id": str(uuid.uuid4())[:8],
        "run_index": len(st.session_state.observations) + 1,
        "timestamp": time.strftime("%H:%M:%S"),
        "module": str(module_name),
        "algorithm": str(algorithm),
        "dataset_name": st.session_state.get("dataset_name", "N/A"),
        "apply_scaling": st.session_state.get("apply_scaling", True),
        "dataset_key": st.session_state.get("dataset_key", "N/A"),
        "X_prep": copy.deepcopy(st.session_state.get("X_prep")),
        "labels": copy.deepcopy(st.session_state.active_run.get("labels")) if st.session_state.active_run else None,
        "centroids": copy.deepcopy(st.session_state.active_run.get("centroids")) if st.session_state.active_run else None,
        "params": copy.deepcopy(params),
        "metrics": copy.deepcopy(metrics),
        "student_notes": str(notes)
    }
    st.session_state.observations.append(run_record)

def clear_observations():
    st.session_state.observations = []
    st.session_state.active_run = None

import streamlit as st
import pandas as pd
import numpy as np

from core.session_manager import init_session_state, add_observation, clear_observations
from core.report_generator import generate_pdf_report, export_observations_csv, generate_auto_conclusion
from components.navbar import render_header, render_navigation
from components.progress_tracker import render_step_tracker

from lab_modules.exp8_clustering.content import AIM, OBJECTIVES, THEORY, PROCEDURE_STEPS, PRE_LAB_QUIZ, POST_LAB_QUIZ
from lab_modules.exp8_clustering.data_generator import generate_cluster_dataset, preprocess_features
from lab_modules.exp8_clustering.models import run_kmeans, run_gmm, run_dbscan
from lab_modules.exp8_clustering.visualizations import plot_cluster_results, plot_comparison_bar

st.set_page_config(page_title="ML Virtual Lab — Clustering Analysis", layout="wide", page_icon="🔬")

init_session_state()

if st.session_state.current_page in ["Home", None]:
    st.session_state.current_page = "Overview"

render_header()
render_navigation()
st.markdown("---")

if st.session_state.current_page == "Overview":
    st.title("🔬 Machine Learning Virtual Laboratory")
    st.subheader("Clustering Analysis & Quality Evaluation")
    
    st.markdown("---")
    st.subheader("🎯 Experiment Aim")
    st.info(f"**Aim:** {AIM}")
    
    st.subheader("📌 Learning Objectives")
    for obj in OBJECTIVES:
        st.markdown(f"- {obj}")
        
    st.markdown("---")
    st.subheader("📖 Theoretical Foundation")
    t1, t2, t3 = st.tabs(["K-Means Clustering", "Expectation Maximization (GMM)", "DBSCAN Density Clustering"])
    with t1:
        st.markdown(THEORY["K-Means"])
    with t2:
        st.markdown(THEORY["EM / GMM"])
    with t3:
        st.markdown(THEORY["DBSCAN"])
        
    st.markdown("---")
    if st.button("Proceed to Pre-Lab Readiness Check ➡️", type="primary", width="stretch"):
        st.session_state.current_page = "Pre-Lab"
        st.rerun()

elif st.session_state.current_page == "Pre-Lab":
    st.title("📚 Pre-Lab Readiness Check")
    st.markdown("Answer these baseline assessment questions before running experiments on the workbench.")
    
    with st.form("pre_lab_form"):
        user_selections = {}
        for q in PRE_LAB_QUIZ:
            st.markdown(f"**{q['question']}**")
            user_selections[q['id']] = st.radio(
                "Select your answer:",
                q['options'],
                index=0,
                key=f"radio_{q['id']}"
            )
            st.markdown("---")
            
        submitted = st.form_submit_button("Submit Pre-Lab Quiz")
        if submitted:
            score = 0
            for q in PRE_LAB_QUIZ:
                user_choice = user_selections[q['id']]
                correct_choice = q['options'][q['answer_idx']]
                if user_choice == correct_choice:
                    score += 1
                    st.success(f"✅ Correct! {q['explanation']}")
                elif user_choice == "Select an option...":
                    st.warning(f"⚠️ {q['question']}: Please select an option.")
                else:
                    st.error(f"❌ Incorrect. {q['explanation']}")
            st.info(f"**Pre-Lab Score: {score} / {len(PRE_LAB_QUIZ)}**")

    if st.button("Proceed to Guided Workbench Procedure ➡️", type="primary", width="stretch"):
        st.session_state.current_page = "Procedure"
        st.rerun()

elif st.session_state.current_page == "Procedure":
    st.title("🗺️ Step-by-Step Practical Procedure")
    render_step_tracker(PROCEDURE_STEPS, st.session_state.current_step)
    curr = PROCEDURE_STEPS[st.session_state.current_step - 1]
    st.subheader(f"Step {curr['step']}: {curr['title']}")
    st.info(curr['desc'])
    if st.button("Enter Experimental Workbench 🔬", type="primary", width="stretch"):
        st.session_state.current_page = "Workbench"
        st.rerun()

elif st.session_state.current_page == "Workbench":
    st.title("⚡ Virtual Experimental Workbench")
    ctrl_col, viz_col = st.columns([1, 2])
    
    with ctrl_col:
        st.subheader("⚙️ Data & Model Controls")
        ds_name = st.selectbox("Select Data Topology", ["Blobs (Separable)", "Overlapping Clusters", "Concentric Circles", "Crescent Moons", "Anisotropic (Elongated)"])
        n_samples = st.slider("Sample Count", 100, 1000, 300)
        noise_level = st.slider("Noise Level", 0.01, 0.20, 0.05, 0.01)
        scale_opt = st.checkbox("Apply Feature Scaling", value=True)
        noise_policy = st.selectbox("DBSCAN Silhouette Policy", ["Exclude Noise", "Include Noise"])
        st.session_state.dbscan_noise_policy = noise_policy
        
        current_ds_key = f"{ds_name}_{n_samples}_{noise_level}_{scale_opt}"

        if st.session_state.dataset_key != current_ds_key or st.button("Reload Dataset", width="stretch"):
            st.session_state.dataset_name = ds_name
            st.session_state.noise_level = noise_level
            st.session_state.dataset_params = {"n_samples": n_samples, "noise_level": noise_level, "random_state": 42}
            st.session_state.apply_scaling = scale_opt
            st.session_state.dataset_key = current_ds_key
            
            df_raw, X_raw, y_raw = generate_cluster_dataset(ds_name, n_samples, noise_level=noise_level, random_state=42)
            st.session_state.X_prep = preprocess_features(X_raw, scale_opt)
            st.session_state.active_run = None
            st.session_state.kmeans_results = None
            st.session_state.em_results = None
            st.session_state.dbscan_results = None

        X = st.session_state.X_prep

        st.markdown("---")
        algo_choice = st.radio("Select Algorithm to Execute", ["K-Means", "EM (GMM)", "DBSCAN"])
        
        if algo_choice == "K-Means":
            k_val = st.slider("Clusters (K)", 2, 10, 3, key="km_k")
            init_opt = st.selectbox("Initialization Method", ["k-means++", "random"], key="km_init")
            max_iter_opt = st.number_input("Max Iterations", 100, 1000, 300, key="km_iter")
            
            if st.button("Run K-Means", type="primary", width="stretch"):
                res = run_kmeans(X, n_clusters=k_val, init=init_opt, max_iter=max_iter_opt)
                res["type"] = "K-Means"
                res["X_prep"] = X
                st.session_state.kmeans_results = res
                st.session_state.active_run = res
                add_observation("Clustering Lab", "K-Means", res["params"], {"silhouette": res["silhouette"], "silhouette_msg": res["silhouette_msg"], "n_clusters": res["n_clusters"], "n_noise": 0})
                st.rerun()
                
        elif algo_choice == "EM (GMM)":
            n_comp = st.slider("Components", 2, 10, 3, key="gmm_c")
            cov_type = st.selectbox("Covariance Type", ["full", "tied", "diag", "spherical"], key="gmm_cov")
            if st.button("Run EM (GMM)", type="primary", width="stretch"):
                res = run_gmm(X, n_components=n_comp, covariance_type=cov_type)
                res["type"] = "EM (GMM)"
                res["X_prep"] = X
                res["centroids"] = None  # Force clear K-Means centroids
                st.session_state.em_results = res
                st.session_state.active_run = res
                add_observation("Clustering Lab", "EM (GMM)", res["params"], {"silhouette": res["silhouette"], "silhouette_msg": res["silhouette_msg"], "n_clusters": res["n_clusters"], "n_noise": 0})
                st.rerun()
                
        elif algo_choice == "DBSCAN":
            eps_val = st.slider("eps (Radius)", 0.1, 2.0, 0.4, 0.05, key="db_eps")
            min_pts = st.slider("Min Samples", 2, 20, 5, key="db_min")
            if st.button("Run DBSCAN", type="primary", width="stretch"):
                ex_noise = (noise_policy == "Exclude Noise")
                res = run_dbscan(X, eps=eps_val, min_samples=min_pts, exclude_noise=ex_noise)
                res["type"] = "DBSCAN"
                res["X_prep"] = X
                res["centroids"] = None  # Force clear K-Means centroids
                st.session_state.dbscan_results = res
                st.session_state.active_run = res
                add_observation("Clustering Lab", "DBSCAN", res["params"], {"silhouette": res["silhouette"], "silhouette_msg": res["silhouette_msg"], "n_clusters": res["n_clusters"], "n_noise": res["n_noise"]})
                st.rerun()

    with viz_col:
        st.subheader("📊 Visualization Workspace")
        st.markdown(f"**Current Dataset:** `{st.session_state.dataset_name}` ({len(X)} samples)")
        vtab1, vtab2, vtab3 = st.tabs(["Active Plot", "Algorithm Comparison", "Observation Log"])
        
        with vtab1:
            active = st.session_state.get("active_run")
            if active and len(active.get("labels", [])) == len(active.get("X_prep", X)):
                plot_X = active.get("X_prep", X)
                c_centers = active.get("centroids") if active["type"] == "K-Means" else None
                title_str = f"Result: {active['type']} on {st.session_state.dataset_name} (Clusters: {active['n_clusters']})"
                
                fig = plot_cluster_results(plot_X, active["labels"], centroids=c_centers, title=title_str)
                
                # Dynamic key forces Streamlit DOM container destruction & redraw
                plot_hash_key = f"plot_{active['type']}_{hash(tuple(active['labels']))}"
                st.plotly_chart(fig, width="stretch", key=plot_hash_key)
                
                m1, m2 = st.columns(2)
                m1.metric("Clusters Detected", active["n_clusters"])
                m2.metric("Silhouette Score", f"{active['silhouette']:.4f}" if active['silhouette'] is not None else "N/A (Invalid)")
                
                if active["silhouette"] is None:
                    st.warning(f"⚠️ Metric Status: {active['silhouette_msg']}")
                if active.get("n_noise") is not None and active["n_noise"] > 0:
                    st.info(f"DBSCAN Noise Points: **{active['n_noise']}** ({active['noise_ratio']}%)")
            else:
                st.info("👈 Select parameters and click **Run Algorithm** to execute.")

        with vtab2:
            st.markdown("#### Comparative Silhouette Metrics")
            comp_dict = {
                "K-Means": st.session_state.get("kmeans_results"),
                "EM (GMM)": st.session_state.get("em_results"),
                "DBSCAN": st.session_state.get("dbscan_results")
            }
            fig_bar = plot_comparison_bar(comp_dict)
            st.plotly_chart(fig_bar, width="stretch", key="bar_comparison_chart")

        with vtab3:
            st.markdown("#### In-Lab Observations & Historical Inspection")
            if st.session_state.observations:
                for o in st.session_state.observations:
                    c_info, c_act = st.columns([4, 1])
                    with c_info:
                        sil_val = f"{o['metrics'].get('silhouette'):.4f}" if isinstance(o['metrics'].get('silhouette'), float) else "N/A"
                        st.markdown(f"**Run {o['run_index']} ({o['algorithm']})** | Dataset: `{o.get('dataset_name')}` | Params: `{o['params']}` | **Silhouette: {sil_val}**")
                    with c_act:
                        if st.button("Inspect Plot", key=f"inspect_{o['run_id']}", width="stretch"):
                            st.session_state.active_run = {
                                "type": o["algorithm"],
                                "X_prep": o["X_prep"],
                                "labels": o["labels"],
                                "centroids": o["centroids"] if o["algorithm"] == "K-Means" else None,
                                "n_clusters": o["metrics"].get("n_clusters"),
                                "n_noise": o["metrics"].get("n_noise", 0),
                                "noise_ratio": 0.0,
                                "silhouette": o["metrics"].get("silhouette"),
                                "silhouette_msg": "Historical Run Inspection"
                            }
                            st.rerun()
                
                st.markdown("---")
                c_dl, c_clr = st.columns(2)
                with c_dl:
                    csv_bytes = export_observations_csv(st.session_state.observations)
                    st.download_button(
                        "📥 Export Observations CSV",
                        data=csv_bytes,
                        file_name=f"Clustering_Observations_{st.session_state.student_id}.csv",
                        mime="text/csv",
                        width="stretch"
                    )
                with c_clr:
                    if st.button("Clear Observation Log", width="stretch"):
                        clear_observations()
                        st.rerun()
            else:
                st.write("No recorded observations.")
                
            st.markdown("---")
            if st.button("Proceed to Post-Lab Assessment & Report ➡️", type="primary", width="stretch"):
                st.session_state.current_page = "Analysis"
                st.rerun()

elif st.session_state.current_page == "Analysis":
    st.title("📊 Post-Lab Assessment & Findings")
    
    with st.form("post_lab_form"):
        st.markdown("### Post-Lab Assessment Quiz")
        post_ans = {}
        for q in POST_LAB_QUIZ:
            st.markdown(f"**{q['question']}**")
            post_ans[q['id']] = st.radio("Select answer:", q['options'], index=0, key=f"post_{q['id']}")
            st.markdown("---")
            
        submitted = st.form_submit_button("Submit Post-Lab Assessment")
        if submitted:
            score = 0
            for q in POST_LAB_QUIZ:
                user_choice = post_ans[q['id']]
                correct_choice = q['options'][q['answer_idx']]
                if user_choice == correct_choice:
                    score += 1
            final_pct = round((score / len(POST_LAB_QUIZ)) * 100, 1)
            st.session_state.quiz_score = final_pct
            st.session_state.quiz_attempted = True
            st.success(f"✅ Assessment Submitted! Score: {final_pct}%")

    st.markdown("### Experimental Findings & Conclusion")
    auto_conc = generate_auto_conclusion(st.session_state.observations)
    user_conc = st.text_area(
        "Edit or provide your experimental conclusion:",
        value=st.session_state.student_conclusion if st.session_state.student_conclusion else auto_conc.replace("<b>", "").replace("</b>", "").replace("<br/>", "\n"),
        height=160
    )
    if st.button("Save Conclusion"):
        st.session_state.student_conclusion = user_conc
        st.success("Conclusion saved to report generator!")
        
    st.markdown("---")
    if st.button("Proceed to Generate PDF Report 📄", type="primary", width="stretch"):
        st.session_state.current_page = "Report"
        st.rerun()

elif st.session_state.current_page == "Report":
    st.title("📄 Practical Report Generation")
    st.session_state.student_name = st.text_input("Student Name", value=st.session_state.student_name)
    st.session_state.student_id = st.text_input("Student ID", value=st.session_state.student_id)
    
    session_data = {
        "student_name": st.session_state.student_name,
        "student_id": st.session_state.student_id,
        "dataset_name": st.session_state.dataset_name,
        "dataset_params": st.session_state.dataset_params,
        "apply_scaling": st.session_state.apply_scaling,
        "dbscan_noise_policy": st.session_state.dbscan_noise_policy,
        "observations": st.session_state.observations,
        "student_conclusion": st.session_state.student_conclusion,
        "quiz_score": st.session_state.quiz_score,
        "quiz_attempted": st.session_state.quiz_attempted
    }
    
    pdf_bytes = generate_pdf_report(session_data)
    st.download_button(
        "Download Formal PDF Report",
        data=pdf_bytes,
        file_name=f"Clustering_Report_{st.session_state.student_id}.pdf",
        mime="application/pdf",
        type="primary"
    )

import io
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_pdf_report(session_data: dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=18, leading=22, textColor=colors.HexColor('#1E3A8A'), alignment=1)
    h2_style = ParagraphStyle('Heading2_Custom', parent=styles['Heading2'], fontSize=12, leading=15, textColor=colors.HexColor('#1E3A8A'), spaceBefore=8, spaceAfter=4)
    body_style = ParagraphStyle('Body_Custom', parent=styles['Normal'], fontSize=8.5, leading=11, textColor=colors.HexColor('#333333'))
    hdr_cell_style = ParagraphStyle('HdrCell', parent=body_style, fontSize=8.5, leading=10, textColor=colors.white, fontName='Helvetica-Bold', alignment=1)
    body_cell_style = ParagraphStyle('BodyCell', parent=body_style, fontSize=8, leading=10, alignment=1)
    param_cell_style = ParagraphStyle('ParamCell', parent=body_style, fontSize=7.5, leading=9, alignment=0)
    
    elements = []
    
    elements.append(Paragraph("ML VIRTUAL LAB - PRACTICAL REPORT", title_style))
    elements.append(Spacer(1, 6))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1E3A8A'), spaceAfter=10))
    
    is_quiz_attempted = session_data.get("quiz_attempted", False)
    quiz_score = session_data.get("quiz_score")
    has_runs = len(session_data.get("observations", [])) > 0
    completion_status = "Completed" if (is_quiz_attempted and has_runs) else "In Progress (Partial)"
    
    meta_data = [
        [Paragraph(f"<b>Student Name:</b> {session_data.get('student_name', 'N/A')}", body_style),
         Paragraph(f"<b>Student ID:</b> {session_data.get('student_id', 'N/A')}", body_style)],
        [Paragraph("<b>Experiment:</b> Clustering & Quality Evaluation", body_style),
         Paragraph(f"<b>Completion Status:</b> {completion_status}", body_style)]
    ]
    t_meta = Table(meta_data, colWidths=[270, 270])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F3F4F6')),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E5E7EB'))
    ]))
    elements.append(t_meta)
    elements.append(Spacer(1, 10))
    
    elements.append(Paragraph("1. Experiment Aim & Objectives", h2_style))
    aim_text = "<b>Aim:</b> Perform clustering using K-Means, Expectation Maximization (EM/GMM), and DBSCAN algorithms and evaluate clustering quality using Silhouette Score and visualization techniques."
    elements.append(Paragraph(aim_text, body_style))
    elements.append(Spacer(1, 8))
    
    elements.append(Paragraph("2. Dataset Configuration", h2_style))
    ds_name = session_data.get("dataset_name", "N/A")
    ds_params = str(session_data.get("dataset_params", {}))
    scaling = "Applied (StandardScaler)" if session_data.get("apply_scaling") else "Raw Features"
    policy = session_data.get("dbscan_noise_policy", "Exclude Noise")
    elements.append(Paragraph(f"<b>Selected Dataset:</b> {ds_name}<br/><b>Parameters:</b> {ds_params}<br/><b>Feature Scaling:</b> {scaling}<br/><b>DBSCAN Silhouette Policy:</b> {policy}", body_style))
    elements.append(Spacer(1, 8))
    
    elements.append(Paragraph("3. Summary of Experimental Runs", h2_style))
    obs_list = session_data.get("observations", [])
    
    if obs_list:
        headers = [
            Paragraph("Run", hdr_cell_style),
            Paragraph("Algorithm", hdr_cell_style),
            Paragraph("Parameters", hdr_cell_style),
            Paragraph("Clusters", hdr_cell_style),
            Paragraph("Noise Pts", hdr_cell_style),
            Paragraph("Silhouette Score", hdr_cell_style)
        ]
        table_data = [headers]
        
        for idx, obs in enumerate(obs_list, 1):
            p_str = ", ".join([f"{k}={v}" for k, v in obs["params"].items()])
            sil = obs["metrics"].get("silhouette")
            sil_str = f"{sil:.4f}" if isinstance(sil, float) else "N/A (Invalid)"
            c_count = str(obs["metrics"].get("n_clusters", "N/A"))
            n_count = str(obs["metrics"].get("n_noise", 0))
            
            row = [
                Paragraph(f"Run {idx}", body_cell_style),
                Paragraph(str(obs.get("algorithm", "N/A")), body_cell_style),
                Paragraph(p_str, param_cell_style),
                Paragraph(c_count, body_cell_style),
                Paragraph(n_count, body_cell_style),
                Paragraph(sil_str, body_cell_style)
            ]
            table_data.append(row)
            
        t_obs = Table(table_data, colWidths=[40, 75, 195, 55, 65, 110], repeatRows=1)
        t_obs.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#D1D5DB')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F9FAFB')]),
            ('PADDING', (0,0), (-1,-1), 4),
        ]))
        elements.append(t_obs)
    else:
        elements.append(Paragraph("<i>No experimental runs were recorded in the observation log.</i>", body_style))
        
    elements.append(Spacer(1, 10))
    
    elements.append(Paragraph("4. Student Findings & Conclusion", h2_style))
    user_conc = session_data.get("student_conclusion", "").strip()
    if not user_conc or len(user_conc) < 15:
        user_conc = generate_auto_conclusion(obs_list)
    elements.append(Paragraph(user_conc, body_style))
    elements.append(Spacer(1, 10))
    
    elements.append(Paragraph("5. Knowledge Check Results", h2_style))
    if is_quiz_attempted and quiz_score is not None:
        score_str = f"{quiz_score}% (Attempted)"
    else:
        score_str = "Quiz Not Attempted"
    elements.append(Paragraph(f"<b>Post-Lab Assessment Score:</b> {score_str}", body_style))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

def generate_auto_conclusion(obs_list: list) -> str:
    if not obs_list:
        return "No experimental runs recorded. Empirical execution is required to assess algorithm suitability."
    
    kmeans_runs = [o for o in obs_list if o.get("algorithm") == "K-Means"]
    dbscan_runs = [o for o in obs_list if o.get("algorithm") == "DBSCAN"]
    
    km_best = max([r["metrics"]["silhouette"] for r in kmeans_runs if r["metrics"]["silhouette"] is not None], default=None)
    db_best = max([r["metrics"]["silhouette"] for r in dbscan_runs if r["metrics"]["silhouette"] is not None], default=None)
    
    conc = "<b>Evidence-Based Experimental Summary:</b><br/>"
    if km_best:
        conc += f"• <b>K-Means:</b> Achieved a peak Silhouette Score of {km_best:.4f}. It partitions space using centroid Voronoi cells, assuming convex cluster geometry.<br/>"
    if db_best:
        conc += f"• <b>DBSCAN:</b> Achieved a peak Silhouette Score of {db_best:.4f} on non-noise samples, successfully identifying arbitrary density shapes.<br/>"
    conc += "• <b>Evaluation Insight:</b> Silhouette Score measures cluster compactness versus separation. However, score alone does not establish universal superiority on non-spherical geometries."
    return conc

def export_observations_csv(obs_list: list) -> str:
    rows = []
    for obs in obs_list:
        row = {
            "Run_ID": obs.get("run_id"),
            "Run_Number": f"Run {obs.get('run_index')}",
            "Algorithm": obs.get("algorithm"),
            "Dataset": obs.get("dataset_name", "N/A"),
            "Scaled": obs.get("apply_scaling", True),
            "Parameters": ", ".join([f"{k}={v}" for k, v in obs.get("params", {}).items()]),
            "Clusters_Detected": obs["metrics"].get("n_clusters"),
            "Noise_Points": obs["metrics"].get("n_noise", 0),
            "Silhouette_Score": obs["metrics"].get("silhouette", "N/A"),
            "Metric_Status": obs["metrics"].get("silhouette_msg", "N/A"),
            "Notes": obs.get("student_notes", "")
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)

"""
app.py
------
Streamlit dashboard for the AI-Based Building Damage Assessment and
Disaster Recovery Recommendation System.

Run with:
    streamlit run app.py
"""

import io

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from predict import load_model, predict_damage
from recommendation import get_recommendation
from report_generator import generate_pdf_report

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Building Damage Assessment",
    page_icon="🏚️",
    layout="centered",
)

# Map safety colors to Streamlit alert types
STATUS_TO_ALERT = {
    "green": "success",
    "yellow": "warning",
    "orange": "warning",
    "red": "error",
    "gray": "info",
}


@st.cache_resource(show_spinner=False)
def get_cached_model():
    """
    Loads the trained model once and caches it across reruns/sessions
    so repeated predictions don't reload the model from disk.
    """
    return load_model()


def render_header():
    st.title("🏚️ AI Building Damage Assessment")
    st.markdown(
        "Upload a post-disaster building image to automatically assess "
        "**damage severity** and receive **recovery recommendations**."
    )
    st.divider()


def render_upload_section():
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a building image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
    )
    return uploaded_file


def safe_load_image(uploaded_file):
    """
    Attempts to open the uploaded file as an image.
    Returns (image, error_message). If loading fails, image is None
    and error_message explains why -- so the app can fail gracefully
    instead of crashing on a corrupt/invalid upload.
    """
    try:
        image = Image.open(uploaded_file)
        image.load()  # force-read to catch truncated/corrupt files early
        return image, None
    except UnidentifiedImageError:
        return None, "The uploaded file is not a valid image. Please upload a JPG or PNG."
    except Exception as e:  # noqa: BLE001 - want to surface any load failure to the user
        return None, f"Could not read the uploaded image: {e}"


def render_results(damage_class, confidence, all_probs, rec_info):
    st.subheader("📊 Results")

    col1, col2 = st.columns(2)
    col1.metric("Predicted Damage Class", damage_class)
    col2.metric("Confidence Score", f"{confidence:.2f}%")

    # Safety status banner, color-coded by severity
    alert_fn = getattr(st, STATUS_TO_ALERT.get(rec_info["color"], "info"))
    alert_fn(f"**Safety Status:** {rec_info['safety_status']}")

    # Probability breakdown across all classes
    with st.expander("View full class probability breakdown"):
        probs_df = pd.DataFrame(
            {"Damage Class": list(all_probs.keys()), "Probability (%)": list(all_probs.values())}
        ).sort_values("Probability (%)", ascending=False)
        st.bar_chart(probs_df.set_index("Damage Class"))
        st.dataframe(probs_df, use_container_width=True, hide_index=True)

    # Recovery recommendations
    st.subheader("🛠️ Recovery Recommendation")
    for rec in rec_info["recommendations"]:
        st.markdown(f"- {rec}")


def render_download_section(image, damage_class, confidence, rec_info):
    st.subheader("📄 Download Assessment Report")
    pdf_bytes = generate_pdf_report(
        image=image,
        damage_class=damage_class,
        confidence=confidence,
        safety_status=rec_info["safety_status"],
        recommendations=rec_info["recommendations"],
    )
    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name="building_damage_assessment_report.pdf",
        mime="application/pdf",
    )


def main():
    render_header()
    uploaded_file = render_upload_section()

    if uploaded_file is None:
        st.info("Please upload an image to begin.")
        return

    image, error = safe_load_image(uploaded_file)
    if error:
        st.error(error)
        return

    st.subheader("🖼️ Uploaded Image")
    st.image(image, caption="Uploaded building image", use_container_width=True)

    if st.button("🔍 Predict Damage", type="primary"):
        with st.spinner("Loading model and analyzing image..."):
            try:
                model = get_cached_model()
            except FileNotFoundError as e:
                st.error(str(e))
                return

            try:
                damage_class, confidence, all_probs = predict_damage(model, image)
            except Exception as e:  # noqa: BLE001
                st.error(f"Prediction failed: {e}")
                return

        rec_info = get_recommendation(damage_class)

        # Persist results in session state so the download button
        # (which triggers a rerun) doesn't lose the prediction.
        st.session_state["result"] = {
            "damage_class": damage_class,
            "confidence": confidence,
            "all_probs": all_probs,
            "rec_info": rec_info,
        }

    # Render results if we have them in session state (survives reruns
    # triggered by the download button click)
    if "result" in st.session_state:
        res = st.session_state["result"]
        render_results(res["damage_class"], res["confidence"], res["all_probs"], res["rec_info"])
        render_download_section(image, res["damage_class"], res["confidence"], res["rec_info"])


if __name__ == "__main__":
    main()

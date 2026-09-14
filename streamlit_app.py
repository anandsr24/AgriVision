"""
AgriVision — Streamlit Chatbot & Vision Frontend.

Communicates exclusively over HTTP with the FastAPI backend (/health, /detect, /reason).
Does NOT load the RT-DETR model locally or replicate reasoning logic.
"""

import io
import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 30

SUGGESTED_QUESTIONS = [
    "How many weeds are present?",
    "How many crops are present?",
    "Are there more crops or weeds?",
    "Are there any weeds?",
    "What objects are present?",
    "What is the crop to weed ratio?",
]

# Set Streamlit page config
st.set_page_config(
    page_title="AgriVision — AI Crop & Weed Chatbot",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Helper Functions: HTTP API Clients & Visualization
# -----------------------------------------------------------------------------
def check_api_health(api_url: str) -> Optional[Dict[str, Any]]:
    """Pings the FastAPI /health endpoint."""
    try:
        res = requests.get(f"{api_url}/health", timeout=5)
        if res.status_code == 200:
            return res.json()
        elif res.status_code == 503:
            return res.json()
        return None
    except Exception:
        return None


def call_detect_api(api_url: str, image_bytes: bytes, filename: str = "image.jpg") -> Dict[str, Any]:
    """Sends image to /detect endpoint."""
    files = {"image": (filename, image_bytes, "image/jpeg")}
    res = requests.post(f"{api_url}/detect", files=files, timeout=REQUEST_TIMEOUT_SECONDS)
    res.raise_for_status()
    return res.json()


def call_reason_api(api_url: str, image_bytes: bytes, question: str, filename: str = "image.jpg") -> Dict[str, Any]:
    """Sends image and question to /reason endpoint."""
    files = {"image": (filename, image_bytes, "image/jpeg")}
    data = {"question": question}
    res = requests.post(f"{api_url}/reason", files=files, data=data, timeout=REQUEST_TIMEOUT_SECONDS)
    res.raise_for_status()
    return res.json()


def annotate_image_with_bboxes(image: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
    """Draws real bounding boxes received from the /detect API onto the image."""
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    # Color scheme: Crop (Emerald Green), Weed (Amber / Orange-Red)
    color_map = {
        "crop": (34, 197, 94),     # #22c55e
        "weed": (245, 158, 11),    # #f59e0b
    }

    img_w, img_h = annotated.size
    line_width = max(2, int(min(img_w, img_h) / 300))

    for det in detections:
        cls_name = det.get("class", "unknown").lower()
        conf = det.get("confidence", 0.0)
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        color = color_map.get(cls_name, (59, 130, 246))

        # Draw bounding rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Label tag
        label_text = f"{cls_name.capitalize()} {conf:.2f}"
        
        # Approximate text bounding box for tag background
        text_w = len(label_text) * 8
        text_h = 14
        tag_y1 = max(0, y1 - text_h - 4)
        tag_y2 = tag_y1 + text_h + 4
        tag_x2 = min(img_w, x1 + text_w + 8)

        draw.rectangle([x1, tag_y1, tag_x2, tag_y2], fill=color)
        draw.text((x1 + 4, tag_y1 + 2), label_text, fill=(255, 255, 255))

    return annotated


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_image_bytes" not in st.session_state:
    st.session_state.uploaded_image_bytes = None

if "uploaded_image_name" not in st.session_state:
    st.session_state.uploaded_image_name = None

if "detection_data" not in st.session_state:
    st.session_state.detection_data = None

if "active_question" not in st.session_state:
    st.session_state.active_question = None


# -----------------------------------------------------------------------------
# Sidebar: System Status, Configuration, & Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🌾 AgriVision Hub")
    st.markdown("---")

    # API Configuration
    api_url = st.text_input("FastAPI Base URL", value=DEFAULT_API_URL, help="URL of the running AgriVision FastAPI server.")

    # Live Health Check
    health_info = check_api_health(api_url)
    if health_info and health_info.get("status") == "ok":
        st.success("🟢 **Backend Connected**")
        st.markdown(f"**Model:** `{health_info.get('model', 'RT-DETR-L')}`")
        st.markdown(f"**Classes:** `{', '.join(health_info.get('classes', []))}`")
        st.markdown(f"**Default Confidence:** `{health_info.get('confidence_threshold', 0.50)}`")
    elif health_info and health_info.get("status") == "degraded":
        st.warning("🟡 **Backend Degraded (Model Loading/Fallback)**")
    else:
        st.error("🔴 **Backend Offline**")
        st.caption(f"Make sure FastAPI is running:  \n`uvicorn src.main:app --host 0.0.0.0 --port 8000`")

    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("AgriVision — RT-DETR-L Agricultural Vision & Deterministic Reasoning API")


# -----------------------------------------------------------------------------
# Main Content Area
# -----------------------------------------------------------------------------
st.title("AgriVision")
st.markdown("#### AI-powered Crop & Weed Detection and Reasoning Assistant")
st.caption("Upload an image of crops or weeds and ask questions about object counts, comparisons, presence, or ratios.")

# Layout: Two columns (Left: Image Studio & Detections, Right: Conversational Chatbot)
col_left, col_right = st.columns([1, 1.2], gap="large")

# -----------------------------------------------------------------------------
# LEFT COLUMN: Image Preview & Detection Details
# -----------------------------------------------------------------------------
with col_left:
    st.subheader("🖼️ Field Image & Detection")

    # Prominent Main File Uploader
    main_uploaded_file = st.file_uploader(
        "📤 Upload a crop/weed photo (or drag & drop here)",
        type=["jpg", "jpeg", "png", "webp"],
        key="main_file_uploader",
    )
    if main_uploaded_file is not None:
        file_bytes = main_uploaded_file.getvalue()
        if file_bytes != st.session_state.uploaded_image_bytes:
            st.session_state.uploaded_image_bytes = file_bytes
            st.session_state.uploaded_image_name = main_uploaded_file.name
            st.session_state.detection_data = None
            st.session_state.messages = []

    # Quick Sample Selector in Main Area
    sample_options = {
    "Or choose a built-in test sample...": None,
    "Sample 1: Mixed Plot (2 Crops, 1 Weed)": "assets/samples/agri_0_1630.jpeg",
    "Sample 2: Balanced Plot (1 Crop, 1 Weed)": "assets/samples/agri_0_7985.jpeg",
    "Sample 3: Dense Crops (Crops Only)": "assets/samples/agri_0_1028.jpeg",
    "Sample 4: Weed Infestation (5 Weeds)": "assets/samples/agri_0_126.jpeg",
    "Sample 5: Weed Spotting (3 Weeds)": "assets/samples/agri_0_1017.jpeg",
    }
    main_sample = st.selectbox("Quick Test Samples:", list(sample_options.keys()), key="main_sample_selector")
    if main_sample and sample_options[main_sample]:
        sample_path = sample_options[main_sample]
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                s_bytes = f.read()
            if s_bytes != st.session_state.uploaded_image_bytes:
                st.session_state.uploaded_image_bytes = s_bytes
                st.session_state.uploaded_image_name = os.path.basename(sample_path)
                st.session_state.detection_data = None
                st.session_state.messages = []

    st.markdown("---")

    if st.session_state.uploaded_image_bytes:
        try:
            pil_image = Image.open(io.BytesIO(st.session_state.uploaded_image_bytes)).convert("RGB")

            # Fetch detections from /detect if not cached
            if st.session_state.detection_data is None and health_info:
                with st.spinner("Running RT-DETR-L object detector..."):
                    try:
                        det_result = call_detect_api(
                            api_url,
                            st.session_state.uploaded_image_bytes,
                            filename=st.session_state.uploaded_image_name or "image.jpg",
                        )
                        st.session_state.detection_data = det_result
                    except Exception as e:
                        st.error(f"Detection API error: {e}")

            # Draw image with bounding boxes if detections are available
            det_data = st.session_state.detection_data
            if det_data and det_data.get("detections"):
                annotated_img = annotate_image_with_bboxes(pil_image, det_data["detections"])
                st.image(
                    annotated_img,
                    caption=f"{st.session_state.uploaded_image_name} (Annotated with RT-DETR-L)",
                    use_container_width=True,
                )
            else:
                st.image(
                    pil_image,
                    caption=f"{st.session_state.uploaded_image_name} (Raw Image)",
                    use_container_width=True,
                )

            # Expandable Detection Details
            with st.expander("🔍 View detection details", expanded=False):
                if det_data:
                    counts = det_data.get("counts", {})
                    crop_count = counts.get("crop", 0)
                    weed_count = counts.get("weed", 0)
                    total_count = det_data.get("total_count", crop_count + weed_count)
                    latency = det_data.get("inference_time_ms", 0.0)

                    # Quick Metric Badges
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("🌾 Crops", crop_count)
                    m2.metric("🌿 Weeds", weed_count)
                    m3.metric("📦 Total", total_count)
                    m4.metric("⚡ Latency", f"{latency:.1f} ms")

                    # Detection Table
                    detections = det_data.get("detections", [])
                    if detections:
                        st.markdown("**Structured Detections:**")
                        table_rows = [
                            {
                                "Class": d.get("class", "").capitalize(),
                                "Confidence": f"{d.get('confidence', 0.0):.4f}",
                                "BBox [x1, y1, x2, y2]": str(d.get("bbox", [])),
                            }
                            for d in detections
                        ]
                        st.dataframe(table_rows, use_container_width=True)
                    else:
                        st.info("No objects met the current confidence threshold.")
                else:
                    st.info("Detection data not yet loaded. Connect to the backend to run detection.")

        except Exception as err:
            st.error(f"Failed to display image: {err}")
    else:
        st.info("👆 Please upload a crop/weed photo above or select a test sample to begin.")


# -----------------------------------------------------------------------------
# RIGHT COLUMN: Chatbot & Reasoning Engine
# -----------------------------------------------------------------------------
with col_right:
    st.subheader("Reasoning Assistant")

    # Suggested Questions Bar
    st.markdown("**Suggested questions:**")
    pill_cols = st.columns(2)
    for idx, q in enumerate(SUGGESTED_QUESTIONS):
        col_idx = idx % 2
        with pill_cols[col_idx]:
            if st.button(q, key=f"sug_{idx}", use_container_width=True):
                st.session_state.active_question = q

    st.markdown("---")

    # Display Chat History
    if not st.session_state.messages:
        # Welcome intro bubble
        with st.chat_message("assistant", avatar="🌾"):
            st.markdown(
                "Hello! I am **AgriVision's Reasoning Assistant**.\n\n"
                "Once you upload a field image, ask me any questions about:\n"
                "- **Counts:** *How many weeds / crops are present?*\n"
                "- **Comparisons:** *Are there more crops or weeds?*\n"
                "- **Presence:** *Are there any weeds visible?*\n"
                "- **Ratios:** *What is the crop to weed ratio?*\n\n"
                "🛡️ *Safety Guardrails: If an unsupported agronomic question is asked (disease, fertilizer, spraying), "
                "I will return **Insufficient Information**.*"
            )

    for msg in st.session_state.messages:
        role = msg["role"]
        avatar = "👤" if role == "user" else "🌾"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])
            if "confidence" in msg:
                conf = msg["confidence"].lower()
                if conf == "high":
                    st.caption("🟢 **Confidence:** `HIGH`")
                elif conf == "insufficient":
                    st.caption("🟡 **Confidence:** `INSUFFICIENT` (Safety Guardrail Triggered)")
                else:
                    st.caption(f"⚪ **Confidence:** `{conf.upper()}`")

            if "evidence" in msg and msg["evidence"]:
                with st.expander("📊 Evidence Details", expanded=False):
                    st.json(msg["evidence"])

    # Handle Question Input
    user_prompt = st.chat_input("Ask a question about the image...")

    # If suggested prompt button was clicked, use it
    if st.session_state.active_question:
        user_prompt = st.session_state.active_question
        st.session_state.active_question = None

    if user_prompt:
        clean_prompt = user_prompt.strip()
        if not clean_prompt:
            st.warning("Please enter a non-empty question.")
        elif not st.session_state.uploaded_image_bytes:
            st.warning("⚠️ Please upload a crop/weed image or select a sample first.")
        else:
            # Append User message
            st.session_state.messages.append({"role": "user", "content": clean_prompt})

            # Display immediately in chat
            with st.chat_message("user", avatar="👤"):
                st.markdown(clean_prompt)

            # Send to /reason API
            with st.chat_message("assistant", avatar="🌾"):
                with st.spinner("AgriVision reasoning engine is analyzing..."):
                    try:
                        reason_result = call_reason_api(
                            api_url=api_url,
                            image_bytes=st.session_state.uploaded_image_bytes,
                            question=clean_prompt,
                            filename=st.session_state.uploaded_image_name or "image.jpg",
                        )

                        answer = reason_result.get("answer", "No answer provided.")
                        confidence = reason_result.get("confidence", "insufficient")
                        evidence = reason_result.get("evidence", {})

                        st.markdown(answer)
                        if confidence.lower() == "high":
                            st.caption("🟢 **Confidence:** `HIGH`")
                        elif confidence.lower() == "insufficient":
                            st.caption("🟡 **Confidence:** `INSUFFICIENT` (Safety Guardrail Triggered)")
                        else:
                            st.caption(f"⚪ **Confidence:** `{confidence.upper()}`")

                        if evidence:
                            with st.expander("📊 Evidence Details", expanded=False):
                                st.json(evidence)

                        # Save to session history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "confidence": confidence,
                            "evidence": evidence,
                        })

                    except requests.exceptions.ConnectionError:
                        err_msg = f"⚠️ Could not connect to FastAPI at `{api_url}`. Please verify that the server is running."
                        st.error(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": err_msg})
                    except requests.exceptions.Timeout:
                        err_msg = "⏱️ Request timed out while waiting for the reasoning engine."
                        st.error(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": err_msg})
                    except requests.exceptions.HTTPError as http_err:
                        # Extract error detail if available
                        detail = "HTTP Error"
                        try:
                            detail = http_err.response.json().get("detail", str(http_err))
                        except Exception:
                            detail = str(http_err)
                        err_msg = f"❌ API Error ({http_err.response.status_code}): {detail}"
                        st.error(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": err_msg})
                    except Exception as general_err:
                        err_msg = f"❌ An unexpected error occurred: {general_err}"
                        st.error(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": err_msg})

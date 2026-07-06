"""
report_generator.py
--------------------
Generates a downloadable PDF assessment report containing:
    - Uploaded building image
    - Predicted damage class + confidence
    - Safety status
    - Recovery recommendations
    - Date and time of assessment

Uses ReportLab (pure-Python, no external binaries needed).
"""

import io
from datetime import datetime
from typing import List

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
)


def generate_pdf_report(
    image: Image.Image,
    damage_class: str,
    confidence: float,
    safety_status: str,
    recommendations: List[str],
) -> bytes:
    """
    Builds a PDF report in-memory and returns its raw bytes, ready
    to be served via st.download_button.

    Args:
        image: the uploaded PIL image
        damage_class: predicted damage class (e.g. "Major Damage")
        confidence: prediction confidence (0-100)
        safety_status: e.g. "Restricted / Unsafe for Normal Use"
        recommendations: list of recommended actions

    Returns:
        bytes: the PDF file content
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], fontSize=20, spaceAfter=6
    )
    heading_style = ParagraphStyle(
        "HeadingStyle", parent=styles["Heading2"], spaceBefore=12, spaceAfter=6
    )
    normal_style = styles["Normal"]

    story = []

    # --- Title ---
    story.append(Paragraph("AI Building Damage Assessment Report", title_style))
    story.append(
        Paragraph(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            normal_style,
        )
    )
    story.append(Spacer(1, 0.5 * cm))

    # --- Uploaded Image ---
    story.append(Paragraph("Uploaded Building Image", heading_style))
    img_buffer = io.BytesIO()
    # Save a resized copy so the PDF stays a reasonable size
    display_img = image.convert("RGB").copy()
    display_img.thumbnail((400, 400))
    display_img.save(img_buffer, format="JPEG")
    img_buffer.seek(0)
    story.append(RLImage(img_buffer, width=8 * cm, height=8 * cm * display_img.height / display_img.width))
    story.append(Spacer(1, 0.5 * cm))

    # --- Prediction Results Table ---
    story.append(Paragraph("Assessment Results", heading_style))
    result_data = [
        ["Field", "Value"],
        ["Predicted Damage Class", damage_class],
        ["Confidence Score", f"{confidence:.2f}%"],
        ["Safety Status", safety_status],
    ]
    result_table = Table(result_data, colWidths=[6 * cm, 8 * cm])
    result_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(result_table)
    story.append(Spacer(1, 0.5 * cm))

    # --- Recovery Recommendations ---
    story.append(Paragraph("Recovery Recommendations", heading_style))
    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", normal_style))

    story.append(Spacer(1, 1 * cm))
    story.append(
        Paragraph(
            "<i>Disclaimer: This is an AI-generated preliminary assessment for "
            "prototype/demo purposes only. It does not replace a certified "
            "structural engineer's inspection.</i>",
            normal_style,
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

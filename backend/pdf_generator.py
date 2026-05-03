"""
Hospital-style PDF reports using ReportLab.
"""
from __future__ import annotations

import os
from io import BytesIO
from datetime import datetime
from html import escape as esc

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


def build_patient_report_pdf(
    patient_name: str,
    age,
    gender: str,
    phone: str,
    address: str,
    result: str,
    confidence: float,
    description: str,
    suggestions: list,
    original_image_abs_path: str,
    gradcam_image_abs_path: str,
    report_code: str | None = None,
) -> BytesIO:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
        title=f"X-Ray Report — {patient_name}",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=14,
        textColor=colors.HexColor("#0c4a6e"),
    )
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=12,
        spaceBefore=10,
        spaceAfter=6,
        textColor=colors.HexColor("#0369a1"),
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
    )

    banner = Table(
        [
            [
                Paragraph(
                    "<b><font color='white'>AI RADIOLOGY REPORT</font></b><br/>"
                    "<font size='9' color='#e0f2fe'>Pneumonia Detection · DenseNet121 + SVM · Grad-CAM</font>",
                    ParagraphStyle(
                        "Ban",
                        parent=styles["Normal"],
                        fontSize=14,
                        leading=18,
                        alignment=1,
                    ),
                )
            ]
        ],
        colWidths=[15 * cm],
    )
    banner.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0369a1")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 14),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#0c4a6e")),
            ]
        )
    )

    story = []
    story.append(banner)
    story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph("Clinical AI Report — Chest X-ray Analysis", title_style))
    story.append(
        Paragraph(
            f"<b>Pipeline:</b> DenseNet121 feature extraction + SVM classification · Grad-CAM visualization",
            body,
        )
    )
    story.append(
        Paragraph(
            f"<b>Generated:</b> {esc(datetime.now().strftime('%Y-%m-%d %H:%M'))}",
            body,
        )
    )
    story.append(Spacer(1, 0.4 * cm))

    pdata = [
        ["Report ID", esc(report_code or "—")],
        ["Patient name", esc(patient_name or "—")],
        ["Age", esc(str(age)) if age is not None else "—"],
        ["Gender", esc(gender or "—")],
        ["Phone", esc(phone or "—")],
        ["Address", esc(address or "—")],
        ["AI result", esc(result or "—")],
        ["Confidence", esc(f"{confidence}%")],
    ]
    tbl = Table(pdata, colWidths=[4 * cm, 11 * cm])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e0f2fe")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(tbl)
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("Original chest X-ray", h2))
    if original_image_abs_path and os.path.isfile(original_image_abs_path):
        try:
            story.append(RLImage(original_image_abs_path, width=14 * cm, height=10.5 * cm))
        except Exception:
            story.append(Paragraph("(Image could not be embedded.)", body))
    else:
        story.append(Paragraph("(No image.)", body))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Grad-CAM explanation map", h2))
    if gradcam_image_abs_path and os.path.isfile(gradcam_image_abs_path):
        try:
            story.append(RLImage(gradcam_image_abs_path, width=14 * cm, height=10.5 * cm))
        except Exception:
            story.append(Paragraph("(Grad-CAM could not be embedded.)", body))
    else:
        story.append(Paragraph("(No Grad-CAM.)", body))

    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Clinical summary (AI — not a diagnosis)", h2))
    story.append(
        Paragraph(esc(description or "—").replace("\n", "<br/>"), body)
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(Paragraph("Recommendations", h2))
    lines = " • ".join(esc(s) for s in (suggestions or [])) or "—"
    story.append(Paragraph(lines, body))
    story.append(Spacer(1, 0.5 * cm))
    story.append(
        Paragraph(
            "<i>Educational AI (DenseNet121 + SVM). Not a substitute for a licensed clinician.</i>",
            body,
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer

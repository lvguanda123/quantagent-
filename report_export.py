"""
PDF report exporter for the academic (高校版) variant.

Builds a self-contained PDF from a single analysis ``full_response`` dict
(the same shape that ``web_interface.analyze()`` returns and that
``history_store.save_record`` persists as JSON). The PDF includes:

  1. Cover line (asset, timeframe, run time, consensus score, decision)
  2. Core view + decision badge
  3. K-line image (decoded from base64 ``pattern_chart``)
  4. Trend image (decoded from base64 ``trend_chart``)
  5. Optional equity-curve image (if the caller passed ``equity_chart``)
  6. Consensus & conflicts block
  7. The three analyst reports (Indicator / Pattern / Trend)
  8. Final decision JSON

We use ``reportlab`` because it has the most reliable CJK font handling on
Windows (the desktop exe target) and bundles no native dependencies.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    REPORTLAB_AVAILABLE = True
except ImportError:  # pragma: no cover - reportlab is an optional dependency
    REPORTLAB_AVAILABLE = False


# Register a CJK-capable CID font once. STSong-Light ships with reportlab
# and renders Simplified Chinese on every supported platform — no font
# install required on the user's machine.
_CJK_FONT_NAME = "STSong-Light"
if REPORTLAB_AVAILABLE:
    try:
        pdfmetrics.registerFont(UnicodeCIDFont(_CJK_FONT_NAME))
    except Exception:  # pragma: no cover - font already registered or unavailable
        pass


def _decode_chart(b64: Optional[str]) -> Optional[io.BytesIO]:
    """Decode a base64 PNG string into an in-memory file handle, or None."""
    if not b64 or not isinstance(b64, str):
        return None
    try:
        raw = base64.b64decode(b64)
    except (ValueError, TypeError):
        return None
    if not raw:
        return None
    return io.BytesIO(raw)


def _truncate(text: Any, limit: int = 4000) -> str:
    """Stringify ``text`` and cap it to ``limit`` characters."""
    if text is None:
        return ""
    s = str(text).strip()
    if len(s) > limit:
        s = s[:limit] + "\n…(已截断)"
    return s


def _format_metrics(metrics: Optional[Dict[str, Any]]) -> str:
    if not metrics:
        return ""
    try:
        total = float(metrics.get("total_return", 0.0)) * 100
        annual = float(metrics.get("annual_return", 0.0)) * 100
        max_dd = float(metrics.get("max_drawdown", 0.0)) * 100
        win = float(metrics.get("win_rate", 0.0)) * 100
        n = int(metrics.get("trade_count", 0))
    except (TypeError, ValueError):
        return ""
    return (
        f"总收益 {total:.2f}% · 年化 {annual:.2f}% · 最大回撤 {max_dd:.2f}% · "
        f"胜率 {win:.1f}% · 交易 {n} 笔"
    )


def build_analysis_pdf(
    results: Dict[str, Any],
    *,
    equity_chart: Optional[str] = None,
    backtest_metrics: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Render the analysis report PDF and return it as raw bytes.

    Parameters
    ----------
    results : dict
        The full_response dict from ``analyze()`` or loaded from history.
    equity_chart : str, optional
        Base64-encoded PNG of the backtest equity curve. Pass an empty
        string or ``None`` to omit.
    backtest_metrics : dict, optional
        Optional metrics dict from ``backtest.run_simple_backtest``.

    Returns
    -------
    bytes
        A complete PDF document.

    Raises
    ------
    RuntimeError
        If reportlab is not installed.
    """
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError(
            "reportlab is required for PDF export. Install it with `pip install reportlab`."
        )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="QuantAgent 分析报告",
        author="QuantAgent",
    )

    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "BodyCJK",
        parent=styles["BodyText"],
        fontName=_CJK_FONT_NAME,
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#1A1712"),
    )
    h1_style = ParagraphStyle(
        "H1CJK",
        parent=styles["Heading1"],
        fontName=_CJK_FONT_NAME,
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17110A"),
    )
    h2_style = ParagraphStyle(
        "H2CJK",
        parent=styles["Heading2"],
        fontName=_CJK_FONT_NAME,
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#17110A"),
    )
    h3_style = ParagraphStyle(
        "H3CJK",
        parent=styles["Heading3"],
        fontName=_CJK_FONT_NAME,
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#C8A044"),
    )

    story: List[Any] = []

    # --- Cover line ---
    asset = results.get("asset_name") or "—"
    timeframe = results.get("timeframe") or "—"
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    consensus = float(results.get("consensus_score", 0.5) or 0.5) * 100
    decision = (results.get("decision") or results.get("decision_direction") or "—").strip() or "—"
    data_len = results.get("data_length") or "—"

    story.append(Paragraph("QuantAgent 量化分析报告", h1_style))
    story.append(Spacer(1, 0.4 * cm))
    cover_table = Table(
        [
            ["标的", asset, "周期", timeframe],
            ["数据点", str(data_len), "共识评分", f"{consensus:.1f}%"],
            ["决策", decision, "生成时间", run_time],
        ],
        colWidths=[2.5 * cm, 6 * cm, 2.5 * cm, 6 * cm],
    )
    cover_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), _CJK_FONT_NAME),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#FBF8F0")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#FBF8F0")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(cover_table)
    story.append(Spacer(1, 0.6 * cm))

    # --- Core view ---
    core_view = _truncate(results.get("core_view") or "暂无核心观点", 200)
    story.append(Paragraph("核心观点", h2_style))
    story.append(Paragraph(core_view, body_style))
    story.append(Spacer(1, 0.4 * cm))

    # --- K-line chart ---
    kline_png = _decode_chart(results.get("pattern_chart"))
    if kline_png is not None:
        story.append(Paragraph("K线 / 形态分析", h2_style))
        try:
            img = Image(kline_png, width=16 * cm, height=9 * cm, kind="proportional")
            story.append(img)
            story.append(Spacer(1, 0.4 * cm))
        except Exception:
            story.append(Paragraph("（K线图渲染失败）", body_style))

    # --- Trend chart ---
    trend_png = _decode_chart(results.get("trend_chart"))
    if trend_png is not None:
        story.append(Paragraph("趋势分析", h2_style))
        try:
            img = Image(trend_png, width=16 * cm, height=8 * cm, kind="proportional")
            story.append(img)
            story.append(Spacer(1, 0.4 * cm))
        except Exception:
            story.append(Paragraph("（趋势图渲染失败）", body_style))

    # --- Optional equity curve ---
    equity_png = _decode_chart(equity_chart)
    if equity_png is not None:
        story.append(PageBreak())
        story.append(Paragraph("回测资金曲线", h2_style))
        try:
            img = Image(equity_png, width=16 * cm, height=8 * cm, kind="proportional")
            story.append(img)
        except Exception:
            story.append(Paragraph("（资金曲线渲染失败）", body_style))
        metrics_text = _format_metrics(backtest_metrics)
        if metrics_text:
            story.append(Spacer(1, 0.3 * cm))
            story.append(Paragraph(metrics_text, body_style))
        story.append(Spacer(1, 0.4 * cm))

    # --- Consensus & conflicts ---
    story.append(PageBreak())
    story.append(Paragraph("共识校验 (Cross-Checker)", h2_style))
    key_points = _truncate(results.get("key_points_summary") or "无", 600)
    conflicts = _truncate(results.get("conflicts") or "无", 600)
    story.append(Paragraph(f"<b>共识要点：</b>{key_points}", body_style))
    story.append(Paragraph(f"<b>冲突点：</b>{conflicts}", body_style))
    story.append(Paragraph(
        f"<b>风险收益比：</b>"
        f"{results.get('risk_reward_ratio', '—')} · "
        f"<b>预测周期：</b>{results.get('forecast_horizon', '—')}",
        body_style,
    ))
    story.append(Spacer(1, 0.4 * cm))

    # --- Three reports ---
    for label, key in [
        ("技术指标报告", "indicator_report"),
        ("形态报告", "pattern_report"),
        ("趋势报告", "trend_report"),
    ]:
        body = _truncate(results.get(key), 4000)
        if not body:
            continue
        story.append(Paragraph(label, h3_style))
        # Convert newlines into <br/> so the Paragraph preserves them
        body_para = body.replace("\n", "<br/>")
        story.append(Paragraph(body_para, body_style))
        story.append(Spacer(1, 0.3 * cm))

    # --- Final decision JSON ---
    final_decision = _truncate(results.get("final_trade_decision"), 4000)
    if final_decision:
        story.append(PageBreak())
        story.append(Paragraph("最终决策（原始 JSON）", h2_style))
        # Escape HTML to avoid injection issues
        escaped = (
            final_decision
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )
        story.append(Paragraph(f"<font name='Courier' size='9'>{escaped}</font>", body_style))

    doc.build(story)
    return buffer.getvalue()

import os
import glob
import json
import logging
from typing import Optional
import pandas as pd
import streamlit as st

from config import settings
from eval.schemas import EvalReport
from eval.run_eval import evaluate_pipeline

logger = logging.getLogger(__name__)


def load_latest_report(dir_path: Optional[str] = None) -> Optional[EvalReport]:
    """Load the most recently generated EvalReport from the eval results directory."""
    path = dir_path or settings.EVAL_RESULTS_PATH
    if not os.path.exists(path):
        return None

    report_files = sorted(glob.glob(os.path.join(path, "eval_*.json")))
    if not report_files:
        return None

    latest_file = report_files[-1]
    try:
        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return EvalReport(**data)
    except Exception as e:
        logger.error(f"Error loading report from {latest_file}: {e}")
        return None


def render_markdown_report(report: EvalReport) -> str:
    """Format an EvalReport into a clean Markdown document."""
    lines = [
        f"# RAG Evaluation Report: {report.timestamp}",
        "",
        "## Overall Aggregate Metrics",
        f"- **Total Questions Evaluated:** {report.total_questions}",
        f"- **Hit Rate @ k:** {report.avg_hit_rate:.1%}",
        f"- **Recall @ k:** {report.avg_recall_at_k:.1%}",
        f"- **Mean Reciprocal Rank (MRR):** {report.avg_mrr:.3f}",
        f"- **Context Precision:** {report.avg_context_precision:.1%}",
        f"- **Faithfulness (Groundedness):** {report.avg_faithfulness:.1%}",
        f"- **Answer Relevance:** {report.avg_relevance:.1%}",
        f"- **Correctness:** {report.avg_correctness:.1%}",
        f"- **Citation Accuracy:** {report.citation_accuracy:.1%}",
        f"- **Average Latency:** {report.avg_latency_ms:.1f} ms",
        "",
    ]

    # Category breakdown table
    rows = []
    for r in report.records:
        rows.append({
            "Category": r.category,
            "Hit Rate": r.retrieval_score.hit,
            "Faithfulness": r.generation_score.faithfulness.score,
            "Relevance": r.generation_score.answer_relevance.score,
            "Correctness": r.generation_score.correctness.score,
            "Passed": r.passed,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        cat_df = df.groupby("Category").mean().round(3)
        lines.append("## Category Breakdown")
        lines.append(cat_df.to_markdown())
        lines.append("")

    if report.failures:
        lines.append(f"## Below Threshold Items ({len(report.failures)})")
        for fail in report.failures:
            lines.append(f"### Question: `{fail.question}`")
            lines.append(f"- **Category:** `{fail.category}`")
            lines.append(f"- **Generated Answer:** {fail.answer}")
            lines.append(f"- **Faithfulness Reasoning:** {fail.generation_score.faithfulness.reasoning}")
            lines.append(f"- **Correctness Reasoning:** {fail.generation_score.correctness.reasoning}")
            lines.append("")
    else:
        lines.append("## Below Threshold Items\nNone! All items passed evaluation thresholds.")

    return "\n".join(lines)


def render_streamlit_eval_tab():
    """Render the interactive Evaluation Dashboard tab within the Streamlit UI."""
    st.header("📊 RAG Evaluation Dashboard")
    st.caption("Benchmark retrieval precision, faithfulness, answer relevance, and correctness using LLM-as-judge.")

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        if st.button("🚀 Re-run Full Evaluation", use_container_width=True):
            if not settings.OPENAI_API_KEY:
                st.error("Please configure OPENAI_API_KEY in your .env file to run evaluation.")
            else:
                progress_bar = st.progress(0, text="Starting evaluation...")
                status_text = st.empty()

                def update_progress(current, total, msg):
                    progress = float(current / total)
                    progress_bar.progress(progress, text=f"[{current}/{total}] {msg}")

                try:
                    with st.spinner("Executing evaluation pipeline against golden dataset..."):
                        new_report = evaluate_pipeline(progress_callback=update_progress)
                        progress_bar.progress(1.0, text="Evaluation complete!")
                        st.success(f"Evaluation complete! Saved report: `eval_{new_report.timestamp}.json`")
                        st.rerun()
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

    report = load_latest_report()
    if not report:
        st.info("No evaluation runs found yet. Click 'Re-run Full Evaluation' to benchmark the pipeline.")
        return

    st.markdown(f"**Latest Run Timestamp:** `{report.timestamp}` | **Total Questions:** `{report.total_questions}`")

    # Metric KPI Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Hit Rate @ k", f"{report.avg_hit_rate:.1%}")
    m2.metric("Faithfulness", f"{report.avg_faithfulness:.1%}")
    m3.metric("Relevance", f"{report.avg_relevance:.1%}")
    m4.metric("Correctness", f"{report.avg_correctness:.1%}")
    m5.metric("Avg Latency", f"{report.avg_latency_ms:.0f} ms")

    st.markdown("---")

    # Category Chart
    st.subheader("Performance by Category")
    table_rows = []
    for r in report.records:
        table_rows.append({
            "ID": r.id,
            "Category": r.category,
            "Question": r.question,
            "Hit Rate": 1.0 if r.retrieval_score.hit else 0.0,
            "Faithfulness": r.generation_score.faithfulness.score,
            "Relevance": r.generation_score.answer_relevance.score,
            "Correctness": r.generation_score.correctness.score,
            "Passed": "✅ Pass" if r.passed else "❌ Flagged",
            "Answer": r.answer,
            "Faithfulness Note": r.generation_score.faithfulness.reasoning,
            "Correctness Note": r.generation_score.correctness.reasoning,
        })
    df = pd.DataFrame(table_rows)

    if not df.empty:
        cat_chart_df = df.groupby("Category")[["Hit Rate", "Faithfulness", "Relevance", "Correctness"]].mean()
        st.bar_chart(cat_chart_df)

    st.markdown("---")

    # Filterable question table
    st.subheader("Detailed Evaluation Records")
    category_filter = st.selectbox(
        "Filter by Category",
        ["All"] + sorted(list(set(r.category for r in report.records))),
    )

    filtered_df = df if category_filter == "All" else df[df["Category"] == category_filter]
    st.dataframe(
        filtered_df[["ID", "Category", "Question", "Hit Rate", "Faithfulness", "Relevance", "Correctness", "Passed"]],
        use_container_width=True,
    )

    # Drill-down into individual records
    st.subheader("🔍 Record Drill-down & Judge Reasoning")
    for r in report.records:
        if category_filter != "All" and r.category != category_filter:
            continue
        status_icon = "✅" if r.passed else "⚠️"
        with st.expander(f"{status_icon} [{r.id}] ({r.category}) {r.question}"):
            st.markdown(f"**Refined Query:** `{r.refined_prompt}`")
            st.markdown(f"**Assistant Answer:**\n{r.answer}")
            st.markdown(f"**Sources:** {', '.join(r.sources)}")
            st.markdown(
                f"- **Faithfulness ({r.generation_score.faithfulness.score:.1f}):** {r.generation_score.faithfulness.reasoning}\n"
                f"- **Answer Relevance ({r.generation_score.answer_relevance.score:.1f}):** {r.generation_score.answer_relevance.reasoning}\n"
                f"- **Correctness ({r.generation_score.correctness.score:.1f}):** {r.generation_score.correctness.reasoning}\n"
                f"- **Citation Check:** {'Passed' if r.generation_score.citation_correct else 'Missing'}\n"
                f"- **Latency:** {r.latency_ms.get('total', 0):.1f} ms"
            )

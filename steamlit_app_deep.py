"""
Math Task Batch Processor – Streamlit UI
Uses math_task_analyzer for core logic.
"""

import io
import time
import zipfile

import pandas as pd
import streamlit as st
from openai import OpenAI

from math_task_analyzer import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_MAX_TOKENS_PER_BATCH,
    DEFAULT_MAX_WORKERS,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_EXTRA_COLUMNS_TEXT,
    parse_extra_columns,
    build_columns_prompt,
    get_result_columns,
    get_encoder,
    load_txt_files,
    load_tsv_file,
    create_batches,
    compute_token_stats,
    combine_results,
    run_batches,
)

# ─────────────────────────────────────────────
# Helper for debug ZIP
# ─────────────────────────────────────────────


def build_debug_zip(
    df_in: pd.DataFrame,
    batches: list[tuple[int, str]],
    df_result: pd.DataFrame,
    warnings: list[str],
    system_prompt: str,
    columns_prompt: str,
    raw_results: list[tuple[int, str]],
) -> bytes:
    """Return bytes of a ZIP archive with debug information."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # 0) Input tasks with batch index
        input_lines = ["batch_idx\trow_idx\ttask_text"]
        for batch_idx, tsv_text in batches:
            lines = tsv_text.strip().splitlines()
            if len(lines) < 2:
                continue
            for line in lines[1:]:
                parts = line.split("\t")
                if len(parts) >= 2:
                    row_idx, task = parts[0], parts[1]
                    input_lines.append(f"{batch_idx}\t{row_idx}\t{task}")
        zf.writestr("0_input_tasks_with_batch.tsv", "\n".join(input_lines))

        # 1) Result TSV (final merged)
        if "df_merged" in st.session_state:
            result_tsv = st.session_state.df_merged.to_csv(sep="\t", index=False)
        else:
            result_tsv = ""
        zf.writestr("1_result.tsv", result_tsv)

        # 2) Warnings
        zf.writestr("2_warnings.txt", "\n".join(warnings))

        # 3) Prompts
        zf.writestr("3_system_prompt.txt", system_prompt)
        zf.writestr("3_columns_prompt.txt", columns_prompt)

        # 4) Raw LLM responses – one file per batch
        for batch_idx, resp in raw_results:
            zf.writestr(f"4_raw_llm_response_batch_{batch_idx}.txt", resp)

    return zip_buffer.getvalue()


# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Math Task Analyzer",
    page_icon="🧮",
    layout="wide",
)

st.title("🧮 Math Task Batch Analyzer")
st.caption("Upload .txt files → batch-send to LLM → structured DataFrame")

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input("API Key", type="password", placeholder="sk-…")
    base_url = st.text_input("API Base URL", value=DEFAULT_BASE_URL)
    model = st.text_input("Model", value=DEFAULT_MODEL)

    st.divider()
    st.subheader("Prompt parts")

    system_prompt = st.text_area(
        "① System prompt (role + rules)",
        value=DEFAULT_SYSTEM_PROMPT,
        height=220,
    )

    st.markdown(
        "**② Extra columns** (one per line: `name — definition`)\n"
        "*`row_idx` and `Текст задачи` are always first — no need to list them.*"
    )
    extra_cols_text = st.text_area(
        "Extra columns",
        value=DEFAULT_EXTRA_COLUMNS_TEXT,
        height=300,
        label_visibility="collapsed",
    )

    extra_cols = parse_extra_columns(extra_cols_text)
    columns_prompt = build_columns_prompt(extra_cols)
    result_columns = get_result_columns(extra_cols)

    if extra_cols:
        with st.expander(
            f"Parsed columns ({len(result_columns)} total)", expanded=False
        ):
            for i, name in enumerate(result_columns, 1):
                st.markdown(f"`{i}.` {name}")
    else:
        st.warning(
            "No extra columns parsed — only row_idx and Текст задачи will be expected."
        )

    st.divider()
    st.subheader("Batch settings")

    max_tokens_batch = st.number_input(
        "Max input tokens/batch",
        min_value=500,
        max_value=320000,
        value=DEFAULT_MAX_TOKENS_PER_BATCH,
        step=500,
    )

    auto_response_tokens = st.checkbox(
        "Auto-size response tokens per batch (recommended)",
        value=True,
        help=(
            "Since the LLM echoes back the task text plus a few short "
            "column fills, output size scales with input size. With this "
            "on, each batch gets its own cap computed from its own input "
            "tokens + a per-row overhead for the extra columns, instead "
            "of one flat number that may be too small for big batches "
            "(truncation) or wastefully large for small ones."
        ),
    )
    if auto_response_tokens:
        max_response_tokens = "auto"
        st.caption(
            "Response cap = batch input tokens + (rows × extra columns × ~12 "
            "tokens) × 1.25 safety margin."
        )
    else:
        max_response_tokens = st.number_input(
            "Max response tokens",
            min_value=256,
            max_value=320000,
            value=8000,
            step=256,
        )
    temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.05)
    max_workers = st.number_input(
        "Parallel workers",
        min_value=1,
        max_value=20,
        value=DEFAULT_MAX_WORKERS,
    )

# ── File upload ───────────────────────────────
input_mode = st.radio(
    "Input source",
    options=["📄 .txt files", "📊 TSV file (columns: file, text)"],
    horizontal=True,
)

df_in = None

if input_mode == "📄 .txt files":
    uploaded = st.file_uploader(
        "Upload .txt task files (one task per line)",
        type=["txt"],
        accept_multiple_files=True,
    )

    if not uploaded:
        st.info("👆 Upload one or more `.txt` files to get started.")
        st.stop()

    df_in = load_txt_files(uploaded)
    st.success(f"Loaded **{len(df_in)}** tasks from **{len(uploaded)}** file(s).")

else:
    uploaded_tsv = st.file_uploader(
        "Upload a TSV file with columns 'file' and 'text'",
        type=["tsv"],
        accept_multiple_files=False,
    )

    if not uploaded_tsv:
        st.info(
            "👆 Upload a `.tsv` file with a **`file`** column (source name) "
            "and a **`text`** column (task text)."
        )
        st.stop()

    try:
        df_in = load_tsv_file(uploaded_tsv)
    except ValueError as e:
        st.error(f"❌ {e}")
        st.stop()

    if df_in.empty:
        st.warning("The TSV was parsed but contained no non-empty rows.")
        st.stop()

    st.success(
        f"Loaded **{len(df_in)}** tasks from TSV "
        f"(**{df_in['source_file'].nunique()}** distinct file value(s))."
    )

# Show token and batch statistics using current batch settings
encoder = get_encoder(model)
total_tokens, num_batches = compute_token_stats(df_in, encoder, int(max_tokens_batch))
st.metric("Total input tokens", f"{total_tokens:,}")
st.metric("Number of batches", f"{num_batches}")

with st.expander("Preview input tasks", expanded=False):
    st.dataframe(df_in, use_container_width=True)

st.divider()

# ── Session state initialization ─────────────
if "run_done" not in st.session_state:
    st.session_state.run_done = False
if "df_in" not in st.session_state:
    st.session_state.df_in = df_in
if "df_result" not in st.session_state:
    st.session_state.df_result = pd.DataFrame()
if "raw_results" not in st.session_state:
    st.session_state.raw_results = []
if "warnings" not in st.session_state:
    st.session_state.warnings = []
if "df_merged" not in st.session_state:
    st.session_state.df_merged = pd.DataFrame()
if "batches" not in st.session_state:
    st.session_state.batches = []
if "system_prompt_saved" not in st.session_state:
    st.session_state.system_prompt_saved = system_prompt
if "columns_prompt_saved" not in st.session_state:
    st.session_state.columns_prompt_saved = columns_prompt
if "extra_cols_text_saved" not in st.session_state:
    st.session_state.extra_cols_text_saved = extra_cols_text
if "result_columns_saved" not in st.session_state:
    st.session_state.result_columns_saved = result_columns

# ── Primary run button ───────────────────────
if st.button("🚀 Send to LLM", type="primary", disabled=not api_key):
    client = OpenAI(api_key=api_key, base_url=base_url)

    st.subheader("Processing…")
    progress = st.progress(0.0)
    status = st.empty()
    t0 = time.time()

    encoder = get_encoder(model)
    batches = create_batches(df_in, encoder, int(max_tokens_batch))

    def update_progress(done, total):
        progress.progress(done / total)
        status.text(f"Completed {done}/{total} batches…")

    raw_results = run_batches(
        client=client,
        batches=batches,
        system_prompt=system_prompt,
        columns_prompt=columns_prompt,
        model=model,
        temperature=temperature,
        max_response_tokens=(
            "auto" if max_response_tokens == "auto" else int(max_response_tokens)
        ),
        max_workers=int(max_workers),
        progress_callback=update_progress,
        encoder=encoder,
        num_extra_columns=len(extra_cols),
    )

    elapsed = time.time() - t0
    status.text(f"Done in {elapsed:.1f}s")

    # Use result_columns (list of column names)
    df_result, warnings = combine_results(raw_results, result_columns)

    # Merge input with results
    if not df_result.empty and "row_idx" in df_result.columns:
        df_result["row_idx"] = pd.to_numeric(df_result["row_idx"], errors="coerce")
    df_merged = pd.merge(
        df_in, df_result, on="row_idx", how="left", suffixes=("_in", "")
    )
    if "task_text_in" in df_merged.columns and result_columns[1] in df_merged.columns:
        df_merged = df_merged.drop(columns=["task_text_in"], errors="ignore")

    # Store everything in session_state
    st.session_state.run_done = True
    st.session_state.df_in = df_in
    st.session_state.df_result = df_result
    st.session_state.raw_results = raw_results
    st.session_state.warnings = warnings
    st.session_state.df_merged = df_merged
    st.session_state.batches = batches
    st.session_state.system_prompt_saved = system_prompt
    st.session_state.columns_prompt_saved = columns_prompt
    st.session_state.extra_cols_text_saved = extra_cols_text
    st.session_state.result_columns_saved = result_columns

    st.rerun()

# ── Show results if a run has been performed ──
if st.session_state.run_done:
    df_merged = st.session_state.df_merged
    df_result = st.session_state.df_result
    warnings = st.session_state.warnings
    raw_results = st.session_state.raw_results
    df_in = st.session_state.df_in
    batches = st.session_state.batches
    system_prompt_saved = st.session_state.system_prompt_saved
    columns_prompt_saved = st.session_state.columns_prompt_saved
    result_columns_saved = st.session_state.result_columns_saved

    if warnings:
        with st.expander(f"⚠️ {len(warnings)} parsing warning(s)", expanded=True):
            for w in warnings:
                st.warning(w)

    st.divider()
    st.subheader("📊 Results")

    if df_merged.empty:
        st.error(
            "No valid results were returned. Check warnings above or the raw responses below."
        )
    else:
        st.dataframe(df_merged, use_container_width=True)

        # Stats – check second column (Текст задачи) for filled count
        check_col = (
            result_columns_saved[1]
            if len(result_columns_saved) > 1
            else result_columns_saved[0]
        )
        filled = (
            df_merged[check_col].notna().sum() if check_col in df_merged.columns else 0
        )
        missing = len(df_in) - filled
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Input tasks", len(df_in))
        c2.metric("Batches sent", len(batches))
        c3.metric("Rows returned", filled)
        c4.metric("Missing rows", missing)

        # Re-run missed tasks
        missed_row_idxs = (
            set(df_in["row_idx"]) - set(df_result["row_idx"])
            if not df_result.empty and "row_idx" in df_result.columns
            else set(df_in["row_idx"])
        )
        if missing > 0 and missed_row_idxs:
            if st.button("🔄 Re-run missed tasks", type="secondary"):
                df_missed = df_in[df_in["row_idx"].isin(missed_row_idxs)].copy()
                st.info(f"Re-running {len(df_missed)} missed tasks...")

                client = OpenAI(api_key=api_key, base_url=base_url)
                encoder = get_encoder(model)
                # Continue numbering after the original batches so re-run
                # batches get fresh, non-colliding indices (no overwriting
                # batch 0, 1, 2… from the original run in warnings/debug ZIP).
                next_batch_idx = max((idx for idx, _ in batches), default=-1) + 1
                missed_batches = create_batches(
                    df_missed, encoder, int(max_tokens_batch), start_idx=next_batch_idx
                )

                if missed_batches:
                    progress = st.progress(0.0)
                    status = st.empty()

                    def update_missed(done, total):
                        progress.progress(done / total)
                        status.text(f"Re-running: {done}/{total} batches…")

                    missed_raw = run_batches(
                        client=client,
                        batches=missed_batches,
                        system_prompt=system_prompt_saved,
                        columns_prompt=columns_prompt_saved,
                        model=model,
                        temperature=temperature,
                        max_response_tokens=(
                            "auto"
                            if max_response_tokens == "auto"
                            else int(max_response_tokens)
                        ),
                        max_workers=int(max_workers),
                        progress_callback=update_missed,
                        encoder=encoder,
                        num_extra_columns=max(len(result_columns_saved) - 2, 0),
                    )
                    # Use result_columns_saved (list of column names)
                    missed_df, missed_warnings = combine_results(
                        missed_raw, result_columns_saved
                    )

                    if not missed_df.empty and "row_idx" in missed_df.columns:
                        missed_df["row_idx"] = pd.to_numeric(
                            missed_df["row_idx"], errors="coerce"
                        )
                        combined_result = pd.concat(
                            [df_result, missed_df], ignore_index=True
                        )
                        combined_result = combined_result.drop_duplicates(
                            subset=["row_idx"], keep="last"
                        )
                    else:
                        combined_result = df_result

                    new_df_merged = pd.merge(
                        df_in,
                        combined_result,
                        on="row_idx",
                        how="left",
                        suffixes=("_in", ""),
                    )
                    if (
                        "task_text_in" in new_df_merged.columns
                        and result_columns_saved[1] in new_df_merged.columns
                    ):
                        new_df_merged = new_df_merged.drop(
                            columns=["task_text_in"], errors="ignore"
                        )

                    st.session_state.df_result = combined_result
                    st.session_state.df_merged = new_df_merged
                    st.session_state.raw_results = raw_results + missed_raw
                    st.session_state.warnings = warnings + missed_warnings
                    # Accumulate the re-run batches too, so: (1) a second
                    # re-run keeps continuing the numbering instead of
                    # recomputing next_batch_idx from only the original
                    # batches, and (2) the debug ZIP / "Batches sent" metric
                    # reflect the re-run batches as well.
                    st.session_state.batches = batches + missed_batches
                    st.rerun()
                else:
                    st.warning("No batches created for missed tasks.")

        # Download buttons
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            tsv_bytes = df_merged.to_csv(sep="\t", index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download results (TSV) – Base",
                data=tsv_bytes,
                file_name="results.tsv",
                mime="text/tab-separated-values",
            )
        with col_dl2:
            zip_data = build_debug_zip(
                df_in=st.session_state.df_in,
                batches=st.session_state.batches,
                df_result=st.session_state.df_result,
                warnings=st.session_state.warnings,
                system_prompt=st.session_state.system_prompt_saved,
                columns_prompt=st.session_state.columns_prompt_saved,
                raw_results=st.session_state.raw_results,
            )
            st.download_button(
                "🐞 Download debug package (ZIP)",
                data=zip_data,
                file_name="debug_package.zip",
                mime="application/zip",
            )

    # Raw responses (debug) display
    with st.expander("Raw LLM responses (debug)", expanded=False):
        for idx, text in raw_results:
            st.markdown(f"**Batch {idx}**")
            st.code(text[:3000] + ("…" if len(text) > 3000 else ""), language="text")
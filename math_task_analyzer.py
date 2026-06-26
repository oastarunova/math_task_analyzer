"""
Math Task Analyzer – core logic for batch processing math tasks with LLM.
No Streamlit dependency – can be used standalone or imported.
"""

import io
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import tiktoken
from openai import OpenAI

# ─────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_MAX_TOKENS_PER_BATCH = 40000
DEFAULT_MAX_WORKERS = 5
TIKTOKEN_FALLBACK = "cl100k_base"

# Part 1 – role + behaviour (no column info here)
DEFAULT_SYSTEM_PROMPT = """\
Ты — аналитик текстовых данных, специализирующийся на классификации математических задач.
Твоя задача — обработать предоставленный набор строк и преобразовать каждую в строго структурированную запись.

ИНСТРУКЦИЯ ПО ОБРАБОТКЕ КАЖДОЙ ЗАДАЧИ:
1. Прочти оригинальный текст задачи. Исправь очевидные опечатки (орфографические, грамматические), НЕ меняя смысл, стиль и имена собственные. Исправленный текст станет значением поля "Текст задачи".
2. Последовательно заполни каждое из полей, строго следуя правилам в разделе КОЛОНКИ.
3. Для отсутствующей информации используй символ '-'. Не добавляй поля, не указанные в списке.

ТРЕБОВАНИЯ К ВЫВОДУ:
- Только CSV с разделителем табуляции (\\t).
- Первой строкой — заголовки столбцов точно как указано в разделе КОЛОНКИ.
- Каждая следующая строка — результат обработки одной задачи из входных данных.
- Без дополнительных комментариев, пояснений или форматирования вне таблицы.
- Все перечисления — через запятую.
- Сохраняй оригинальный текст задачи (только исправь опечатки).
- Сохраняй последовательность полей как указано в разделе КОЛОНКИ.\
"""

# Fixed first two columns — always present, hardcoded into the columns prompt
FIXED_COLUMNS = [
    ("row_idx", "числовой индекс строки из входных данных (перенеси без изменений)"),
    ("Текст задачи", "полный текст задачи с исправленными опечатками"),
]

# Default user-defined columns (name — definition, one per line)
DEFAULT_EXTRA_COLUMNS_TEXT = """\
Женский персонаж(и) — женщины и девочки в именительном падеже. Если пол группы невозможно определить (ученики, ребята…) — оставь пустым ('-'). Если определить можно (машинистки, покупательницы…) — укажи.
Мужской персонаж(и) — мужчины и мальчики в именительном падеже. Аналогичное правило для групп. «Некто» — мужской персонаж.
Мама — как упомянута в задаче в именительном падеже (мама, мать, мамочка…). '-' если нет.
Папа — как упомянут в задаче в именительном падеже (папа, отец, папочка…). '-' если нет.
Родители — все формы упоминания, если есть. '-' если нет.
Упоминаются ли дети — 'Да' или 'Нет'. Детские роли: дети, ребята, мальчик, девочка, ученики, пионеры, школьники, учащиеся, воспитанники, малыши и т.д. Краткие имена (Маша, Петя…) считаются детьми, если из контекста не следует обратного. Детские семейные роли (дочка, сын, сестра…) считаются детьми, если из контекста не следует обратного.
Детские персонажи — в том числе (единственном или множественном), в котором упомянуты. '-' если нет.
Девочки — только персонажи-девочки (не взрослые женщины, не группы без определённого пола). '-' если нет.
Мальчики — только персонажи-мальчики (не взрослые мужчины, не группы без определённого пола). '-' если нет.\
"""


def parse_extra_columns(text: str) -> list[tuple[str, str]]:
    """Parse user-supplied column definitions. Each line: 'name — definition'."""
    cols = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for sep in (" — ", " -- "):
            if sep in line:
                name, _, definition = line.partition(sep)
                cols.append((name.strip(), definition.strip()))
                break
    return cols


def build_columns_prompt(extra_cols: list[tuple[str, str]]) -> str:
    """Assemble the full columns-definition message sent to the LLM."""
    all_cols = FIXED_COLUMNS + extra_cols
    lines = ["КОЛОНКИ (в точном порядке):", ""]
    for i, (name, definition) in enumerate(all_cols, start=1):
        lines.append(f"{i}. {name} — {definition}")
    return "\n".join(lines)


def get_result_columns(extra_cols: list[tuple[str, str]]) -> list[str]:
    """Return the full ordered list of expected output column names."""
    return [name for name, _ in FIXED_COLUMNS] + [name for name, _ in extra_cols]


def get_encoder(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding(TIKTOKEN_FALLBACK)


def load_txt_files(uploaded_files) -> pd.DataFrame:
    """Parse uploaded .txt files → DataFrame[row_idx, source_file, task_text]."""
    rows = []
    for uf in uploaded_files:
        text = uf.read().decode("utf-8-sig", errors="replace")
        for line in text.splitlines():
            line = line.replace("\t", " ").strip()
            if line:
                rows.append({"source_file": uf.name, "task_text": line})
    if not rows:
        return pd.DataFrame(columns=["row_idx", "source_file", "task_text"])
    df = pd.DataFrame(rows).sort_values("source_file").reset_index(drop=True)
    df.insert(0, "row_idx", df.index)
    return df


def load_tsv_file(uploaded_file) -> pd.DataFrame:
    """Parse an uploaded TSV with 'file' and 'text' columns → DataFrame[row_idx, source_file, task_text].

    Raises ValueError if required columns are missing.
    """
    raw = uploaded_file.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8-sig", errors="replace")
    df_raw = pd.read_csv(io.StringIO(raw), sep="\t")

    missing = {"file", "text"} - set(df_raw.columns)
    if missing:
        raise ValueError(
            f"TSV is missing required column(s): {', '.join(sorted(missing))}. "
            f"Found columns: {', '.join(df_raw.columns)}"
        )

    df = pd.DataFrame(
        {
            "source_file": df_raw["file"].astype(str).str.strip(),
            "task_text": (
                df_raw["text"]
                .fillna("")
                .astype(str)
                .str.replace("\t", " ")
                .str.strip()
            ),
        }
    )
    df = df[df["task_text"] != ""].reset_index(drop=True)
    df = df.sort_values("source_file").reset_index(drop=True)
    df.insert(0, "row_idx", df.index)
    return df


def create_batches(df: pd.DataFrame, encoder, max_tokens: int) -> list[tuple[int, str]]:
    """Split df into token-limited TSV batches. Returns [(batch_idx, tsv_text), ...]."""
    if df.empty:
        return []

    send_cols = ["row_idx", "task_text"]
    header = "\t".join(send_cols)
    lines = [f"{row.row_idx}\t{row.task_text}" for row in df.itertuples()]

    batches, current, current_tok, batch_idx = [], [], 0, 0

    for line in lines:
        toks = len(encoder.encode(line))
        if toks > max_tokens:
            line = encoder.decode(encoder.encode(line)[:max_tokens])
            toks = max_tokens
        if current and current_tok + toks > max_tokens:
            batches.append((batch_idx, header + "\n" + "\n".join(current)))
            batch_idx += 1
            current, current_tok = [line], toks
        else:
            current.append(line)
            current_tok += toks

    if current:
        batches.append((batch_idx, header + "\n" + "\n".join(current)))

    return batches


def compute_token_stats(df_in: pd.DataFrame, encoder, max_tokens_batch: int):
    """Return (total_tokens, num_batches)."""
    if df_in.empty:
        return 0, 0
    lines = [f"{row.row_idx}\t{row.task_text}" for row in df_in.itertuples()]
    total_tokens = sum(len(encoder.encode(line)) for line in lines)
    batches = create_batches(df_in, encoder, max_tokens_batch)
    return total_tokens, len(batches)


def estimate_response_tokens(
    batch_text: str,
    encoder,
    num_extra_columns: int,
    overhead_per_row: int = 12,
    safety_factor: float = 1.25,
    min_tokens: int = 256,
    max_tokens_ceiling: int = 320000,
) -> int:
    """
    Estimate a safe max_response_tokens for a single batch, derived from
    that batch's own input size — not a flat global guess.

    Rationale: the model is asked to echo back the task text (≈ same
    token count as the input) plus fill in a small number of short
    extra-column values per row (each typically a short word/name or
    '-', ~overhead_per_row tokens). So:

        output_tokens ≈ input_tokens + num_rows * num_extra_columns * overhead_per_row

    A safety_factor multiplier covers natural variance (slightly longer
    fills, typo corrections that add a word, tokenizer mismatches
    between tiktoken's estimate and the actual model tokenizer, etc.)
    without resorting to one giant flat ceiling for every batch
    regardless of size.
    """
    lines = [l for l in batch_text.splitlines() if l.strip()]
    num_rows = max(len(lines) - 1, 0)  # minus header
    input_tokens = len(encoder.encode(batch_text))

    estimate = input_tokens + num_rows * num_extra_columns * overhead_per_row
    estimate = int(estimate * safety_factor)

    return max(min_tokens, min(estimate, max_tokens_ceiling))


def call_llm(
    client: OpenAI,
    batch_idx: int,
    batch_text: str,
    system_prompt: str,
    columns_prompt: str,
    model: str,
    temperature: float,
    max_response_tokens: int,
) -> tuple[int, str]:
    """Send one batch to the LLM. Returns (batch_idx, response_text).

    If the response was cut short because max_response_tokens was hit
    (finish_reason == "length"), a "TRUNCATED:" marker is prepended so
    parse_llm_response/combine_results can flag it distinctly from a
    generic malformed response.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": columns_prompt},
                {
                    "role": "assistant",
                    "content": "Понял. Жду входные данные для обработки.",
                },
                {
                    "role": "user",
                    "content": f"Обработай следующие строки (batch {batch_idx}):\n\n{batch_text}",
                },
            ],
            temperature=temperature,
            max_tokens=max_response_tokens,
            stream=False,
        )
        content = resp.choices[0].message.content
        finish_reason = resp.choices[0].finish_reason
        if finish_reason == "length":
            content = f"TRUNCATED:{content}"
        return batch_idx, content
    except Exception as e:
        return batch_idx, f"ERROR: {e}"


def parse_llm_response(
    batch_idx: int,
    text: str,
    expected_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Parse TSV response from LLM. Never skips an entire batch.
    Uses expected_columns as the DataFrame columns (ignores LLM's header if malformed).
    Only rows with exactly len(expected_columns) fields are kept.
    Returns (df, warnings).
    """
    warnings = []
    expected_col_count = len(expected_columns)

    if text.startswith("ERROR:"):
        warnings.append(f"Batch {batch_idx}: API error — {text}")
        return pd.DataFrame(), warnings

    truncated = text.startswith("TRUNCATED:")
    if truncated:
        text = text[len("TRUNCATED:"):]
        warnings.append(
            f"Batch {batch_idx}: response was TRUNCATED — it hit "
            f"max_response_tokens before finishing. The last row (and any "
            f"rows after it) may be cut off mid-line and will be skipped "
            f"or missing. Raise max_response_tokens or lower the input "
            f"batch size to fix this for good."
        )

    # Remove markdown fences
    text = re.sub(r"```[^\n]*\n?", "", text).strip()
    lines = [l for l in text.splitlines() if l.strip()]

    if len(lines) < 2:
        warnings.append(f"Batch {batch_idx}: response had fewer than 2 lines, skipped.")
        return pd.DataFrame(), warnings

    # Extract header (first non-empty line) – we may or may not use it
    header_line = lines[0]
    header_cols = header_line.split("\t")
    use_llm_header = (
        len(header_cols) == expected_col_count
        and len(set(header_cols)) == expected_col_count
    )

    if not use_llm_header:
        warnings.append(
            f"Batch {batch_idx}: header invalid (cols={len(header_cols)}, duplicates={len(header_cols) != len(set(header_cols))}). "
            f"Using expected columns: {expected_columns}"
        )

    # Determine the columns to use for the DataFrame
    final_columns = expected_columns if not use_llm_header else header_cols

    # Process data rows (skip header line)
    good_rows = []
    for i, line in enumerate(lines[1:], start=2):
        parts = line.split("\t")
        if len(parts) != expected_col_count:
            warnings.append(
                f"Batch {batch_idx}, line {i}: {len(parts)} columns (expected {expected_col_count}) — row skipped."
            )
        else:
            good_rows.append(parts)

    if not good_rows:
        warnings.append(f"Batch {batch_idx}: no valid data rows after column check.")
        return pd.DataFrame(), warnings

    df = pd.DataFrame(good_rows, columns=final_columns)
    # If we used LLM's header but column names might not match expected order, we could reorder,
    # but we assume the LLM respects the order as instructed. If needed, reorder:
    if use_llm_header and list(df.columns) != expected_columns:
        # Attempt to reorder: map LLM columns to expected order by position
        # Actually simpler: since we validated length, we can just assign expected_columns
        warnings.append(
            f"Batch {batch_idx}: LLM header columns differ from expected order. Overriding with expected columns."
        )
        df.columns = expected_columns
    return df, warnings


def combine_results(
    raw_results: list[tuple[int, str]],
    expected_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Parse + combine all batch responses. Returns (combined_df, all_warnings).
    If multiple rows have the same row_idx, the last one (by batch order) wins.
    """
    frames, all_warnings = [], []

    for batch_idx, text in raw_results:
        df, warns = parse_llm_response(batch_idx, text, expected_columns)
        all_warnings.extend(warns)

        if df.empty:
            continue

        if "row_idx" not in df.columns:
            # The whole batch is unusable without row_idx — every row
            # would be unmergeable/orphaned. Drop the batch entirely
            # rather than silently appending a frame that breaks the
            # later concat/drop_duplicates/merge steps.
            all_warnings.append(
                f"Batch {batch_idx}: response was missing the 'row_idx' "
                f"column entirely — whole batch dropped, rows will show "
                f"as missing and can be re-run."
            )
            continue

        # Ensure row_idx is numeric for proper merging later.
        # The LLM sometimes returns row_idx with stray characters
        # (e.g. "2)", " 2 ", "Row 2") — strip everything except
        # digits/minus sign before coercing, so a row's data never
        # gets silently orphaned from its row_idx.
        cleaned = df["row_idx"].astype(str).str.extract(r"(-?\d+)", expand=False)
        numeric = pd.to_numeric(cleaned, errors="coerce")
        bad_mask = numeric.isna()
        if bad_mask.any():
            bad_originals = df.loc[bad_mask, "row_idx"].tolist()
            all_warnings.append(
                f"Batch {batch_idx}: {bad_mask.sum()} row(s) had an "
                f"unparseable row_idx ({bad_originals}) — dropped, "
                f"will show as missing and can be re-run."
            )
        df = df.loc[~bad_mask].copy()
        df["row_idx"] = numeric.loc[~bad_mask]

        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(), all_warnings

    combined = pd.concat(frames, ignore_index=True)
    # Every frame here is guaranteed to have row_idx, so this is now safe.
    combined = combined.drop_duplicates(subset=["row_idx"], keep="last")
    return combined, all_warnings


def run_batches(
    client: OpenAI,
    batches: list[tuple[int, str]],
    system_prompt: str,
    columns_prompt: str,
    model: str,
    temperature: float,
    max_response_tokens,
    max_workers: int,
    progress_callback=None,
    encoder=None,
    num_extra_columns: int = 0,
) -> list[tuple[int, str]]:
    """
    Run all batches in parallel.
    If progress_callback is provided, it will be called after each batch with (done, total).

    max_response_tokens can be:
      - an int: used as a fixed cap for every batch (old behavior).
      - "auto": each batch gets its own cap computed by
        estimate_response_tokens() from that batch's own input size and
        num_extra_columns, instead of one flat guess for every batch
        regardless of size. Requires `encoder` to be passed.

    Returns sorted list of (batch_idx, response_text).
    """
    results = []
    total = len(batches)
    done = 0

    def resolve_cap(batch_text: str) -> int:
        if max_response_tokens == "auto":
            return estimate_response_tokens(
                batch_text, encoder, num_extra_columns
            )
        return max_response_tokens

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {
            exe.submit(
                call_llm,
                client,
                idx,
                text,
                system_prompt,
                columns_prompt,
                model,
                temperature,
                resolve_cap(text),
            ): idx
            for idx, text in batches
        }
        for future in as_completed(futures):
            batch_idx, result = future.result()
            results.append((batch_idx, result))
            done += 1
            if progress_callback:
                progress_callback(done, total)

    return sorted(results, key=lambda x: x[0])
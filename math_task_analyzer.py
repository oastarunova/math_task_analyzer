"""
Math Task Analyzer – core logic for batch processing math tasks with LLM.
No Streamlit dependency – can be used standalone or imported.
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import tiktoken
from openai import OpenAI

# ─────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_MAX_TOKENS_PER_BATCH = 4000
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
        for sep in (" — ", " - "):
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
    """Send one batch to the LLM. Returns (batch_idx, response_text)."""
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
        return batch_idx, resp.choices[0].message.content
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
        if not df.empty:
            # Ensure row_idx is numeric for proper merging later
            if "row_idx" in df.columns:
                df["row_idx"] = pd.to_numeric(df["row_idx"], errors="coerce")
            frames.append(df)

    if not frames:
        return pd.DataFrame(), all_warnings

    combined = pd.concat(frames, ignore_index=True)
    # Drop duplicate row_idx, keep last occurrence (most recent batch)
    if "row_idx" in combined.columns:
        combined = combined.drop_duplicates(subset=["row_idx"], keep="last")
    return combined, all_warnings


def run_batches(
    client: OpenAI,
    batches: list[tuple[int, str]],
    system_prompt: str,
    columns_prompt: str,
    model: str,
    temperature: float,
    max_response_tokens: int,
    max_workers: int,
    progress_callback=None,
) -> list[tuple[int, str]]:
    """
    Run all batches in parallel.
    If progress_callback is provided, it will be called after each batch with (done, total).
    Returns sorted list of (batch_idx, response_text).
    """
    results = []
    total = len(batches)
    done = 0

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
                max_response_tokens,
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

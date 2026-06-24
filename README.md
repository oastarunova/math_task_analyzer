# 🧮 Math Task Batch Analyzer

Upload `.txt` files with math problems → send to LLM in batches → get a structured table (TSV/DataFrame) with custom columns.

Perfect for teachers, researchers, or anyone who needs to classify many math tasks using a large language model.

---

## ✨ Features

- **Batch processing** – splits thousands of tasks into token‑limited batches.
- **Parallel LLM calls** – processes multiple batches simultaneously (configurable concurrency).
- **Custom output columns** – define your own classification fields (e.g., “Женский персонаж(и)”, “Мама”, “Упоминаются ли дети”).
- **Error resilient** – even if one row in a batch is malformed, the rest are kept.
- **Re‑run missed tasks** – automatically find and reprocess any rows the LLM missed.
- **Debug package** – download a ZIP with:
  - Input tasks + batch indices  
  - Final result TSV  
  - All warnings  
  - System & column prompts  
  - Raw LLM responses (one file per batch)

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/math-task-analyzer.git
cd math-task-classifier
2. Install dependencies
```bash
pip install -r requirements.txt
3. Run the app
```bash
streamlit run streamlit_app.py

The app will open in your browser.

🧪 How to Use
Enter your API key (supports any OpenAI‑compatible endpoint – DeepSeek, OpenAI, etc.).

Define extra columns – one per line, using name —(or --) definition.
Example:
Упоминаются ли дети -- 'Да' или 'Нет'

Upload .txt files – one task per line.

Click “Send to LLM” – wait for processing (progress bar shown).

Download results:

Base – TSV file with all input tasks + LLM output.

Debug – ZIP package for inspection.

Re‑run missed tasks – if some rows were not returned, click the button to process only those.

⚙️ Configuration (Sidebar)
Setting	Description
API Key	Your LLM provider key (e.g., DeepSeek, OpenAI).
API Base URL	Endpoint URL (default: https://api.deepseek.com).
Model	Model name (e.g., deepseek-chat, gpt-4o).
System prompt	Role + behaviour instructions (editable).
Extra columns	Your custom classification fields.
Max input tokens/batch	Token limit for each batch (default: 4000).
Max response tokens	Max tokens LLM may generate per batch.
Temperature	Randomness (0.0 = deterministic, 2.0 = creative).
Parallel workers	Number of batches to process simultaneously.
📁 Project Structure
text
.
├── streamlit_app.py          # Streamlit UI
├── math_task_analyzer.py     # Core logic (batching, LLM calls, parsing)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
🧠 How It Works
Load tasks – each line from .txt files becomes a task with a unique row_idx.

Build batches – tasks are grouped into TSV strings respecting the token limit.

Send to LLM – each batch is processed in parallel. The prompt consists of:

System prompt (role + rules)

Columns definition (fixed + extra)

Assistant pre‑fill (Понял. Жду данные.)

Batch data (TSV with row_idx and task_text)

Parse response – each row is validated against the expected number of columns.
Malformed rows are skipped with a warning; the batch is never discarded completely.

Merge & display – results are merged with the original input, missing rows are shown as NaN.

Re‑run – only missing row_idx values are re‑sent to the LLM.

🔧 Customising Columns
The first two columns are fixed:

row_idx – index of the task in the input.

Текст задачи – corrected task text.

All other columns are defined under Extra columns.
Each line must contain — (em‑dash with spaces) or - (regular dash).
Example:

text
Женский персонаж(и) — женщины и девочки в именительном падеже...
Мужской персонаж(и) — мужчины и мальчики...
Мама — как упомянута в задаче...
The LLM will output exactly those columns in the order you define them.

🐞 Troubleshooting
TypeError: object of type 'int' has no len()
Make sure you are using the latest math_task_analyzer.py that expects a list of column names, not an integer count. The provided code already works correctly.

Missing rows after processing
Check the warnings (expand the yellow section). Common causes:

LLM didn’t return a row for that row_idx.

Row had wrong number of fields (e.g., extra/missing tabs).

API key errors
Verify your key and base URL. For DeepSeek, the default URL is https://api.deepseek.com.

📜 License
MIT – use freely, modify as you like.

🙏 Acknowledgements
Built with Streamlit, OpenAI Python library, tiktoken, and pandas.

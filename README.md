# Log Classification

A hybrid log classification system that combines **Regex**, **Sentence Transformer + Logistic Regression**, and an **LLM** to automatically label system logs, served through a **FastAPI** endpoint that takes a CSV file and returns it with predicted labels.

![Architecture](resources/arch.png)

## How It Works

| Layer | File | Handles | Notes |
|---|---|---|---|
| Regex | `processor_regex.py` | High-frequency logs with a fixed format (logins, backups, system updates, ...) | Rules derived from patterns found by DBSCAN clustering during training. Fastest and free |
| BERT + LR | `processor_bert.py` | Logs with enough training samples but no fixed format | `all-MiniLM-L6-v2` produces sentence embeddings, Logistic Regression classifies them. Returns `Unclassified` when confidence < 0.5 |
| LLM | `processor_llm.py` | `LegacyCRM` logs (too few samples to train on) and logs where BERT confidence < 0.5 | Calls `openai/gpt-oss-120b` on Groq. Outputs `Workflow Error`, `Deprecation Warning`, or `Unclassified` |

Routing logic in `classify.py`:

```
source == "LegacyCRM" ?
   ├── yes → LLM
   └── no  → Regex match?
               ├── yes → Regex label
               └── no  → BERT + Logistic Regression
                          └── confidence < 0.5 → LLM
```

Supported labels: `HTTP Status`, `Security Alert`, `System Notification`, `Error`, `Resource Usage`, `Critical Error`, `User Action`, `Workflow Error`, `Deprecation Warning`, `Unclassified`.

## Project Structure

```
classification-logs/
├── server.py              # FastAPI endpoint: POST /classify/
├── classify.py            # Routing across the three classifiers
├── processor_regex.py     # Regex rules
├── processor_bert.py      # Sentence Transformer + Logistic Regression
├── processor_llm.py       # Groq LLM classifier
├── requirements.txt
├── .env                   # GROQ_API_KEY (not committed)
├── resources/
│   ├── arch.png           # Architecture diagram
│   ├── test.csv           # Sample input
│   └── output.csv         # Classification output (generated)
└── training/
    ├── dataset/
    │   ├── synthetic_logs.csv   # Training data (2,410 logs)
    │   └── training.py          # Training script
    └── models/
        └── log_classifier.joblib  # Trained Logistic Regression model
```

## Installation

Requires Python 3.10+.

```bash
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root with your Groq API key (get one at https://console.groq.com):

```
GROQ_API_KEY=your_key_here
```

> If several Python versions are installed, always use `python -m pip` and `python -m uvicorn` so packages are installed into and run by the same interpreter.

## Training

A trained model is already included at `training/models/log_classifier.joblib`. To retrain:

```bash
cd training/dataset
python training.py
```

Training steps:

1. Embed all logs with a Sentence Transformer, cluster them with DBSCAN, and sort clusters by size
2. Print the largest clusters to inspect recurring patterns
3. Write Regex rules for those patterns and filter out the logs they match
4. Drop `LegacyCRM` logs, which have too few samples (handled by the LLM at inference time)
5. Embed the remaining logs with the Sentence Transformer
6. Train a Logistic Regression classifier and save it

## Running the Server

From the project root:

```bash
python -m uvicorn server:app --reload
```

The server is ready when you see `Uvicorn running on http://127.0.0.1:8000`.

> Opening http://127.0.0.1:8000/ returns `{"detail":"Not Found"}`. This is expected: no route is defined at the root path.

## API

**`POST /classify/`**

- Request: `multipart/form-data` with a `.csv` file in the `file` field
- The CSV must contain `source` and `log_message` columns
- Response: the same CSV with an added `target_label` column, also saved to `resources/output.csv`

Example input:

```csv
source,log_message
ModernCRM,IP 192.168.133.114 blocked due to potential attack
AnalyticsEngine,Backup completed successfully.
LegacyCRM,Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active.
```

Example output:

```csv
source,log_message,target_label
ModernCRM,IP 192.168.133.114 blocked due to potential attack,Security Alert
AnalyticsEngine,Backup completed successfully.,System Notification
LegacyCRM,Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active.,Workflow Error
```

### Option 1: Swagger UI (/docs)

1. Open http://127.0.0.1:8000/docs
2. Expand **POST /classify/** → **Try it out**
3. Choose `resources/test.csv` for `file` → **Execute**
4. A response code of 200 means success

### Option 2: Postman

1. Method **POST**, URL `http://127.0.0.1:8000/classify/`
2. **Body → form-data**, key `file`, switch the type from Text to **File**, select a CSV
3. Click **Send** (do not set a `Content-Type` header manually)

### Option 3: curl

```bash
curl -X POST -F "file=@resources/test.csv" http://127.0.0.1:8000/classify/
```

### Common Errors

| Status | Cause |
|---|---|
| 400 | File is not a `.csv`, or the `source` / `log_message` columns are missing |
| 405 | Sent a GET request instead of POST |
| 422 | No file uploaded, or the form field is not named `file` |
| 500 | Error during classification; check the uvicorn terminal output |

## Testing Individual Modules

```bash
python processor_regex.py
python processor_bert.py
python processor_llm.py
python classify.py        # Classifies resources/test.csv and writes resources/output.csv
```

## Notes

- **Groq usage**: every `LegacyCRM` log and every low-confidence BERT prediction triggers one Groq API call, billed to the key in `.env`.
- **No authentication**: the endpoint has no access control. Add an API key check before exposing it publicly.
- **Shared output file**: all requests write to the same `resources/output.csv`, so concurrent requests overwrite each other.
- **Network access**: the server listens on localhost by default. For LAN access, add `--host 0.0.0.0` and allow it through the firewall. For public access, deploy to a cloud server or use a tunneling tool.

# Testing Guide - Streaming Transformations

## 🌊 Available Test Scripts

All test scripts now use **real-time streaming** to show progress!

### 1. **test_api.py** - Quick Test (Recommended First)
Small dataset, fast results (~5-10 seconds)

```powershell
python test_api.py
```

**What it does:**
- Tests health check
- Tests root endpoint  
- Streams transformation of minimal data
- Shows real-time progress
- Saves output to `test_output.json`

**Expected output:**
```
============================================================
TEST 3: Transform Endpoint with Streaming
============================================================
🌊 Connecting to streaming endpoint...
🔄 Transforming: HarvestBowls → TrendWave

✅ Streaming progress:
------------------------------------------------------------
⚙️  IngestorNode
✅ IngestorNode
⚙️  AnalyzerNode
✅ AnalyzerNode
⚙️  TransformerNode
🤖 OpenAI generating (streaming)...
{"topicWizardData":{"lessonInformation... [streaming continues]
✅ TransformerNode
⚙️  ConsistencyCheckerNode
✅ ConsistencyCheckerNode
⚙️  ValidatorNode
✅ ValidatorNode
🎉 Complete!
------------------------------------------------------------

⏱️  Total time: 8.23s
```

---

### 2. **test_with_sample.py** - Large File Test
Uses your actual `sample_input.json` (~65K characters, 30-60 seconds)

```powershell
python test_with_sample.py
```

**What it does:**
- Loads `sample_input.json`
- Streams transformation with timestamps
- Shows character count progress
- Displays node-by-node completion
- Saves to `transformed_output.json` and `transformed_data_only.json`

**Expected output:**
```
============================================================
SAMPLE FILE TRANSFORMATION TEST (STREAMING)
============================================================
File size: 65211 characters (~16302 tokens)

🌊 Starting streaming transformation...
🔄 This is a large file - you'll see real-time progress
⏱️  Expected time: 30-60 seconds

✅ Connected! Streaming progress:
------------------------------------------------------------
[  0.1s] 🚀 Starting transformation
[  0.2s] ⚙️  [1/6] IngestorNode
[  1.5s] ✅ IngestorNode completed
[  1.6s] ⚙️  [2/6] AnalyzerNode
[  2.0s] ✅ AnalyzerNode completed
[  2.1s] ⚙️  [3/6] TransformerNode
[  3.5s] 🤖 OpenAI is generating response...
[  8.2s] 🌊 OpenAI streaming started...
[ 12.5s] 📝 Generated 5,000 characters...
[ 18.3s] 📝 Generated 10,000 characters...
[ 35.1s] 📝 Generated 65,000 characters...
[ 36.2s] ✅ TransformerNode completed
[ 36.5s] ⚙️  [4/6] ConsistencyCheckerNode
[ 38.0s] ✅ ConsistencyCheckerNode completed
[ 38.1s] ⚙️  [5/6] ValidatorNode
[ 40.5s] ✅ ValidatorNode completed
[ 40.8s] 🎉 Transformation Complete!
------------------------------------------------------------

⏱️  Total time: 40.82 seconds
```

---

### 3. **test_openai_stream.py** - See Raw OpenAI Output
Shows the actual JSON being generated character-by-character!

```powershell
python test_openai_stream.py
```

**What it does:**
- Displays the raw OpenAI streaming response
- Shows JSON being built in real-time
- Saves streamed text to `openai_streamed_text.txt`
- Most detailed view of the generation process

---

### 4. **test_stream.py** - Alternative Streaming Test
Similar to test_api.py but with different formatting

```powershell
python test_stream.py
```

---

## 🎯 Recommended Testing Sequence

1. **First time:** `python test_api.py` (fast, validates everything works)
2. **Full test:** `python test_with_sample.py` (your real data)
3. **Debugging:** `python test_openai_stream.py` (see what OpenAI generates)

---

## 📊 What Gets Saved

| File | From | Contains |
|------|------|----------|
| `test_output.json` | test_api.py | Minimal test output |
| `transformed_output.json` | test_with_sample.py | Full response with validation |
| `transformed_data_only.json` | test_with_sample.py | Just the transformed JSON |
| `openai_stream_output.json` | test_openai_stream.py | Complete streaming result |
| `openai_streamed_text.txt` | test_openai_stream.py | Raw streamed text |

---

## 🚀 Quick Start

```powershell
# Make sure server is running
uvicorn src.main:app --reload

# In another terminal, run:
python test_api.py
```

---

## 🔧 Troubleshooting

**Timeout errors?**
- Large files need more time
- Check your OpenAI API key in `.env`
- Verify you have API credits

**No streaming output?**
- Server must be restarted to load new streaming endpoints
- Run: `uvicorn src.main:app --reload`

**Import errors?**
- Activate virtual environment: `.venv\Scripts\Activate.ps1`
- Install dependencies: `pip install -r requirements.txt`

---

## 📡 Streaming Endpoints

Your API now has these endpoints:

1. `POST /api/v1/transform` - Standard (no streaming)
2. `POST /api/v1/transform/stream` - Streaming with node progress
3. `POST /api/v1/transform/stream-openai` - **NEW!** Full OpenAI streaming

Use endpoint #3 for the best real-time experience!

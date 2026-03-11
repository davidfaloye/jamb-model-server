import sys
import os
from flask import Flask, request, jsonify
from loader import load_all, generate_response, get_model, list_adapters, _models
import time

app = Flask(__name__)

# ── Health check ───────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    loaded = list(_models.keys())
    return jsonify({
        'status':         'ok',
        'models_loaded':  len(loaded),
        'adapters_ready': [s.capitalize() for s in loaded],
    })

# ── Warmup ────────────────────────────────────────────────────────────────

@app.route('/warmup', methods=['GET'])
def warmup():
    return jsonify({ 'status': 'ready', 'message': 'All models loaded at startup.' })

# ── Explain endpoint ───────────────────────────────────────────────────────

@app.route('/explain', methods=['POST'])
def explain():
    data     = request.json or {}
    subject  = data.get('subject', 'General')
    question = data.get('question', '')
    answer   = data.get('answer', '')
    correct  = data.get('correct_answer', '')
    is_right = data.get('is_correct', False)

    start    = time.time()
    response = generate_response(subject, prompt)
    elapsed  = time.time() - start
    tokens   = len(response.split())
    print(f'⏱ {elapsed:.1f}s | ~{tokens} words | ~{tokens/elapsed:.1f} words/sec', flush=True)

    print(f'📥 /explain called', flush=True)
    print(f'⚡ Subject: {subject} (model already in memory)', flush=True)

    prompt = (
        f"A JAMB student answered a {subject} question.\n"
        f"Question: {question}\n"
        f"Their answer: {answer}\n"
        f"Correct answer: {correct}\n"
        f"They were {'correct' if is_right else 'incorrect'}.\n"
        f"Give a clear, concise explanation of why {correct} is correct. "
        f"Keep it under 100 words."
    )

    try:
        response = generate_response(subject, prompt)
        return jsonify({ 'explanation': response })
    except Exception as e:
        print(f'ERROR /explain: {e}', flush=True)
        return jsonify({ 'error': str(e) }), 500

# ── Chat endpoint ──────────────────────────────────────────────────────────

@app.route('/chat', methods=['POST'])
def chat():
    data    = request.json or {}
    subject = data.get('subject', 'General')
    message = data.get('message', '')
    topic   = data.get('topic', '')

    print(f'📥 /chat | Subject: {subject} | Topic: {topic}', flush=True)
    print(f'⚡ Model already in memory — no load time', flush=True)

    prompt = (
        f"You are a JAMB tutor helping a Nigerian student with {subject}.\n"
        f"Topic: {topic}\n"
        f"Student: {message}\n"
        f"Give a clear, helpful answer suitable for JAMB exam preparation."
    )

    try:
        response = generate_response(subject, prompt)
        return jsonify({ 'response': response })
    except Exception as e:
        print(f'ERROR /chat: {e}', flush=True)
        return jsonify({ 'error': str(e) }), 500

# ── Startup ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.getenv('MODEL_SERVER_PORT', 8000))

    print('=' * 50, flush=True)
    print('🚀 JAMB AI Model Server starting...', flush=True)
    print(f'📂 Loading all adapters from disk...', flush=True)
    print('=' * 50, flush=True)

    # Load every adapter into memory before accepting requests
    load_all()

    print('=' * 50, flush=True)
    print(f'✅ Server ready on port {port}', flush=True)
    print('=' * 50, flush=True)
    sys.stdout.flush()

    # threaded=False — models are not thread-safe
    app.run(host='0.0.0.0', port=port, threaded=False)


## What changes at startup

# Before (lazy — loaded on first request):
# 🚀 Server ready
# 📥 /explain called
# ⏳ Loading adapter for Physics...   ← user waits 30-90s
# ✓ Response sent

# After (eager — all loaded before first request):
# 🚀 Server starting...
# ✓ Geography adapter ready
# ✓ Physics adapter ready
# ✓ Chemistry adapter ready
# ✅ Server ready on port 8000        ← user connects here
# 📥 /explain called
# ⚡ Model already in memory
# ✓ Response sent                     ← instant lookup
"""
loader.py
JAMB AI Model Server — adapter loader.

Startup behaviour:
  - Loads tokenizer once (shared across all models)
  - Loads one PeftModel per discovered adapter
  - No separate fallback base model (saves ~2.5 GB RAM on CPU)
  - Unknown subjects fall back to the first loaded adapter

Inference behaviour:
  - get_model(subject) is an instant dict lookup — no per-request loading
  - generate_response(subject, prompt) runs inference on the correct model
"""

import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ── Config ─────────────────────────────────────────────────────────────────

BASE_MODEL_PATH = os.getenv('BASE_MODEL',   '../models/base/tinyllama')
ADAPTERS_DIR    = os.getenv('ADAPTERS_DIR', '../models/adapters')

# ── Global state ───────────────────────────────────────────────────────────

_tokenizer  = None   # shared tokenizer, loaded once
_models     = {}     # { 'physics': PeftModel, 'chemistry': PeftModel, ... }
_base_model = None   # only populated if zero adapters are found
DEVICE      = None   # set during load_all()

# ── Device ─────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

# ── Adapter discovery ──────────────────────────────────────────────────────

def list_adapters() -> list[str]:
    """
    Scans ADAPTERS_DIR and returns sorted list of subject folder names
    that contain a valid final/adapter_config.json.
    """
    if not os.path.exists(ADAPTERS_DIR):
        print(f'[Loader] Adapters directory not found: {ADAPTERS_DIR}', flush=True)
        return []

    found = []
    for entry in os.listdir(ADAPTERS_DIR):
        final_path  = os.path.join(ADAPTERS_DIR, entry, 'final')
        config_path = os.path.join(final_path, 'adapter_config.json')
        if os.path.isdir(final_path) and os.path.exists(config_path):
            found.append(entry.lower())

    return sorted(found)

# ── Base model loader ──────────────────────────────────────────────────────

def _load_base(device: str) -> AutoModelForCausalLM:
    """
    Loads a fresh LlamaForCausalLM from disk.
    Called once per adapter — each PeftModel needs its own base instance
    because PeftModel.from_pretrained() mutates the base it receives.
    """
    if device == 'cuda':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            quantization_config=bnb_config,
            device_map='auto',
            local_files_only=True,
        )
    else:
        # CPU / MPS — plain float32, no quantization
        return AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float32,
            device_map='cpu',
            local_files_only=True,
        )

# ── Startup ────────────────────────────────────────────────────────────────

def load_all():
    """
    Called ONCE at server startup before Flask begins accepting requests.

    Steps:
      1. Detect device
      2. Load shared tokenizer
      3. Discover all adapters in ADAPTERS_DIR
      4. Load each adapter into _models{}
         - Each gets its own fresh base model instance
         - Skips failed adapters without crashing
      5. If no adapters found, load one raw base model as last resort

    After load_all() returns, all inference is instant dict lookups.
    No fallback base model is loaded when adapters exist — saves ~2.5 GB RAM.
    """
    global _tokenizer, _base_model, _models, DEVICE

    DEVICE = get_device()
    print(f'[Loader] Device       : {DEVICE}', flush=True)
    print(f'[Loader] Base model   : {BASE_MODEL_PATH}', flush=True)
    print(f'[Loader] Adapters dir : {ADAPTERS_DIR}', flush=True)

    # ── 1. Tokenizer ──────────────────────────────────────────────────────
    print('[Loader] Loading tokenizer...', flush=True)
    _tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        local_files_only=True,
    )
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    print('[Loader] Tokenizer ready.', flush=True)

    # ── 2. Discover adapters ──────────────────────────────────────────────
    subjects = list_adapters()

    if not subjects:
        # Edge case: no adapters trained yet — load one raw base model
        print('[Loader] No adapters found — loading raw base model.', flush=True)
        _base_model = _load_base(DEVICE)
        _base_model.eval()
        print('[Loader] ✅ Base model ready (no adapters).', flush=True)
        return

    print(f'[Loader] Adapters found: {subjects}', flush=True)

    # ── 3. Load each adapter ──────────────────────────────────────────────
    for subject in subjects:
        adapter_path = os.path.join(ADAPTERS_DIR, subject, 'final')
        print(f'[Loader] Loading: {subject}...', flush=True)

        try:
            # Fresh base per adapter — avoids peft_config collision warning
            base = _load_base(DEVICE)

            model = PeftModel.from_pretrained(
                base,
                adapter_path,
                local_files_only=True,
            )
            model.eval()

            _models[subject] = model
            print(f'[Loader] ✓ {subject.capitalize()} ready.', flush=True)

        except Exception as e:
            # Log and skip — don't crash the whole server for one bad adapter
            print(f'[Loader] ✗ Failed to load {subject}: {e}', flush=True)

        # Clean up between loads to keep memory tidy
        gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    # ── 4. Done ───────────────────────────────────────────────────────────
    # _base_model intentionally left as None when adapters are loaded.
    # This avoids loading an extra ~2.5 GB model that isn't needed.
    # get_model() falls back to the first loaded adapter for unknown subjects.

    loaded = list(_models.keys())
    print(f'[Loader] ✅ All models ready: {loaded}', flush=True)

# ── Model lookup ───────────────────────────────────────────────────────────

def get_model(subject: str):
    """
    Returns (model, tokenizer) for the given subject.
    Zero loading — instant dict lookup.

    Fallback chain:
      1. Exact key match     'physics'       -> _models['physics']
      2. Normalised match    'Use of English' -> _models['use_of_english']
      3. Fuzzy substring     'eng'           -> _models['use_of_english']
      4. First loaded model  anything else   -> _models[first key]
      5. Raw base model      if _models is empty (no adapters loaded)
    """
    # Normalise: lowercase + spaces to underscores
    key = subject.lower().strip().replace(' ', '_')

    # 1. Exact match
    if key in _models:
        return _models[key], _tokenizer

    # 2. Normalised match (handle e.g. 'use of english' vs 'use_of_english')
    normalised = {k.replace('_', ' '): k for k in _models}
    readable   = key.replace('_', ' ')
    if readable in normalised:
        matched = normalised[readable]
        print(f'[Loader] Normalised match: "{subject}" -> "{matched}"', flush=True)
        return _models[matched], _tokenizer

    # 3. Fuzzy substring match
    for stored_key in _models:
        stored_readable = stored_key.replace('_', ' ')
        if stored_readable in readable or readable in stored_readable:
            print(f'[Loader] Fuzzy match: "{subject}" -> "{stored_key}"', flush=True)
            return _models[stored_key], _tokenizer

    # 4. First available adapter (better than nothing — same base weights)
    if _models:
        fallback_key = next(iter(_models))
        print(f'[Loader] No match for "{subject}" — using "{fallback_key}" as fallback.', flush=True)
        return _models[fallback_key], _tokenizer

    # 5. Raw base model (only reachable if zero adapters were loaded)
    print(f'[Loader] Using raw base model for "{subject}".', flush=True)
    return _base_model, _tokenizer

# ── Inference ──────────────────────────────────────────────────────────────

def generate_response(subject: str, prompt: str) -> str:
    """
    Looks up the correct model for the subject and runs inference.
    Model is already in memory — no loading delay.

    GPU  : sampling, 256 new tokens, ~1–3 seconds
    CPU  : greedy decode, 50 new tokens, ~15–30 seconds
    """
    model, tokenizer = get_model(subject)

    if model is None:
        raise RuntimeError(
            'No model available. Ensure load_all() ran successfully at startup.'
        )

    # ChatML prompt format expected by TinyLlama
    formatted = (
        f"<|im_start|>user\n"
        f"{prompt}"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(
        formatted,
        return_tensors='pt',
        truncation=True,
        max_length=512,
    )

    if DEVICE == 'cuda':
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        if DEVICE == 'cuda':
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            # CPU: greedy decode (do_sample=False) + low token count for speed.
            # max_new_tokens only — do NOT set max_length alongside it.
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    # Decode only the newly generated tokens (skip echoing the input prompt)
    input_length = inputs['input_ids'].shape[1]
    generated    = output[0][input_length:]
    response     = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return response
"""
loader.py
Single-model strategy for GPU with limited VRAM.

Instead of loading all adapters simultaneously (which fills VRAM),
we keep only ONE model in memory at a time.

- First request for Physics  → loads Physics  (~3.5 GB VRAM)
- Second request for Physics → instant cache hit, no reload
- Request for Chemistry      → unloads Physics, loads Chemistry
- Request for Chemistry      → instant cache hit

This keeps VRAM usage at ~3.5 GB instead of ~14 GB.
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

_tokenizer    = None   # shared, loaded once
_active_model = None   # only ONE model in memory at a time
_active_subj  = None   # which subject is currently loaded
DEVICE        = None   # set during init

# ── Device ─────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

# ── Adapter discovery ──────────────────────────────────────────────────────

def list_adapters() -> list[str]:
    if not os.path.exists(ADAPTERS_DIR):
        return []
    found = []
    for entry in os.listdir(ADAPTERS_DIR):
        final_path  = os.path.join(ADAPTERS_DIR, entry, 'final')
        config_path = os.path.join(final_path, 'adapter_config.json')
        if os.path.isdir(final_path) and os.path.exists(config_path):
            found.append(entry.lower())
    return sorted(found)

# ── VRAM monitor ───────────────────────────────────────────────────────────

def print_vram():
    if DEVICE == 'cuda':
        allocated = torch.cuda.memory_allocated(0)  / 1e9
        reserved  = torch.cuda.memory_reserved(0)   / 1e9
        total     = torch.cuda.get_device_properties(0).total_memory / 1e9
        free      = total - reserved
        print(f'[VRAM] {allocated:.1f}GB allocated | {reserved:.1f}GB reserved | {free:.1f}GB free / {total:.1f}GB total', flush=True)

# ── Clear active model from VRAM ───────────────────────────────────────────

def _clear_model():
    global _active_model, _active_subj

    if _active_model is None:
        return

    print(f'[Loader] Unloading {_active_subj} from VRAM...', flush=True)
    del _active_model
    _active_model = None
    _active_subj  = None

    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print_vram()

# ── Load base model ────────────────────────────────────────────────────────

def _load_base() -> AutoModelForCausalLM:
    print('[Loader] Loading base model...', flush=True)
    if DEVICE == 'cuda':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            quantization_config=bnb_config,
            device_map='cuda',
            local_files_only=True,
        )
    else:
        return AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float32,
            device_map='cpu',
            local_files_only=True,
        )

# ── Startup ────────────────────────────────────────────────────────────────

def load_all():
    """
    Startup: load tokenizer only.
    Models are loaded on first request per subject (lazy per-subject loading)
    but cached so subsequent requests for the same subject are instant.
    """
    global _tokenizer, DEVICE

    DEVICE = get_device()
    print(f'[Loader] Device: {DEVICE}', flush=True)
    print_vram()

    print('[Loader] Loading tokenizer...', flush=True)
    _tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        local_files_only=True,
    )
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    print('[Loader] Tokenizer ready.', flush=True)

    # Show what adapters are available without loading them
    available = list_adapters()
    print(f'[Loader] Adapters available: {available}', flush=True)
    print(f'[Loader] ✅ Server ready — models load on first request per subject.', flush=True)
    print_vram()

# ── Subject normalisation ───────────────────────────────────────────────────

def _normalise(subject: str) -> str:
    """
    Converts any subject string to the folder name format.
    e.g. 'Use of English' -> 'use_of_english'
         'Physics'        -> 'physics'
    """
    return subject.lower().strip().replace(' ', '_')

def _find_adapter_key(subject: str) -> str | None:
    """
    Finds the best matching adapter folder name for a given subject string.
    Returns None if no match found.
    """
    key       = _normalise(subject)
    available = list_adapters()

    # 1. Exact match
    if key in available:
        return key

    # 2. Normalised match (spaces vs underscores)
    readable = key.replace('_', ' ')
    for a in available:
        if a.replace('_', ' ') == readable:
            return a

    # 3. Fuzzy substring
    for a in available:
        if a in key or key in a:
            print(f'[Loader] Fuzzy match: "{subject}" -> "{a}"', flush=True)
            return a

    # 4. No match
    return None

# ── Load subject model ─────────────────────────────────────────────────────

def _load_subject(subject: str):
    """
    Loads the adapter for the given subject into VRAM.
    Clears the currently active model first to free VRAM.
    """
    global _active_model, _active_subj

    adapter_key  = _find_adapter_key(subject)
    adapter_path = os.path.join(ADAPTERS_DIR, adapter_key, 'final') if adapter_key else None

    # Clear current model from VRAM before loading new one
    _clear_model()
    print_vram()

    base = _load_base()

    if adapter_path and os.path.exists(adapter_path):
        print(f'[Loader] Wrapping with adapter: {adapter_key}', flush=True)
        model = PeftModel.from_pretrained(
            base,
            adapter_path,
            local_files_only=True,
        )
    else:
        print(f'[Loader] No adapter for "{subject}" — using base model.', flush=True)
        model = base

    model.eval()

    _active_model = model
    _active_subj  = adapter_key or subject

    print(f'[Loader] ✓ {_active_subj} loaded.', flush=True)
    print_vram()

    return _active_model

# ── Get model ──────────────────────────────────────────────────────────────

def get_model(subject: str):
    """
    Returns (model, tokenizer) for the given subject.

    - Same subject as last time → instant return (no VRAM change)
    - Different subject         → unload current, load new (~5s on GPU)
    """
    global _active_model, _active_subj

    adapter_key = _find_adapter_key(subject) or _normalise(subject)

    # Cache hit — already loaded
    if _active_subj == adapter_key and _active_model is not None:
        print(f'[Loader] Cache hit: {subject}', flush=True)
        return _active_model, _tokenizer

    # Cache miss — load the new subject
    print(f'[Loader] Cache miss: {subject} (current: {_active_subj})', flush=True)
    model = _load_subject(subject)
    return model, _tokenizer

# ── Inference ──────────────────────────────────────────────────────────────

def generate_response(subject: str, prompt: str) -> str:
    import time

    model, tokenizer = get_model(subject)

    if model is None:
        raise RuntimeError('No model loaded.')

    formatted = (
        f"<|im_start|>user\n"
        f"{prompt}"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    t0 = time.time()

    inputs = tokenizer(
        formatted,
        return_tensors='pt',
        truncation=True,
        max_length=512,
    )

    if DEVICE == 'cuda':
        inputs = {k: v.to('cuda', non_blocking=True) for k, v in inputs.items()}
        torch.cuda.synchronize()

    t1 = time.time()

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
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    t2 = time.time()

    input_length = inputs['input_ids'].shape[1]
    generated    = output[0][input_length:]
    response     = tokenizer.decode(generated, skip_special_tokens=True).strip()

    print(f'[Timing] Inference: {t2-t1:.2f}s | Tokens: {len(generated)} | Total: {t2-t0:.2f}s', flush=True)
    print_vram()

    return response

"""Image generator — local ComfyUI only. Fal.ai STRICT fallback.

Reference: ~/echo-os/echo-3/atlas-rules.md, Theme 7

Default behavior:
  Local ComfyUI (Klein 9B on Mac Mini at echo-world.local:8188) — free, ~3-4 min.
  If local fails, the run FAILS. No automatic Fal fallback.

Fal fallback (DISABLED by default):
  Only activated when allow_fal=True is passed explicitly.
  This requires direct instruction from Natalie. Never auto-fallback.
  Chain: Fal Klein 9B edit → Grok Imagine edit → Seedream v4 edit.

Modes:
  - "edit"           : image-to-image; requires reference image_url. Local only by default.
  - "text-to-image"  : text-only; Fal-only (requires allow_fal=True).

Returns a file:// path (local) or URL (Fal, only if explicitly allowed).
"""
from __future__ import annotations

import json
import mimetypes
import os
import random
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────

COMFY_URL = "http://echo-world.local:8188"

# Klein 9B local
UNET_FILE = "flux-2-klein-9b-Q5_K_M.gguf"
CLIP_FILE = "qwen_3_8b_fp8mixed.safetensors"
VAE_FILE = "flux2-vae.safetensors"
STEPS = 4
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1440

FAL_API_KEY = os.getenv("FAL_API_KEY")

# Fal fallback chain — edit (image-to-image) variants:
FAL_EDIT_MODELS = [
    "fal-ai/flux-2/klein/9b/edit",
    "xai/grok-imagine-image/edit",
    "fal-ai/bytedance/seedream/v4/edit",
]

# Fal fallback chain — text-to-image variants (not used by current pipeline):
FAL_TEXT_TO_IMAGE_MODELS = [
    "fal-ai/flux-2/klein/9b",
    "xai/grok-imagine-image",
    "fal-ai/bytedance/seedream/v4/text-to-image",
]

FAL_POLL_INTERVAL_SEC = 2
FAL_POLL_TIMEOUT_SEC = 600

FAL_EDIT_MODEL_PARAMS = {
    "fal-ai/flux-2/klein/9b/edit": {
        "image_size": {"width": 1080, "height": 1440},
        "enable_safety_checker": False,
    },
    "xai/grok-imagine-image/edit": {
        "aspect_ratio": "3:4",
    },
    "fal-ai/bytedance/seedream/v4/edit": {
        "image_size": {"width": 1080, "height": 1440},
    },
}


# ── HTTP helpers ──────────────────────────────────────────────────────────

def _http_post_json(url, body, headers=None):
    hdrs = headers or {}
    hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=hdrs)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _http_get_json(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _http_download(url, dest, headers=None):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        Path(dest).write_bytes(r.read())


def _http_post_multipart(url, fields, files):
    boundary = f"----comfy-{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        body += str(value).encode() + b"\r\n"
    for name, (filename, data, mime) in files.items():
        body += f"--{boundary}\r\n".encode()
        body += (f'Content-Disposition: form-data; name="{name}"; '
                 f'filename="{filename}"\r\n').encode()
        body += f"Content-Type: {mime}\r\n\r\n".encode()
        body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url, data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


# ── Local ComfyUI ─────────────────────────────────────────────────────────

def _comfy_is_available() -> bool:
    """Check if ComfyUI is reachable and has GGUF nodes."""
    try:
        info = _http_get_json(f"{COMFY_URL}/object_info")
        return "UnetLoaderGGUF" in info
    except (urllib.error.URLError, OSError):
        return False


def _comfy_upload_image(local_path: str) -> str:
    """Upload a local image file to ComfyUI. Returns server-side filename."""
    p = Path(local_path)
    mime, _ = mimetypes.guess_type(str(p))
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    data = p.read_bytes()
    resp = _http_post_multipart(
        f"{COMFY_URL}/upload/image",
        fields={"overwrite": "true", "type": "input"},
        files={"image": (p.name, data, mime)},
    )
    return resp.get("name", p.name)


def _comfy_build_edit_workflow(prompt, ref_filename, steps, seed):
    """Klein 9B edit workflow — fixed 1080x1440 output."""
    return {
        "76": {"class_type": "LoadImage",
               "inputs": {"image": ref_filename}},
        "9":  {"class_type": "SaveImage",
               "inputs": {"filename_prefix": "AtlasGen/Critter", "images": ["65", 0]}},
        "80": {"class_type": "ImageScaleToTotalPixels",
               "inputs": {"image": ["76", 0], "upscale_method": "nearest-exact",
                          "megapixels": 1.0, "resolution_steps": 1}},
        "70": {"class_type": "UnetLoaderGGUF",
               "inputs": {"unet_name": UNET_FILE}},
        "71": {"class_type": "CLIPLoader",
               "inputs": {"clip_name": CLIP_FILE,
                          "type": "flux2", "device": "default"}},
        "72": {"class_type": "VAELoader",
               "inputs": {"vae_name": VAE_FILE}},
        "124": {"class_type": "VAEEncode",
                "inputs": {"pixels": ["80", 0], "vae": ["72", 0]}},
        "74": {"class_type": "CLIPTextEncode",
               "inputs": {"clip": ["71", 0], "text": prompt}},
        "82": {"class_type": "ConditioningZeroOut",
               "inputs": {"conditioning": ["74", 0]}},
        "125": {"class_type": "ReferenceLatent",
                "inputs": {"conditioning": ["74", 0], "latent": ["124", 0]}},
        "123": {"class_type": "ReferenceLatent",
                "inputs": {"conditioning": ["82", 0], "latent": ["124", 0]}},
        "61": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "62": {"class_type": "Flux2Scheduler",
               "inputs": {"steps": steps, "width": OUTPUT_WIDTH, "height": OUTPUT_HEIGHT}},
        "63": {"class_type": "CFGGuider",
               "inputs": {"model": ["70", 0], "positive": ["125", 0],
                          "negative": ["123", 0], "cfg": 1.0}},
        "73": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "66": {"class_type": "EmptyFlux2LatentImage",
               "inputs": {"width": OUTPUT_WIDTH, "height": OUTPUT_HEIGHT, "batch_size": 1}},
        "64": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["73", 0], "guider": ["63", 0],
                          "sampler": ["61", 0], "sigmas": ["62", 0],
                          "latent_image": ["66", 0]}},
        "65": {"class_type": "VAEDecode",
               "inputs": {"samples": ["64", 0], "vae": ["72", 0]}},
    }


def _comfy_poll(prompt_id, timeout_s=900):
    """Poll until done. Returns history entry or None."""
    start = time.time()
    while True:
        time.sleep(2)
        try:
            history = _http_get_json(f"{COMFY_URL}/history/{prompt_id}")
        except (urllib.error.URLError, OSError):
            if time.time() - start > timeout_s:
                return None
            continue
        if prompt_id in history:
            return history[prompt_id]
        if time.time() - start > timeout_s:
            return None


def _comfy_generate_edit(prompt: str, image_url: str) -> str | None:
    """Generate via local ComfyUI. Downloads ref image, uploads to ComfyUI,
    runs edit workflow, saves output to temp file. Returns file:// URL or None."""
    print(f"[img-gen] TRYING LOCAL ComfyUI (Klein 9B)", flush=True, file=sys.stderr)

    if not _comfy_is_available():
        print(f"[img-gen] ComfyUI not reachable at {COMFY_URL}", flush=True, file=sys.stderr)
        return None

    # Download reference image to temp file
    try:
        ext = Path(urllib.parse.urlparse(image_url).path).suffix or ".jpg"
        if ext.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            ext = ".jpg"
        ref_path = Path(tempfile.mktemp(suffix=ext, prefix="comfy-ref-"))
        _http_download(image_url, ref_path)
    except Exception as e:
        print(f"[img-gen] LOCAL failed to download ref image: {e}", flush=True, file=sys.stderr)
        return None

    # Upload to ComfyUI
    try:
        comfy_filename = _comfy_upload_image(str(ref_path))
    except Exception as e:
        print(f"[img-gen] LOCAL failed to upload to ComfyUI: {e}", flush=True, file=sys.stderr)
        return None

    # Build and submit workflow
    seed = random.randint(0, 2**63 - 1)
    workflow = _comfy_build_edit_workflow(prompt, comfy_filename, STEPS, seed)
    try:
        resp = _http_post_json(f"{COMFY_URL}/prompt", {"prompt": workflow})
    except Exception as e:
        print(f"[img-gen] LOCAL submit error: {e}", flush=True, file=sys.stderr)
        return None

    if resp.get("node_errors"):
        print(f"[img-gen] LOCAL ComfyUI rejected prompt: {json.dumps(resp['node_errors'])[:400]}", flush=True, file=sys.stderr)
        return None

    prompt_id = resp["prompt_id"]
    print(f"[img-gen] LOCAL queued: {prompt_id}", flush=True, file=sys.stderr)

    # Poll
    result = _comfy_poll(prompt_id)
    if not result:
        print(f"[img-gen] LOCAL timed out or failed", flush=True, file=sys.stderr)
        return None

    images = result.get("outputs", {}).get("9", {}).get("images", [])
    if not images:
        print(f"[img-gen] LOCAL no images in result", flush=True, file=sys.stderr)
        return None

    # Download output to temp file
    try:
        img_meta = images[0]
        params = urllib.parse.urlencode({
            "filename": img_meta["filename"],
            "subfolder": img_meta.get("subfolder", ""),
            "type": img_meta.get("type", "output"),
        })
        out_path = Path(tempfile.mktemp(suffix=".png", prefix="comfy-out-"))
        _http_download(f"{COMFY_URL}/view?{params}", out_path)
    except Exception as e:
        print(f"[img-gen] LOCAL failed to download output: {e}", flush=True, file=sys.stderr)
        return None

    # Clean up ref temp file
    ref_path.unlink(missing_ok=True)

    file_url = f"file://{out_path}"
    print(f"[img-gen] LOCAL SUCCESS: {file_url}", flush=True, file=sys.stderr)
    return file_url


# ── Fal.ai fallback ──────────────────────────────────────────────────────

def _fal_headers() -> dict:
    if not FAL_API_KEY:
        raise RuntimeError("FAL_API_KEY not set in environment")
    return {
        "Authorization": f"Key {FAL_API_KEY}",
        "Content-Type": "application/json",
    }


def _fal_submit_and_wait(model: str, payload: dict) -> dict | None:
    """Submit to Fal queue, poll until COMPLETED. Returns result dict or None."""
    submit_url = f"https://queue.fal.run/{model}"
    try:
        sub = _http_post_json(submit_url, payload, headers=_fal_headers())
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
        print(f"[img-gen] FAL submit ERROR with {model}: {e}", flush=True, file=sys.stderr)
        return None

    status_url = sub.get("status_url")
    response_url = sub.get("response_url")
    if not status_url or not response_url:
        print(f"[img-gen] FAL unexpected submit response from {model}: {sub}", flush=True, file=sys.stderr)
        return None

    start = time.time()
    while True:
        time.sleep(FAL_POLL_INTERVAL_SEC)
        try:
            st = _http_get_json(status_url, headers=_fal_headers())
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"[img-gen] FAL status check transient error on {model}: {e}", flush=True, file=sys.stderr)
            continue
        status = st.get("status")
        if status == "COMPLETED":
            break
        if status in ("FAILED", "CANCELED", "ERROR"):
            print(f"[img-gen] FAL {model} returned status={status}: {st}", flush=True, file=sys.stderr)
            return None
        if time.time() - start > FAL_POLL_TIMEOUT_SEC:
            print(f"[img-gen] FAL {model} timed out after {FAL_POLL_TIMEOUT_SEC}s", flush=True, file=sys.stderr)
            return None

    try:
        return _http_get_json(response_url, headers=_fal_headers())
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
        print(f"[img-gen] FAL result fetch ERROR on {model}: {e}", flush=True, file=sys.stderr)
        return None


def _fal_extract_image_url(result: dict) -> str | None:
    """Most Fal models return {'images':[{'url':...}]}; some use {'image':{'url':...}}."""
    if not isinstance(result, dict):
        return None
    if "images" in result and result["images"]:
        first = result["images"][0]
        if isinstance(first, dict) and first.get("url"):
            return first["url"]
        if isinstance(first, str):
            return first
    if "image" in result and isinstance(result["image"], dict) and result["image"].get("url"):
        return result["image"]["url"]
    return None


# ── public API (unchanged signature) ─────────────────────────────────────

def generate_image(prompt: str,
                   image_url: str | None = None,
                   mode: str = "edit",
                   allow_fal: bool = False) -> str:
    """Generate an image via local ComfyUI. Returns file:// path.

    Fal.ai fallback is DISABLED by default. Only used when allow_fal=True,
    which requires explicit instruction from Natalie. Never auto-fallback.

    For mode="edit": local ComfyUI Klein 9B only (unless allow_fal=True).
    For mode="text-to-image": Fal-only (requires allow_fal=True).
    """
    if mode == "edit":
        if not image_url:
            raise ValueError("mode='edit' requires image_url")

        # Try local first (and only, unless Fal explicitly allowed)
        local_result = _comfy_generate_edit(prompt, image_url)
        if local_result:
            return local_result

        if not allow_fal:
            raise RuntimeError(
                "LOCAL COMFYUI FAILED. Fal fallback is disabled. "
                "To allow Fal, pass allow_fal=True (requires explicit instruction from Natalie)."
            )

        # Fal fallback — only reached when explicitly allowed
        print(f"[img-gen] LOCAL failed, Fal EXPLICITLY ALLOWED — falling back to Fal.ai", flush=True, file=sys.stderr)
        fal_payload = {"prompt": prompt, "image_urls": [image_url]}
        for model in FAL_EDIT_MODELS:
            print(f"[img-gen] FAL TRYING: {model}", flush=True, file=sys.stderr)
            run_payload = dict(fal_payload)
            run_payload.update(FAL_EDIT_MODEL_PARAMS.get(model, {}))
            result = _fal_submit_and_wait(model, run_payload)
            if result is None:
                continue
            url = _fal_extract_image_url(result)
            if url:
                print(f"[img-gen] FAL SUCCESS WITH: {model}", flush=True, file=sys.stderr)
                return url
            print(f"[img-gen] FAL {model} returned no image: keys={list(result.keys())}", flush=True, file=sys.stderr)

    elif mode == "text-to-image":
        if image_url:
            raise ValueError("mode='text-to-image' must NOT pass image_url")
        if not allow_fal:
            raise RuntimeError(
                "text-to-image mode requires Fal, but Fal fallback is disabled. "
                "Pass allow_fal=True (requires explicit instruction from Natalie)."
            )
        fal_payload = {"prompt": prompt}
        for model in FAL_TEXT_TO_IMAGE_MODELS:
            print(f"[img-gen] FAL TRYING: {model}", flush=True, file=sys.stderr)
            result = _fal_submit_and_wait(model, fal_payload)
            if result is None:
                continue
            url = _fal_extract_image_url(result)
            if url:
                print(f"[img-gen] FAL SUCCESS WITH: {model}", flush=True, file=sys.stderr)
                return url
            print(f"[img-gen] FAL {model} returned no image: keys={list(result.keys())}", flush=True, file=sys.stderr)

    else:
        raise ValueError(f"unknown mode: {mode!r}")

    raise RuntimeError(f"ALL MODELS FAILED in mode={mode}")

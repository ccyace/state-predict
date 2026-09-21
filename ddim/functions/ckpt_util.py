import os, hashlib
import requests
from tqdm import tqdm

URL_MAP = {
    "cifar10": "https://heibox.uni-heidelberg.de/f/869980b53bf5416c8a28/?dl=1",
    "ema_cifar10": "https://heibox.uni-heidelberg.de/f/2e4f01e2d9ee49bab1d5/?dl=1",
    "lsun_bedroom": "https://heibox.uni-heidelberg.de/f/f179d4f21ebc4d43bbfe/?dl=1",
    "ema_lsun_bedroom": "https://heibox.uni-heidelberg.de/f/b95206528f384185889b/?dl=1",
    "lsun_cat": "https://heibox.uni-heidelberg.de/f/fac870bd988348eab88e/?dl=1",
    "ema_lsun_cat": "https://heibox.uni-heidelberg.de/f/0701aac3aa69457bbe34/?dl=1",
    "lsun_church": "https://heibox.uni-heidelberg.de/f/2711a6f712e34b06b9d8/?dl=1",
    "ema_lsun_church": "https://heibox.uni-heidelberg.de/f/44ccb50ef3c6436db52e/?dl=1",
}
CKPT_MAP = {
    "cifar10": "diffusion_cifar10_model/model-790000.ckpt",
    "ema_cifar10": "ema_diffusion_cifar10_model/model-790000.ckpt",
    "lsun_bedroom": "diffusion_lsun_bedroom_model/model-2388000.ckpt",
    "ema_lsun_bedroom": "ema_diffusion_lsun_bedroom_model/model-2388000.ckpt",
    "lsun_cat": "diffusion_lsun_cat_model/model-1761000.ckpt",
    "ema_lsun_cat": "ema_diffusion_lsun_cat_model/model-1761000.ckpt",
    "lsun_church": "diffusion_lsun_church_model/model-4432000.ckpt",
    "ema_lsun_church": "ema_diffusion_lsun_church_model/model-4432000.ckpt",
}
MD5_MAP = {
    "cifar10": "82ed3067fd1002f5cf4c339fb80c4669",
    "ema_cifar10": "1fa350b952534ae442b1d5235cce5cd3",
    "lsun_bedroom": "f70280ac0e08b8e696f42cb8e948ff1c",
    "ema_lsun_bedroom": "1921fa46b66a3665e450e42f36c2720f",
    "lsun_cat": "bbee0e7c3d7abfb6e2539eaf2fb9987b",
    "ema_lsun_cat": "646f23f4821f2459b8bafc57fd824558",
    "lsun_church": "eb619b8a5ab95ef80f94ce8a5488dae3",
    "ema_lsun_church": "fdc68a23938c2397caba4a260bc2445f",
}


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    # Misconfigured HTTP(S)_PROXY often causes ProxyError/SSL errors here. Default: do not use env proxy.
    # If you must use a corporate proxy: set DIFFUSION_CKPT_TRUST_PROXY=1 before running.
    trust_env = os.environ.get("DIFFUSION_CKPT_TRUST_PROXY", "").lower() in (
        "1",
        "true",
        "yes",
    )
    session = requests.Session()
    get_kwargs = {"stream": True, "timeout": (30, 600)}
    if hasattr(session, "trust_env"):
        session.trust_env = trust_env
    elif not trust_env:
        get_kwargs["proxies"] = {"http": None, "https": None}
    with session.get(url, **get_kwargs) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(len(data))


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root=None, check=False):
    if 'church_outdoor' in name:
        name = name.replace('church_outdoor', 'church')
    assert name in URL_MAP
    # Modify the path when necessary
    cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    root = (
        root
        if root is not None
        else os.path.join(cachedir, "diffusion_models_converted")
    )
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


def _strip_module_prefix(state_dict):
    """Remove DataParallel 'module.' prefix if present."""
    if not state_dict:
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module.") :] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def load_ddim_state_dict(ckpt_path, map_location="cpu", prefer_ema=True):
    """
    Load a DDIM/DDPM UNet state_dict from common checkpoint layouts.

    Supports:
      - plain state_dict (optionally with 'module.' prefix)
      - {'state_dict'/'model'/'ema'...: ...}
      - pytorch_diffusion list layout:
          [model(module.), optimizer, epoch, step, ema_state_dict]
    """
    import torch

    raw = torch.load(ckpt_path, map_location=map_location)

    if isinstance(raw, (list, tuple)):
        # Prefer EMA at index 4 when present (standard pytorch_diffusion save).
        candidates = []
        if prefer_ema and len(raw) >= 5 and isinstance(raw[4], dict):
            candidates.append(raw[4])
        for item in raw:
            if isinstance(item, dict) and item and hasattr(next(iter(item.values())), "shape"):
                candidates.append(item)
        if not candidates:
            raise ValueError(f"No state_dict found in list checkpoint: {ckpt_path}")
        state_dict = candidates[0]
    elif isinstance(raw, dict):
        if any(hasattr(v, "shape") for v in raw.values()):
            # Likely already a state_dict (may mix non-tensor meta; keep tensors only later via load).
            nested_keys = ("ema", "ema_state_dict", "state_dict", "model", "model_ema")
            if prefer_ema:
                for key in nested_keys:
                    if key in raw and isinstance(raw[key], dict):
                        state_dict = raw[key]
                        break
                else:
                    state_dict = raw
            else:
                for key in ("state_dict", "model", "ema", "ema_state_dict"):
                    if key in raw and isinstance(raw[key], dict):
                        state_dict = raw[key]
                        break
                else:
                    state_dict = raw
        else:
            raise ValueError(f"Unrecognized dict checkpoint layout: keys={list(raw.keys())[:20]}")
    else:
        raise TypeError(f"Unsupported checkpoint type {type(raw)} from {ckpt_path}")

    # Drop non-tensor entries (e.g. meta ints accidentally mixed in).
    state_dict = {
        k: v for k, v in state_dict.items() if hasattr(v, "shape")
    }
    return _strip_module_prefix(state_dict)

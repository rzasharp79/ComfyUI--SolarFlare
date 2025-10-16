import json
import os
import time
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import torch
from PIL import Image


MODEL_ID = "qwen/qwen-image"
MODEL_VERSION = "905e345fe1dfe10d628daac2140dd8dea471c0d99793ef0fdc46a15c688b62fb"
POLL_INTERVAL = 2.0
MAX_WAIT_SECONDS = 240


def _maybe_strip(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _download_image_tensor(uri: str) -> torch.Tensor:
    response = requests.get(uri, timeout=60)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0)
    return tensor


class QwenImage:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "seed": ("INT", {"default": -1, "min": -1}),
                "image": ("STRING", {"multiline": False, "default": ""}),
                "go_fast": ("BOOLEAN", {"default": True}),
                "guidance": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0}),
                "strength": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "image_size": (["optimize_for_quality", "optimize_for_speed"], {"default": "optimize_for_quality"}),
                "lora_scale": ("FLOAT", {"default": 1.0, "min": 0.0}),
                "aspect_ratio": (
                    ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"],
                    {"default": "2:3"},
                ),
                "lora_weights": ("STRING", {"multiline": False, "default": ""}),
                "output_format": (["webp", "jpg", "png"], {"default": "png"}),
                "enhance_prompt": ("BOOLEAN", {"default": True}),
                "output_quality": ("INT", {"default": 100, "min": 0, "max": 100}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "num_inference_steps": ("INT", {"default": 40, "min": 1, "max": 50}),
                "disable_safety_checker": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "api_token": "STRING",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "output_uris")
    FUNCTION = "generate"
    CATEGORY = "🌅SolarFlare/Replicate/Image Generation"

    def _resolve_token(self, api_token: Optional[str]) -> str:
        token = _maybe_strip(api_token) or _maybe_strip(os.environ.get("REPLICATE_API_TOKEN", ""))
        if not token:
            raise RuntimeError("Replicate API token is required (set REPLICATE_API_TOKEN or pass it via hidden input).")
        return token

    def _start_prediction(self, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post("https://api.replicate.com/v1/predictions", headers=headers, json=payload, timeout=30)
        if response.status_code >= 400:
            raise RuntimeError(f"Failed to start prediction ({response.status_code}): {response.text}")
        return response.json()

    def _poll_prediction(self, url: str, headers: Dict[str, str]) -> Dict[str, Any]:
        deadline = time.time() + MAX_WAIT_SECONDS
        while True:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code >= 400:
                raise RuntimeError(f"Failed to poll prediction ({response.status_code}): {response.text}")
            data = response.json()
            status = data.get("status")
            if status in {"succeeded", "failed", "canceled"}:
                return data
            if time.time() > deadline:
                raise TimeoutError("Timed out waiting for Replicate prediction.")
            time.sleep(POLL_INTERVAL)

    def generate(
        self,
        api_token: str = "",
        prompt: str = "",
        seed: int = -1,
        image: str = "",
        go_fast: bool = True,
        guidance: float = 3.0,
        strength: float = 1.0,
        image_size: str = "optimize_for_quality",
        lora_scale: float = 1.0,
        aspect_ratio: str = "2:3",
        lora_weights: str = "",
        output_format: str = "png",
        enhance_prompt: bool = True,
        output_quality: int = 100,
        negative_prompt: str = "",
        num_inference_steps: int = 40,
        disable_safety_checker: bool = True,
    ):
        if not prompt.strip():
            raise ValueError("Prompt is required.")

        token = self._resolve_token(api_token)
        headers = {
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
        }

        input_payload: Dict[str, Any] = {
            "prompt": prompt,
            "go_fast": go_fast,
            "guidance": guidance,
            "strength": strength,
            "image_size": image_size,
            "lora_scale": lora_scale,
            "aspect_ratio": aspect_ratio,
            "output_format": output_format,
            "enhance_prompt": enhance_prompt,
            "output_quality": output_quality,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "disable_safety_checker": disable_safety_checker,
        }

        if seed >= 0:
            input_payload["seed"] = seed

        image_uri = _maybe_strip(image)
        if image_uri:
            input_payload["image"] = image_uri

        lora_uri = _maybe_strip(lora_weights)
        if lora_uri:
            input_payload["lora_weights"] = lora_uri

        payload = {
            "version": MODEL_VERSION,
            "input": input_payload,
        }

        prediction = self._start_prediction(headers, payload)
        poll_url = prediction.get("urls", {}).get("get")
        if not poll_url:
            raise RuntimeError("Replicate response did not include a polling URL.")

        prediction = self._poll_prediction(poll_url, headers)
        if prediction.get("status") != "succeeded":
            error_message = prediction.get("error") or "Prediction did not succeed."
            raise RuntimeError(error_message)

        output_uris: List[str] = prediction.get("output") or []
        if not output_uris:
            raise RuntimeError("Replicate returned no output URIs.")

        tensor = _download_image_tensor(output_uris[0])
        return (tensor, json.dumps(output_uris))


NODE_CLASS_MAPPINGS = {
    "QwenImage": QwenImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImage": "Qwen Image",
}

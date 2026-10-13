import os
import time

from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass
import base64
import httpx

from openai import OpenAI

from PIL import Image

@dataclass
class ModelConfig:
    """Configuration for model loading and inference."""
    model_path: str
    model_type: str
    device: str = "cuda"
    torch_dtype: str = "bfloat16"  # Use string representation instead of torch.dtype
    max_new_tokens: int = 2048
    temperature: float = 0.2
    top_p: Optional[float] = None
    num_beams: int = 1
    model_version: Optional[str] = None  # Added for DeepSeek model version support
    reasoning_effort: Optional[str] = None
    thinking_type: Optional[str] = None
    thinking_budget_tokens: Optional[int] = None
    max_video_frames: Optional[int] = None  # Maximum number of frames to process from video
    video_fps: Optional[int] = None  # Target FPS for video processing

class VisionLanguageModel:
    """Base class for vision-language model inference."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the appropriate model based on configuration."""
        raise NotImplementedError
    
    def process_input(self, text: str, images: Union[str, List[str]], system_text: Optional[str] = None) -> Any:
        """Process input for model inference."""
        raise NotImplementedError
    
    def generate(self, text: str, images: Union[str, List[str]], system_text: Optional[str] = None) -> str:
        """Generate response from the model."""
        raise NotImplementedError

    def _log_timing(
        self,
        stage_name: str,
        prepare_time: float,
        request_time: float,
        total_time: float,
    ) -> None:
        print(
            f"[{stage_name} Timing] "
            f"input_prepare/upload_ready={prepare_time:.3f}s, "
            f"api_round_trip={request_time:.3f}s, "
            f"total={total_time:.3f}s"
        )

class QwenVLModel(VisionLanguageModel):
    """Implementation for Qwen-based models."""
    
    def __init__(self, config: ModelConfig):
        self.max_image_size = 512  # Default max image dimension
        # Import torch and related libraries only when initializing Qwen model
        import torch
        from transformers import (
            AutoProcessor,
            AutoModelForImageTextToText,
            Qwen2VLForConditionalGeneration,
            Qwen2_5_VLForConditionalGeneration,
        )
        from qwen_vl_utils import process_vision_info
        
        # Store imported modules as instance variables
        self.torch = torch
        self.AutoProcessor = AutoProcessor
        self.AutoModelForImageTextToText = AutoModelForImageTextToText
        self.Qwen2VLForConditionalGeneration = Qwen2VLForConditionalGeneration
        self.Qwen2_5_VLForConditionalGeneration = Qwen2_5_VLForConditionalGeneration
        self.process_vision_info = process_vision_info
        
        super().__init__(config)
    
    def _load_model(self) -> None:
        if "2.5" in self.config.model_path:
            model_class = self.Qwen2_5_VLForConditionalGeneration
        elif "3" in self.config.model_path:
            model_class = self.AutoModelForImageTextToText
        else:
            model_class = self.Qwen2VLForConditionalGeneration
            
        # Convert string dtype to actual torch.dtype
        torch_dtype_map = {
            "bfloat16": self.torch.bfloat16,
            "float16": self.torch.float16,
            "float32": self.torch.float32,
            "float64": self.torch.float64
        }
        torch_dtype = torch_dtype_map.get(self.config.torch_dtype, self.torch.bfloat16)
        self.model = model_class.from_pretrained(
            self.config.model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        self.processor = self.AutoProcessor.from_pretrained(self.config.model_path)
    
    def _resize_image(self, image_path: str) -> Image.Image:
        """Resize image to limit memory usage while maintaining aspect ratio."""
        img = Image.open(image_path).convert('RGB')
        width, height = img.size
        
        # Calculate new dimensions while maintaining aspect ratio
        if width > height:
            if width > self.max_image_size:
                new_width = self.max_image_size
                new_height = int(height * (self.max_image_size / width))
        else:
            if height > self.max_image_size:
                new_height = self.max_image_size
                new_width = int(width * (self.max_image_size / height))
        
        # Only resize if the image is larger than max_image_size
        if width > self.max_image_size or height > self.max_image_size:
            img = img.resize((new_width, new_height), Image.LANCZOS)
            print(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        return img
    
    def process_input(self, text: str, data_path: Union[str, List[str]], system_text: Optional[str] = None) -> Any:
        # Prepare input based on media type
        is_video = isinstance(data_path, str) and data_path.lower().endswith(('.mp4', '.avi', '.mov'))
        
        # Process different media types
        if is_video:
            message = self._get_video_message(text, data_path, system_text)
        else:
            # For images, resize first
            img = self._resize_image(data_path)
            message = self._get_image_message(text, img, system_text)
        
        # Common processing for both media types
        text = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = self.process_vision_info(message, return_video_kwargs=True)
        
        # Remove fps if present and not needed (to avoid compatibility issues)
        if 'fps' in video_kwargs:
            print(f"Removing fps from video_kwargs: {video_kwargs['fps']}")
            del video_kwargs['fps']
        
        return self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self.model.device)
    
    def generate(self, text: str, data_path: Union[str, List[str]], system_text: Optional[str] = None) -> str:
        inputs = self.process_input(text, data_path, system_text)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature if self.config.temperature else 0.2,
            top_p=self.config.top_p if self.config.top_p else None,
            num_beams=self.config.num_beams if self.config.num_beams else 1
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
    
    
    def _get_image_message(self, text: str, data_path: Union[str, Image.Image], system_text: Optional[str] = None) -> List[Dict]:
        content = [
            {"type": "image", "image": data_path},
            {"type": "text", "text": text}
        ]
        
        if system_text:
            return [
                {"role": "system", "content": system_text},
                {"role": "user", "content": content}
            ]
        else:
            return [
                {"role": "user", "content": content}
            ]
    
    def _get_video_message(self, text: str, data_path: Union[str, List[str]], system_text: Optional[str] = None) -> List[Dict]:
        """
        Create a message format for video input.
        
        Args:
            text: The text prompt to accompany the video
            data_path: Either a path to a video file or a list of image paths
                      representing video frames
        
        Returns:
            A formatted message for the model
        """
        # Handle different video input formats
        if isinstance(data_path, list):
            # List of image frames
            content = [
                {
                    "type": "video",
                    "video": data_path,
                },
                {"type": "text", "text": text},
            ]
        else:
            # Single video file path
            is_url = data_path.startswith(('http://', 'https://', 'file:///'))
            
            video_info = {
                "type": "video",
                "video": data_path,
                "max_pixels": 360 * 360,  # Default settings
            }
            
            # Add frame control parameters if specified in config
            if self.config.max_video_frames is not None:
                video_info["max_frames"] = self.config.max_video_frames
            if self.config.video_fps is not None:
                video_info["fps"] = self.config.video_fps
            
            content = [
                video_info,
                {"type": "text", "text": text},
            ]
        
        if system_text:
            return [
                {"role": "system", "content": system_text},
                {"role": "user", "content": content}
            ]
        else:
            return [
                {"role": "user", "content": content}
            ]

class DeepSeekAPIModel(VisionLanguageModel):
    """Implementation for DeepSeek API-based models."""
    
    def __init__(self, config: ModelConfig, api_key: str):
        super().__init__(config)  # Only pass config to parent class
        self.api_key = api_key
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"  # DeepSeek API endpoint
        )
        self.model_version = config.model_version or "deepseek-r1"  # Default to R1 if not specified
    
    def _load_model(self) -> None:
        pass  # DeepSeek models are accessed via API
    
    def process_input(self, text: str, images: Union[str, List[str]], system_text: Optional[str] = None) -> List[Dict]:
        # DeepSeek models only process text input
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": text}
        ]
        # Remove system message if system_text is None
        if system_text is None:
            messages.pop(0)
        return messages
    
    def generate(self, text: str, images: Union[str, List[str]], system_text: Optional[str] = None) -> str:
        total_start = time.perf_counter()
        prepare_start = total_start
        messages = self.process_input(text, images, system_text)
        prepare_time = time.perf_counter() - prepare_start
        try:
            request_start = time.perf_counter()
            result = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False
            )
            request_time = time.perf_counter() - request_start
            total_time = time.perf_counter() - total_start
            self._log_timing("DeepSeek API", prepare_time, request_time, total_time)
            return result.choices[0].message.content
        except Exception as e:
            print(f"Error calling DeepSeek API: {e}")
            return "Error generating response from DeepSeek API"

class OpenAIAPIModel(VisionLanguageModel):
    """Implementation for OpenAI API-based models."""
    
    def __init__(self, config: ModelConfig, api_key: str):
        super().__init__(config)  # Only pass config to parent class
        self.api_key = api_key
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.openai.com/v1"  # OpenAI API endpoint
        )
        self.model_version = config.model_version or "gpt-5"  # Default to gpt-4o if not specified
    
    def _load_model(self) -> None:
        pass  # OpenAI models are accessed via API
    
    def process_input(self, text: str, images: Union[str, List[str]], system_text: Optional[str] = None) -> List[Dict]:
        # OpenAI models only process text input
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": text}
        ]
        # Remove system message if system_text is None
        if system_text is None:
            messages.pop(0)
        return messages
    
    def generate(self, text: str, images: Union[str, List[str]], system_text: Optional[str] = None) -> str:
        total_start = time.perf_counter()
        prepare_start = total_start
        messages = self.process_input(text, images, system_text)
        prepare_time = time.perf_counter() - prepare_start
        try:
            request_start = time.perf_counter()
            result = self.client.chat.completions.create(
                model=self.model_version,  # Use the specified model version
                messages=messages,
                stream=False
            )
            request_time = time.perf_counter() - request_start
            total_time = time.perf_counter() - total_start
            self._log_timing("OpenAI API", prepare_time, request_time, total_time)
            return result.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return "Error generating response from OpenAI API"

class GrokAPIModel(VisionLanguageModel):
    """Implementation for Grok API-based models."""
    
    def __init__(self, config: ModelConfig, api_key: str):
        super().__init__(config)  # Only pass config to parent class
        self.api_key = api_key
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"  # Grok API endpoint
        )
        self.model_version = config.model_version or "grok-4-latest"  # Default to grok-4-latest if not specified
    
    def _load_model(self) -> None:
        pass  # Grok models are accessed via API
    
    def process_input(self, text: str, images: Union[str, List[str]], system_text: Optional[str] = None) -> List[Dict]:
        # Grok models only process text input
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": text}
        ]
        # Remove system message if system_text is None
        if system_text is None:
            messages.pop(0)
        return messages
    
    def generate(self, text: str, images: Union[str, List[str]], system_text: Optional[str] = None) -> str:
        total_start = time.perf_counter()
        prepare_start = total_start
        messages = self.process_input(text, images, system_text)
        prepare_time = time.perf_counter() - prepare_start
        try:
            request_start = time.perf_counter()
            result = self.client.chat.completions.create(
                model=self.model_version,  # Use the specified model version
                messages=messages,
                stream=False,
                temperature=self.config.temperature  # Use temperature from config
            )
            request_time = time.perf_counter() - request_start
            total_time = time.perf_counter() - total_start
            self._log_timing("Grok API", prepare_time, request_time, total_time)
            return result.choices[0].message.content
        except Exception as e:
            print(f"Error calling Grok API: {e}")
            return "Error generating response from Grok API"

class GeminiAPIModel(VisionLanguageModel):
    """Implementation for Gemini API-based models."""
    
    def __init__(self, config: ModelConfig, api_key: str):
        # Import genai only when initializing Gemini model
        from google import genai
        
        # Store imported module as instance variable
        self.genai = genai
        
        super().__init__(config)  # Only pass config to parent class
        self.api_key = api_key
        self.client = self.genai.Client(api_key=api_key)  # Pass API key directly to Client constructor
        self.model_version = config.model_version or "gemini-3-pro-preview"  # Default to gemini-3-pro-preview if not specified
    
    def _load_model(self) -> None:
        pass  # Gemini models are accessed via API
    
    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def process_input(self, text: str, images: Union[str, List[str]], system_text: Optional[str] = None) -> List[Dict]:
        # Gemini models process multimodal input
        contents = []
        
        # Add system message if provided
        if system_text:
            contents.append({
                "role": "user",
                "parts": [{"text": system_text}]
            })
        
        # Build user message with text and images
        parts = [{"text": text}]
        
        # Process images if provided
        if images:
            if isinstance(images, str):
                images = [images]
            for img in images:
                # For Gemini API, we need to use the correct format
                # The genai library handles image loading directly from paths
                parts.append({"image": img})
        
        # Add user message
        contents.append({
            "role": "user",
            "parts": parts
        })
        
        return contents
    
    def generate(self, text: str, images: Union[str, List[str]], system_text: Optional[str] = None) -> str:
        total_start = time.perf_counter()
        prepare_start = total_start
        contents = self.process_input(text, images, system_text)
        prepare_time = time.perf_counter() - prepare_start
        try:
            # For Gemini API, we need to use the correct method signature
            # The generate_content method expects the content in the right format
            model = self.genai.GenerativeModel(self.model_version)
            request_start = time.perf_counter()
            response = model.generate_content(
                contents=contents,
                generation_config=self.genai.types.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_new_tokens
                )
            )
            request_time = time.perf_counter() - request_start
            total_time = time.perf_counter() - total_start
            self._log_timing("Gemini API", prepare_time, request_time, total_time)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return "Error generating response from Gemini API"

def _env_value(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _normalize_openai_base_url(base_url: str) -> str:
    base = (base_url or "").strip().rstrip("/")
    if base and not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def _reject_qwen_key_for_doubao(api_key: str) -> None:
    qwen_key = _env_value("MODEL_KEY")
    if qwen_key and (api_key or "").strip() == qwen_key:
        raise ValueError(
            "Doubao must not use MODEL_KEY. Export DOUBAO_API_KEY with the Doubao key."
        )


class DoubaoAPIModel(VisionLanguageModel):
    backend_name = "Doubao"
    default_model_version = "doubao-seed-1-6-251015"

    def __init__(self, config: ModelConfig, api_key: str, base_url: Optional[str] = None):
        super().__init__(config)
        if self.backend_name == "Doubao":
            _reject_qwen_key_for_doubao(api_key)
            base_url = base_url or os.getenv("DOUBAO_BASE_URL", "https://124.220.71.236:8000/v1")
        if not base_url:
            raise ValueError(f"{self.backend_name} API requires a base URL")
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            http_client=httpx.Client(verify=False, follow_redirects=True)
        )

        self.model_version = config.model_version or self.default_model_version

    def _load_model(self) -> None:
        pass  # Doubao models are accessed via API

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def process_input(self, text, images, system_text=None):
        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})

        user_content = [{"type": "text", "text": text}]
        if images:
            if isinstance(images, str):
                images = [images]
            for img in images:
                b64 = self._encode_image(img)
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}"
                    }
                })
        messages.append({
            "role": "user",
            "content": user_content
        })
        return messages

    @staticmethod
    def _extract_response_text(result) -> str:
        if not hasattr(result, "choices") or not result.choices:
            return str(result)

        message = result.choices[0].message
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(str(item.get("text", "")))
                else:
                    item_text = getattr(item, "text", None)
                    item_type = getattr(item, "type", None)
                    if item_type == "text" and item_text:
                        text_parts.append(str(item_text))
            if text_parts:
                return "".join(text_parts)
        return str(message)

    def generate(self, text: str, data_path: Union[str, List[str]], system_text: Optional[str] = None) -> str:
        total_start = time.perf_counter()
        prepare_start = total_start
        messages = self.process_input(text, data_path, system_text)
        prepare_time = time.perf_counter() - prepare_start
        try:
            request_kwargs = {
                "model": self.model_version,
                "temperature": self.config.temperature,
                "messages": messages,
                "stream": False,
            }
            if self.config.max_new_tokens:
                request_kwargs["max_tokens"] = self.config.max_new_tokens
            if self.config.top_p is not None:
                request_kwargs["top_p"] = self.config.top_p
            if self.config.thinking_type:
                thinking_payload = {"type": self.config.thinking_type}
                if (
                    self.config.thinking_type == "enabled"
                    and self.config.thinking_budget_tokens is not None
                ):
                    thinking_payload["budget_tokens"] = self.config.thinking_budget_tokens
                request_kwargs["extra_body"] = {"thinking": thinking_payload}
            elif self.config.reasoning_effort:
                request_kwargs["reasoning_effort"] = self.config.reasoning_effort

            request_start = time.perf_counter()
            result = self.client.chat.completions.create(**request_kwargs)
            request_time = time.perf_counter() - request_start
            total_time = time.perf_counter() - total_start
            self._log_timing(f"{self.backend_name} API", prepare_time, request_time, total_time)
            return self._extract_response_text(result)
        except Exception as e:
            raise RuntimeError(
                f"{self.backend_name} API failed via OpenAI-compatible chat.completions at {self.base_url}: {e}"
            )


class QwenAPIModel(DoubaoAPIModel):
    backend_name = "Qwen"
    default_model_version = "qwen3.7-plus"

    def __init__(self, config: ModelConfig, api_key: str):
        base_url = _normalize_openai_base_url(_env_value("BASE_URL"))
        if not base_url:
            raise ValueError("BASE_URL is required for Qwen API")
        super().__init__(config, api_key, base_url=base_url)
        if not config.model_version:
            self.model_version = _env_value("MODEL_NAME") or self.default_model_version


def get_model(config: ModelConfig, api_key: Optional[str] = None, 
            min_pixels: Optional[int] = None, max_pixels: Optional[int] = None) -> VisionLanguageModel:
    """Factory function to create appropriate model instance."""
    model_type = config.model_type.lower().replace("_", "-")
    if model_type in ("qwen-api", "qwenapi"):
        key = (api_key or "").strip() or _env_value("MODEL_KEY")
        if not key:
            raise ValueError("MODEL_KEY is required for Qwen API")
        return QwenAPIModel(config, key)
    if "qwen" in model_type:
        return QwenVLModel(config)
    elif "deepseek" in model_type:
        if not api_key:
            raise ValueError("API key is required for DeepSeek models")
        if config.model_version and config.model_version not in ["deepseek-r1", "deepseek-chat"]:
            raise ValueError(f"Unsupported DeepSeek model version: {config.model_version}. Supported versions are: deepseek-r1, deepseek-v3")
        return DeepSeekAPIModel(config, api_key)
    elif "openai" in model_type:
        if not api_key:
            raise ValueError("API key is required for OpenAI models")
        return OpenAIAPIModel(config, api_key)
    elif "grok" in model_type:
        if not api_key:
            raise ValueError("API key is required for Grok models")
        return GrokAPIModel(config, api_key)
    elif "gemini" in model_type:
        if not api_key:
            raise ValueError("API key is required for Gemini models")
        return GeminiAPIModel(config, api_key)
    elif "doubao" in model_type:
        if not api_key:
            raise ValueError("API key is required for Doubao models")
        return DoubaoAPIModel(config, api_key)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

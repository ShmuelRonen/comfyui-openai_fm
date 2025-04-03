import json
import os
import torch
import torchaudio
import datetime
import requests
from io import BytesIO

class OPENAIFM:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            voices = cls.load_json_file("voices.json")
            vibes = cls.load_json_file("vibes.json")
            
            # Add verbose logging
            print(f"Loaded {len(voices['voices'])} voices")
            print(f"Loaded {len(vibes.keys())} vibes")
            
        except Exception as e:
            print(f"Error loading voice/vibe data: {e}")
            voices = {"voices": ["ALLoy"]}
            vibes = {"Calm": []}
            
        return {"required": {
            "text": ("STRING", {"multiline": True, "default": "Enter text here"}),
            "voice": (voices["voices"], {"default": "ALLoy"}),
            "vibe": (list(vibes.keys()), {"default": "Calm"}),
        }, "optional": {
            "user_vibe": ("STRING", {"multiline": True, "default": ""})
        }}
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "audio"
    
    @staticmethod
    def load_json_file(file_name):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "data", file_name)
        
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            
        if isinstance(data, list):
            unique_items = {json.dumps(item, sort_keys=True) for item in data}
            data = [json.loads(item) for item in unique_items]
            
        return data
    
    def generate(self, text, voice, vibe, user_vibe=None):
        try:
            # Determine which vibe to use based on priority
            if user_vibe and user_vibe.strip():
                print(f"Using custom user vibe")
                vibe_prompt = user_vibe.strip()
            else:
                print(f"Using vibe: {vibe}")
                vibes_data = self.load_json_file("vibes.json")
                vibe_prompt = self.format_vibe_prompt(vibe, vibes_data)
            
            # Send request to OpenAI FM
            audio_data = self.send_request(text, voice, vibe_prompt)
            
            # Process audio data
            if audio_data:
                # Convert to tensor for ComfyUI
                waveform, sample_rate = self.audio_bytes_to_tensor(audio_data)
                
                # Save audio to a file for easy access
                self.save_audio_file(audio_data)
                
                # Ensure we have a proper 2D tensor 
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                
                # Make sure we have a batch dimension too
                # ComfyUI expects [batch, channels, samples]
                if waveform.dim() == 2:
                    # Add batch dimension if it's missing (channels, samples) -> (1, channels, samples)
                    waveform = waveform.unsqueeze(0)
                    
                print(f"Final audio tensor shape: {waveform.shape}, {sample_rate}Hz")
                
                # Return in ComfyUI-compatible format
                return ({"waveform": waveform, "sample_rate": sample_rate},)
            else:
                print("Failed to generate audio from OpenAI FM")
                # Return empty audio on failure (1 second of silence)
                dummy_waveform = torch.zeros(1, 1, 44100)  # [batch, channels, samples]
                return ({"waveform": dummy_waveform, "sample_rate": 44100},)
                
        except Exception as e:
            print(f"Error generating audio: {e}")
            dummy_waveform = torch.zeros(1, 1, 44100)  # [batch, channels, samples]
            return ({"waveform": dummy_waveform, "sample_rate": 44100},)
    
    def format_vibe_prompt(self, vibe_name, vibes_data):
        default_prompt = "Voice Affect: Calm, composed, and reassuring; project quiet authority and confidence.\n\nTone: Sincere, empathetic, and gently authoritative—express genuine apology while conveying competence.\n\nPacing: Steady and moderate; unhurried enough to communicate care, yet efficient enough to demonstrate professionalism.\n\nEmotion: Genuine empathy and understanding; speak with warmth, especially during apologies (\"I'm very sorry for any disruption...\").\n\nPronunciation: Clear and precise, emphasizing key reassurances (\"smoothly,\" \"quickly,\" \"promptly\") to reinforce confidence.\n\nPauses: Brief pauses after offering assistance or requesting details, highlighting willingness to listen and support."
        
        if not vibes_data or vibe_name not in vibes_data:
            print(f"Vibe '{vibe_name}' not found. Using default prompt.")
            return default_prompt
            
        vibe_content = vibes_data.get(vibe_name)
        if vibe_content:
            return "\n\n".join(vibe_content)
        return default_prompt
    
    def send_request(self, text, voice, vibe_prompt):
        url = "https://www.openai.fm/api/generate"
        boundary = "----WebKitFormBoundary" + ''.join(['0123456789abcdef'[i % 16] for i in range(16)])
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "*/*",
            "Origin": "https://www.openai.fm",
            "Referer": "https://www.openai.fm/",
        }
        
        data = []
        for name, value in [
            ("input", text),
            ("prompt", vibe_prompt),
            ("voice", voice.lower()),
            ("vibe", "null")
        ]:
            data.append(f"--{boundary}")
            data.append(f'Content-Disposition: form-data; name="{name}"')
            data.append("")
            data.append(value)
        
        data.append(f"--{boundary}--")
        body = "\r\n".join(data).encode('utf-8')
        
        try:
            response = requests.post(url, headers=headers, data=body, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get("Content-Type", "")
            if "audio/wav" in content_type or "audio/" in content_type:
                print(f"Received audio data: {len(response.content)} bytes")
                return response.content
            else:
                print(f"API returned unexpected content type: {content_type}")
                print(f"Response preview: {response.text[:100]}...")
                return None
        except Exception as e:
            print(f"API request failed: {e}")
            return None
    
    def audio_bytes_to_tensor(self, audio_bytes):
        try:
            buffer = BytesIO(audio_bytes)
            waveform, sample_rate = torchaudio.load(buffer)
            print(f"Audio loaded: {waveform.shape}, {sample_rate}Hz")
            
            # Handle mono audio loading as 1D array
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)  # Add channel dimension [samples] -> [1, samples]
            
            return waveform, sample_rate
        except Exception as e:
            print(f"Failed to convert audio to tensor: {e}")
            return torch.zeros(1, 44100), 44100
    
    def save_audio_file(self, audio_bytes):
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"openaifm_{timestamp}.wav"
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Try multiple possible paths
            possible_paths = [
                os.path.join(os.path.dirname(os.path.dirname(script_dir)), "output"),
                os.path.join(script_dir, "output"),
                script_dir
            ]
            
            output_dir = None
            for path in possible_paths:
                try:
                    os.makedirs(path, exist_ok=True)
                    output_dir = path
                    break
                except:
                    continue
            
            if output_dir is None:
                output_dir = script_dir
                
            file_path = os.path.join(output_dir, filename)
            with open(file_path, "wb") as f:
                f.write(audio_bytes)
            
            print(f"Audio saved to: {file_path}")
            return file_path
        except Exception as e:
            print(f"Failed to save audio file: {e}")
            return None

# Required for ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "OpenAIFMNode": OPENAIFM
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIFMNode": "OpenAI FM_TTS"
}
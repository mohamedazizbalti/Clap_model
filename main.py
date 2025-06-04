from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import torchaudio
import torch
import torch.nn as nn
from transformers import ClapProcessor, ClapModel
import io
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import fitz

app = FastAPI()

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torchaudio.set_audio_backend("soundfile")
# Load CLAP model
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)

# Define your classifier (adjust architecture as needed)
class MultimodalClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)

classifier = MultimodalClassifier(input_dim=1024, num_classes=4).to(device)
classifier.load_state_dict(torch.load("model.pt", map_location=device))
classifier.eval()

# Load LabelEncoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(
    audio: UploadFile = File(..., description="Audio file in WAV format"),
    text: str = Form(..., description="Text description for classification")
):
    try:
        # Validate audio file
        if not audio.filename.lower().endswith('.wav'):
            return JSONResponse(
                status_code=400,
                content={"error": "Only .wav audio files are supported"}
            )

        # Read and process audio file
        audio_bytes = await audio.read()
        
        # Create a proper file-like object
        audio_buffer = io.BytesIO(audio_bytes)
        
        try:
            # Method 1: Use soundfile directly (more reliable for BytesIO)
            import soundfile as sf
            audio_buffer.seek(0)  # Reset buffer position
            
            # Get file info first to validate
            info = sf.info(audio_buffer)
            print(f"Audio file info - Duration: {info.duration:.2f}s, Sample rate: {info.samplerate}, Channels: {info.channels}")
            
            # Reset buffer and read with proper dtype
            audio_buffer.seek(0)
            waveform_np, sample_rate = sf.read(audio_buffer, dtype='float32')
            
            # Validate reasonable audio length (8 seconds should be ~384k samples at 48kHz)
            expected_samples = int(info.duration * info.samplerate)
            if waveform_np.shape[0] > expected_samples * 2:  # Allow 2x buffer for safety
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Audio file seems corrupted - unexpected sample count"}
                )
            
            # Convert to torch tensor with proper dimensions
            waveform = torch.from_numpy(waveform_np).float()
            
            # Ensure proper tensor shape: (channels, samples)
            if waveform.dim() == 1:
                # Mono audio: (samples,) -> (1, samples)
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() == 2:
                # Stereo audio: (samples, channels) -> (channels, samples)
                waveform = waveform.transpose(0, 1)
            
            # Convert to mono if stereo (take mean of channels)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                
            print(f"Audio loaded - Shape: {waveform.shape}, Sample rate: {sample_rate}, Duration: {waveform.shape[1]/sample_rate:.2f}s")
            
        except Exception as e:
            try:
                # Method 2: Temporary file fallback
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_file_path = tmp_file.name
                
                waveform, sample_rate = torchaudio.load(tmp_file_path)
                os.unlink(tmp_file_path)  # Clean up temp file
                
                # Ensure mono audio
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                    
                print(f"Audio loaded via temp file - Shape: {waveform.shape}, Sample rate: {sample_rate}, Duration: {waveform.shape[1]/sample_rate:.2f}s")
                
            except Exception as e2:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Failed to load audio file: {str(e2)}"}
                )

        # Validate audio dimensions before processing
        if waveform.numel() == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Audio file appears to be empty"}
            )
        
        # Additional validation - 8 seconds at 48kHz should be ~384k samples
        max_expected_samples = 8 * 48000 * 2  # 8 seconds * 48kHz * 2 for safety margin
        if waveform.numel() > max_expected_samples:
            return JSONResponse(
                status_code=400,
                content={"error": f"Audio file too large or corrupted - {waveform.numel()} samples (expected ~{8*sample_rate})"}
            )
        
        # Truncate if audio is longer than expected (just in case)
        max_samples = 8 * sample_rate  # 8 seconds worth of samples
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
            print(f"Truncated audio to 8 seconds: {waveform.shape}")
        
        # Resample if necessary (CLAP typically expects 48kHz)
        target_sample_rate = 48000
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            waveform = resampler(waveform)
            sample_rate = target_sample_rate
            print(f"Resampled to {target_sample_rate}Hz - New shape: {waveform.shape}")

        # Convert to numpy for CLAP processor (squeeze to remove channel dimension)
        audio_numpy = waveform.squeeze(0).numpy()
        print(f"Final audio shape for CLAP: {audio_numpy.shape}, dtype: {audio_numpy.dtype}")
        
        # Process with CLAP processor
        inputs = processor(
            text=[text], 
            audios=audio_numpy,
            sampling_rate=sample_rate, 
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get embeddings and make prediction
        with torch.no_grad():
            # Method 1: Try to get joint embeddings if available
            try:
                # Some CLAP models support joint processing
                inputs = processor(
                    text=[text], 
                    audios=audio_numpy,
                    sampling_rate=sample_rate, 
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Try different methods to get combined features
                if hasattr(clap_model, 'get_text_features') and hasattr(clap_model, 'get_audio_features'):
                    # Get separate embeddings and combine them
                    text_features = clap_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                    audio_features = clap_model.get_audio_features(input_features=inputs['input_features'])
                    
                    print(f"Text features shape: {text_features.shape}")
                    print(f"Audio features shape: {audio_features.shape}")
                    
                    # Check if we need to concatenate or if they're already combined
                    if text_features.shape[1] == 512 and audio_features.shape[1] == 512:
                        # Standard CLAP embedding size, concatenate to get 1024
                        embeddings = torch.cat([text_features, audio_features], dim=1)
                    elif text_features.shape[1] == 1024 or audio_features.shape[1] == 1024:
                        # One of them is already 1024, use that or average
                        embeddings = (text_features + audio_features) / 2 if text_features.shape == audio_features.shape else (text_features if text_features.shape[1] == 1024 else audio_features)
                    else:
                        # Fallback: just use audio features if they match expected dimension
                        embeddings = audio_features if audio_features.shape[1] == 1024 else text_features
                        
                elif hasattr(clap_model, 'forward'):
                    # Try direct forward pass
                    outputs = clap_model(**inputs)
                    # Extract embeddings from outputs (this varies by model version)
                    if hasattr(outputs, 'multimodal_embeds'):
                        embeddings = outputs.multimodal_embeds
                    elif hasattr(outputs, 'pooler_output'):
                        embeddings = outputs.pooler_output
                    else:
                        # Fallback to last hidden state
                        embeddings = outputs.last_hidden_state.mean(dim=1)
                        
                else:
                    raise Exception("Cannot find appropriate method to extract embeddings")
                    
            except Exception as e:
                print(f"Joint embedding failed: {e}")
                # Fallback: use only audio features
                audio_inputs = processor(audios=audio_numpy, sampling_rate=sample_rate, return_tensors="pt")
                audio_inputs = {k: v.to(device) for k, v in audio_inputs.items()}
                embeddings = clap_model.get_audio_features(**audio_inputs)
            
            print(f"Final embeddings shape: {embeddings.shape}")
            
            # Ensure embeddings match expected classifier input dimension
            if embeddings.shape[1] != 1024:
                if embeddings.shape[1] > 1024:
                    # Reduce dimension (simple linear projection)
                    projection = nn.Linear(embeddings.shape[1], 1024).to(device)
                    embeddings = projection(embeddings)
                else:
                    # Pad with zeros if too small
                    padding = torch.zeros(embeddings.shape[0], 1024 - embeddings.shape[1]).to(device)
                    embeddings = torch.cat([embeddings, padding], dim=1)
                print(f"Adjusted embeddings shape: {embeddings.shape}")
            
            # Make prediction
            logits = classifier(embeddings)
            probs = torch.nn.functional.softmax(logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).cpu().numpy()

        label = label_encoder.inverse_transform(pred_label)[0]

        return JSONResponse(content={
            "prediction": label,
            "status": "success",
            "filename": audio.filename
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
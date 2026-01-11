from Speaker.speakerlab.bin.infer import embedding
from CMGAN.inference import enhancement
import torchaudio
import torch


class enhance_and_embeding:
    def __init__(self, noisy, enhancement_model=None, embedding_model=None, feature_extractor=None):
        self.noisy = noisy
        self.enhancement_model = enhancement_model
        self.embedding_model = embedding_model
        self.feature_extractor = feature_extractor

    def encoder(self):
        import os
        import torch
        
        # 1. Enhancement
        if self.enhancement_model is not None:
            # Use pre-loaded CMGAN model
            est_audio = self.enhancement_model.enhance_one_tensor(self.enhancement_model.model, self.noisy)
            est_audio = torch.tensor(est_audio)
        else:
            # Try loading or skip
            model_path = "./model/ckpt"
            if os.path.exists(model_path):
                # This still creates a new instance (inefficient, but kept for compatibility)
                est_audio = enhancement(noisy_path=self.noisy, model_path=model_path).enhance()
            else:
                # print(f"Warning: Enhancement model not found at {model_path}. Skipping enhancement.")
                est_audio = self.noisy
                if isinstance(est_audio, torch.Tensor):
                    if est_audio.dim() > 1:
                         est_audio = est_audio.squeeze()

        torch.cuda.empty_cache()
        
        # 2. Embedding
        # If models are pre-loaded, use them
        embedder = embedding(wav_file=est_audio, model=self.embedding_model, feature_extractor=self.feature_extractor)
        audio_tensor = embedder.compute_embedding()
        return torch.tensor(audio_tensor[-1])


# if __name__ == '__main__':
#     audio, _ = torchaudio.load("data/noisy/1_1_1.wav")
#     audio_tensor = enhance_and_embeding(noisy=audio).encoder()
#     print(type(audio_tensor))
#     print(audio_tensor.shape)
#     print(audio_tensor)

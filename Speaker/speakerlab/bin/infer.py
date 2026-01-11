# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import os
import sys
import torch
import torchaudio

try:
    from speakerlab.process.processor import FBank
    from speakerlab.utils.builder import dynamic_import
except ImportError:
    sys.path.append('%s/../..' % os.path.dirname(__file__))
    from speakerlab.process.processor import FBank
    from speakerlab.utils.builder import dynamic_import

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines.util import is_official_hub_path

ERes2Net_COMMON = {
    'obj': 'speakerlab.models.eres2net.ResNet_aug.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    }
}

supports = {
    'damo/speech_eres2net_sv_zh-cn_16k-common': {
        'revision': 'v1.0.4',
        'model': ERes2Net_COMMON,
        'model_pt': 'pretrained_eres2net_aug.ckpt',
    }
}


class embedding:
    def __init__(self, wav_file, model=None, feature_extractor=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        if model is not None:
            self.embedding_model = model
            self.feature_extractor = feature_extractor
        else:
            model_id = 'damo/speech_eres2net_sv_zh-cn_16k-common'
            conf = supports[model_id]
            
            # Check if local model exists
            pretrained_model = "./model/pretrained_eres2net_aug.ckpt"
            if not os.path.exists(pretrained_model):
                print(f"Local model not found at {pretrained_model}. Downloading from ModelScope...")
                model_dir = snapshot_download(model_id, revision=conf['revision'])
                pretrained_model = os.path.join(model_dir, conf['model_pt'])
                print(f"Model downloaded to {pretrained_model}")

            pretrained_state = torch.load(pretrained_model, map_location=self.device)

            model_conf = conf['model']
            self.embedding_model = dynamic_import(model_conf['obj'])(**model_conf['args'])
            self.embedding_model.load_state_dict(pretrained_state)
            self.embedding_model.to(self.device)
            self.embedding_model.eval()
            self.feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)
            
        self.wav_file = wav_file

    def load_wav(self):
        obj_fs = 16000
        wav_file = self.wav_file
        if type(wav_file) == str:
            wav, fs = torchaudio.load(wav_file)
            if fs != obj_fs:
                print(f'[WARNING]: The sample rate of {wav_file} is not {obj_fs}, resample it.')
                wav, fs = torchaudio.sox_effects.apply_effects_tensor(
                    wav, fs, effects=[['rate', str(obj_fs)]]
                )
            if wav.shape[0] > 1:
                wav = wav[0, :].unsqueeze(0)
            return wav
        elif type(wav_file) == torch.Tensor:
            return wav_file

    def compute_embedding(self):
        wav = self.load_wav()
        feat = self.feature_extractor(wav).unsqueeze(0)
        feat = feat.to(self.device)
        with torch.no_grad():
            embedding = self.embedding_model(feat).detach().cpu().numpy()
            return embedding

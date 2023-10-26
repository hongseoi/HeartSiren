import pandas as pd

from pathlib import Path
from typing import Tuple, Union

import torch
import torchaudio

from torch.utils.data import Dataset


_RELEASE_CONFIGS = {
    'release1': {
        "folder_in_archive": "the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data",
        "url": "https://physionet.org/static/published-projects/circor-heart-sound/the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip"
    }
}


class Phonocardiogram(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        url: str = _RELEASE_CONFIGS['release1']['url'],
        folder_in_archive: str = _RELEASE_CONFIGS['release1']['folder_in_archive'],
        download: bool = False,
    ) -> None:
        self._root = Path(root)
        self._dataset_dir = self._root / folder_in_archive
        
        self._patient_ids = [p.stem for p in self._dataset_dir.glob('*.wav')]
    
    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        patient_id = self._patient_ids[n]
        wav_path = self._dataset_dir / (patient_id + '.wav')
        tsv_path = self._dataset_dir / (patient_id + '.tsv')
        
        waveform, sample_rate = torchaudio.load(wav_path)
        
        labels_df = pd.read_csv(tsv_path, delimiter='\t', header=None)
        s1_ranges = torch.tensor(labels_df[labels_df[2] == 1].drop(2, axis=1).values)
        s2_ranges = torch.tensor(labels_df[labels_df[2] == 3].drop(2, axis=1).values)
        
        return waveform, sample_rate, s1_ranges, s2_ranges
    
    def __len__(self) -> int:
        return len(self._patient_ids)
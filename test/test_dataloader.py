import torch
from avssl.data import (
    get_simple_image_transform,
    FlickrImageCaptionDataset,
    PlacesImageCaptionDataset,
    FlickrDataset,
    collate_general,
)
from torch.utils.data import DataLoader

WORKERS_NUM = 16
BATCH_SZ = 64

flickr_base_path = "/work/vjsalt22/dataset/flickr/"
places_image_base_path = (
    "/work/vjsalt22/dataset/places/image_data/vision/torralba/deeplearning/images256/"
)
places_audio_base_path = "/work/vjsalt22/dataset/places/PlacesAudio_400k_distro/"
places_json_file_train = (
    "/work/vjsalt22/dataset/places/PlacesAudio_400k_distro/metadata/train.json"
)
places_json_file_val = (
    "/work/vjsalt22/dataset/places/PlacesAudio_400k_distro/metadata/val.json"
)


def test_datalaoder():
    tr_set = FlickrDataset(
        dataset_root=flickr_base_path,
        split="train",
        load_image=False,
        modalities=["audio", "image", "text"],
    )

    dv_set = FlickrDataset(
        dataset_root=flickr_base_path,
        split="dev",
        load_image=False,
        modalities=["audio", "image", "text"],
    )

    test_set = FlickrDataset(
        dataset_root=flickr_base_path,
        split="test",
        load_image=False,
        modalities=["audio", "image", "text"],
    )

    tr_loader = DataLoader(
        tr_set,
        batch_size=BATCH_SZ,
        shuffle=True,
        num_workers=WORKERS_NUM,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_general,
    )

    dv_loader = DataLoader(
        dv_set,
        batch_size=BATCH_SZ,
        shuffle=False,
        num_workers=WORKERS_NUM,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_general,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SZ,
        shuffle=False,
        num_workers=WORKERS_NUM,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_general,
    )

    for batch in tr_loader:
        assert "wav" in batch
        assert "wav_len" in batch
        assert "image" in batch
        assert "text" in batch
        break

    for batch in dv_loader:
        assert "wav" in batch
        assert "wav_len" in batch
        assert "image" in batch
        assert "text" in batch
        break

    for batch in test_loader:
        assert "wav" in batch
        assert "wav_len" in batch
        assert "image" in batch
        assert "text" in batch
        break

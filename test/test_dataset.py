import torch
from avssl.data import (
    get_simple_image_transform,
    FlickrImageCaptionDataset,
    PlacesImageCaptionDataset,
    FlickrDataset,
)

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


def get_flickr_dataset(split, image_transform=None):
    dataset = FlickrImageCaptionDataset(
        dataset_root=flickr_base_path, split=split, image_transform=image_transform
    )
    return dataset


def get_flickr_general_dataset(split, image_transform=None):
    dataset = FlickrDataset(
        dataset_root=flickr_base_path,
        split=split,
        image_transform=image_transform,
        modalities=["audio", "image", "text"],
    )
    return dataset


def get_places_dataset(json_file, split, image_transform=None):
    dataset = PlacesImageCaptionDataset(
        json_file,
        places_audio_base_path,
        places_image_base_path,
        split=split,
        image_transform=image_transform,
    )
    return dataset


def try_dataset(dataset):
    audio_feat, image = dataset[0]
    assert audio_feat.ndim == 1
    assert image.shape == (3, 224, 224)
    assert isinstance(image, torch.Tensor)


def try_Flickr(dataset):
    audio_feat, image, id = dataset[0]
    assert audio_feat.ndim == 1
    assert image.shape == (3, 224, 224)
    assert isinstance(id, torch.LongTensor)
    assert isinstance(image, torch.Tensor)


def try_FlickrGeneral(dataset):
    _data = dataset[0]

    assert "id" in _data
    id = _data["id"]
    assert isinstance(id, torch.LongTensor)

    if "wav" in _data:
        audio_feat = _data["wav"]
        assert audio_feat.ndim == 1
    if "image" in _data:
        image = _data["image"]
        assert image.shape == (3, 224, 224)
        assert isinstance(image, torch.Tensor)
    if "text" in _data:
        text = _data["text"]
        assert isinstance(text, torch.LongTensor)


def test_dataset():
    image_transform = get_simple_image_transform(224)

    # Test Flickr8k
    for split, length in [("train", 30000), ("dev", 5000), ("test", 5000)]:
        dataset = get_flickr_dataset(split, image_transform)
        try_Flickr(dataset)
        assert len(dataset) == length
        del dataset

    # Test General Flickr8k
    for split, length in [("train", 30000), ("dev", 5000), ("test", 5000)]:
        dataset = get_flickr_general_dataset(split, image_transform)
        try_FlickrGeneral(dataset)
        assert len(dataset) == length
        del dataset

    # Test Places
    for json_file, split, length in [
        (places_json_file_train, "train", 30000),
        (places_json_file_val, "dev", 5000),
    ]:
        dataset = get_places_dataset(json_file, split, image_transform)
        try_dataset(dataset)
        # assert len(dataset) == length
        del dataset

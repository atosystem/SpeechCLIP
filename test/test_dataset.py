import torch
from avssl.data import (
    get_simple_image_transform,
    FlickrImageCaptionDataset,
    PlacesImageCaptionDataset,
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
    dataset = FlickrImageCaptionDataset(flickr_base_path, split, image_transform)
    return dataset


def get_places_dataset(json_file, split, image_transform=None):
    dataset = PlacesImageCaptionDataset(
        places_audio_base_path,
        places_image_base_path,
        split=split,
        dataset_json_file=json_file,
    )
    return dataset


def try_dataset(dataset):
    audio_feat, image = dataset[0]
    assert audio_feat.ndim == 2
    assert audio_feat.shape[1] == 1
    assert image.shape == (3, 224, 224)
    assert isinstance(image, torch.Tensor)


def test_dataset():
    image_transform = get_simple_image_transform(224)

    # Test Flickr8k
    for split, length in [("train", 30000), ("dev", 5000), ("test", 5000)]:
        dataset = get_flickr_dataset(split, image_transform)
        try_dataset(dataset)
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

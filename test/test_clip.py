import torch
from avssl.module import ClipModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def test_clip():
    model = ClipModel(
        "ViT-B/32",
        device=device,
        image_encoder_trainable=False,
        text_encoder_trainable=False,
    ).to(device)
    model.eval()

    if device == "cuda:0":
        assert model.device.type == "cuda"
    else:
        assert model.device.type == "cpu"

    with torch.no_grad():
        image_tensor = model.prep_image(["./test/samples/cat.jpg"])
        text_tensor = model.prep_text(["a cat", "a dog"])

        assert isinstance(image_tensor, torch.Tensor)
        assert isinstance(text_tensor, torch.Tensor)

        image_rep = model.encode_image(image_tensor)
        text_rep = model.encode_text(text_tensor)
        logits_img, logits_text = model.get_scores(image_tensor, text_tensor)
        probs = logits_img.softmax(-1).cpu()

        assert image_rep.shape == (1, model.out_dim)
        assert text_rep.shape == (2, model.out_dim)
        assert logits_img.shape == (1, 2)
        assert logits_text.shape == (2, 1)
        assert probs.argmax().item() == 0

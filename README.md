# SpeechCLIP


# Prequisite

## Install packages
```bash
pip install -r requirements.txt
```

## Download Pretrained Checkpoints



> Notice that it reuires 2 GPUs for training base models and 4 GPUs for large models

# Usage

## Train

Example: train Parallel SpeechCLIP base:

```bash
bash egs/model_base/parallel/train.sh
```

## Inference

Example: test Parallel SpeechCLIP base:
(Using pretrained checkpoint)
```bash
bash egs/model_base/parallel/test.sh
```

> For more settings, please see the folders in `./egs/`.

# Authors

...

# Contribute
Please run autoformatter before opening PR!
Autoformat `audio-visual-ssl/dev-support/`

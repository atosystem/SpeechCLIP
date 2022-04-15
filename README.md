# audio-visual-ssl
Apply two types of vetor quantization - gumbel softmax and k-means (refer from [fairseq.modules](https://github.com/pytorch/fairseq/tree/main/fairseq/modules))

* If you want to change the type of vector quantization, please modify the config yaml file under `config/speechclip_c/train_flickr.yaml`.
* If you want to run only for **validation** or **testing**, add `--eval` or `--test` flag at `egs/run_speechclip_c.sh`
* If you want to resume your training from specific checkpoint, add `--ckpt/your_checkpoint_path` flag at  `egs/run_speechclip_c.sh`

To run cascaded speechclip, run
```bash
bash egs/run_speechclip_c.sh

```

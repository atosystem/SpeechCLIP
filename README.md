# audio-visual-ssl
Apply two types of vetor quantization - gumbel softmax and k-means (refer from [fairseq.modules](https://github.com/pytorch/fairseq/tree/main/fairseq/modules))

* If you want to change the type of vector quantization, please modify the config yaml file under `config/speechclip_c/` 

To run cascaded speechclip, run
```bash
bash egs/run_speechclip_c.sh

```
# Some statistic on Flickr8k Dataset

In this folder, we provide the vocab usage on Flickr8k (using the CLIP subword embeddings)

You can run `python stat_textCLIP_input.py` to produce the result files. (Remember to change the Flickr8k Dataset Root )

```
├── text_clip_vocab_usage_byID.npy
├── text_clip_vocab_usage_byID.txt
├── text_clip_vocab_usage_byfreq.npy
└── text_clip_vocab_usage_byfreq.txt
```

FileNames startswith `text_clip_vocab_usage_byfreq`: 
*the subwords are sorted in decreasing order of frequency appeared in Flickr8k*

FileNames startswith `text_clip_vocab_usage_byID`: 
*the subwords are sorted in increasing order of subword ID*


These subword usages are used when training Cascaded SpeechCLIP to reduce the vocab size.

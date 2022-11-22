import torch
import avssl.model
import librosa

# load model to GPU
device = torch.device("cuda")

# use Parallel SpeechCLIP trained on Flickr8k for example
model_fp = "slt_ckpts/SpeechCLIP/base/flickr/parallel/epoch_131-step_15443-val_recall_mean_1_36.0100.ckpt"
model = avssl.model.KWClip_GeneralTransformer.load_from_checkpoint(model_fp)
model.to(device)
model.eval()

# load input wav (should be 16kHz)
# wav_fps = ["data/flickr/flickr_audio/wavs/997722733_0cb5439472_3.wav","data/flickr/flickr_audio/wavs/997722733_0cb5439472_4.wav"]
wav_fps = ["/home/atosystem/nas_disk/speechlab/svqa/flickr_dataset/flickr/flickr_audio/wavs/997722733_0cb5439472_3.wav"]

wav_data = []

for _w in wav_fps:
    wav_data.append(
        torch.FloatTensor(librosa.load(_w, sr=16_000)[0]).to(device)
    )

with torch.no_grad():
    # Get Hidden Representations
    output_embs, hidden_states = model.feature_extractor_s3prl(wav=wav_data)
    # output_embs: torch.Tensor shape = (N,max_seq_len,hidden_dim)
    # hidden_states: tuples of torch.Tensor total 14 for base model
    #   for each layer of hidden states: shape = (N,max_seq_len,hid_dim)
    # max_seq_len is the maximum seq_len in the same batch
    
    # Get semantic embedding for speech input
    output = model.encode_speech(wav=wav_data)
    # output = {
    #   "cascaded_audio_feat" : if cascaded branch exists
    #   "parallel_audio_feat" : if parallel branch exists
    #   "vq_results"          : if cascaded branch exists
    #   "keywords"            : if cascaded branch exists
    # }
    



# CoVR-R Submission Notes

Method: zero-shot Reason→Describe→Contrastive-Retrieve→Rerank.
- Frozen Qwen3-VL (no fine-tuning) generates structured after-effect reasoning
  and a post-edit target description.
- Frozen contrastive video-text encoder (SigLIP2 frame-mean) embeds the
  target description (query) and all pool videos (gallery).
- Cosine retrieval within the sample's pool; reference video masked.
- Single Qwen3-VL pointwise rerank pass over top-50.

Generated data disclosure: model-generated target descriptions and reasoning
traces are used as intermediate text. No human labeling of test data. No
training/fine-tuning on val or test. External datasets used only for offline
dev/ablation (not for training).

## Compliance checklist

- [ ] No training on val/test (all models frozen).
- [ ] No human labeling of test data.
- [ ] Generated descriptions disclosed (this file).
- [ ] Test submission set public.
- [ ] Datasets used only for this challenge.

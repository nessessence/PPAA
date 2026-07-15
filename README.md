# Push-Pull Attentional Anchoring for Diffusion Concept Erasure

Official code repository for:

**Push-Pull Attentional Anchoring for Diffusion Concept Erasure**

ECCV 2026

Project page: https://push-pull-concept-eraser.github.io/

## Minimal Training and Generation Example

The built-in case `0` erases `Barack Obama` using the generic concept `person` and
generates images for both `a photo of Barack Obama` and `a photo of Will Smith`.

### Training

This trains one configuration with `m_push=0.20`, `m_pull=0.40`, and token selection
`C` while preserving all semantic tokens:

```bash
python script.py \
  --device cuda:0 \
  --case-index 0 \
  --m-push-values 0.20 \
  --m-pull 0.40 \
  --token-selections C \
  --pull-token-selections "" \
  --semantic-preserve-tokens all \
  --pull-weights 1.0 \
  --skip-generate
```

The checkpoint is written to:

```text
data_root/logs/ppaa/barack-obama_to_person_push0.20_pull0.40_relative_C_seed0/step1000.safetensors
```

### Inference (Generation)

After training, generate 50 images for both prompts using the completed checkpoint:

```bash
python script.py \
  --device cuda:0 \
  --case-index 0 \
  --m-push-values 0.20 \
  --m-pull 0.40 \
  --token-selections C \
  --pull-token-selections "" \
  --semantic-preserve-tokens all \
  --pull-weights 1.0 \
  --n-generation 50 \
  --cfg-scale 7.50 \
  --num-inference-steps 50 \
  --skip-train
```

Images are saved under:

```text
data_root/generated/ppaa/barack-obama_to_person_push0.20_pull0.40_relative_C_seed0/step1000/a photo of Barack Obama/7.50/0000.jpg
data_root/generated/ppaa/barack-obama_to_person_push0.20_pull0.40_relative_C_seed0/step1000/a photo of Will Smith/7.50/0000.jpg
```

## Citation

Citation information will be added when the paper metadata is public.

```bibtex
@inproceedings{chatthee2026pushpull,
  title     = {Push-Pull Attentional Anchoring for Diffusion Concept Erasure},
  author    = {Chatthee, Nattanat and Sompong, Tagon and Chuangsuwanich, Ekapol and Suwajanakorn, Supasorn},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## License

License information will be added with the public release.

## Contact

Please follow the project page for updates:

https://push-pull-concept-eraser.github.io/

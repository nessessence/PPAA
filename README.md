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
python ppaa/train_ppaa_erase.py \
  --erase_concept "Barack Obama" \
  --generic_concept "person" \
  --device cuda:0 \
  --token_selection C \
  --pull_token_selection "" \
  --m_push 0.20 \
  --steps 1000 \
  --precision bf16 \
  --save_dir data_root/logs/ppaa \
  --report_to none
```

The checkpoint is written to:

```text
data_root/logs/ppaa/barack-obama_to_person_push0.20_pull0.40_relative_C_seed0/step1000.safetensors
```

### Inference (Generation)

After training, use the completed weight file directly with the standalone inference script:

```bash
python inference_ppaa.py \
  --load_unet_weight_path data_root/logs/ppaa/barack-obama_to_person_push0.20_pull0.40_relative_C_seed0/step1000.safetensors \
  --instance_prompt "a photo of Barack Obama;a photo of Will Smith" \
  --gen_image_path data_root/generated/ppaa/barack-obama_to_person_push0.20_pull0.40_relative_C_seed0/step1000 \
  --num_validation_images 50 \
  --cfg_scale 7.50 \
  --num_inference_steps 50 \
  --device cuda:0
```

Images are saved under:

```text
data_root/generated/ppaa/barack-obama_to_person_push0.20_pull0.40_relative_C_seed0/step1000/a photo of Barack Obama/7.50/0000.png
data_root/generated/ppaa/barack-obama_to_person_push0.20_pull0.40_relative_C_seed0/step1000/a photo of Will Smith/7.50/0000.png
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

# configs/

Every model in this project is now `ModelConfig` (architecture) +
`TrainConfig` (hyperparameters), both plain JSON. Pick a preset, override
one field at a time, done -- see the top of `src/train/cli.py` for examples.

## configs/model/*.json

| file | backbone | num_classes | class_names | matches (original script) |
|---|---|---|---|---|
| `resnet_100k.json` | resnet50 | 7 | 7-class | `train_resnet_100K.py` |
| `resnet_1m.json` | resnet50 | 7 | 7-class | `train_resnet_1M.py` |
| `resnet_6m.json` | resnet50 | 7 | 7-class | `train_resnet_6M.py` |
| `densenet_1m.json` | densenet121 | 7 | 7-class | `train_densenet_1M.py` |
| `vit_1m.json` | vit_b16 | 7 | 7-class | `train_vit_1M.py` |
| `gate_1m.json` | densenet121 | 2 | cancer/non_cancer | `train_gate_model.py` |
| `specialist_1m.json` | densenet121 | 2 | bkl/mel | `train_specialist_mel_bkl.py` |
| `custom_cnn.json` | custom_cnn | 7 | 7-class | `train_model.py` (see caveat below) |

`resnet_100k` / `resnet_1m` / `resnet_6m` are identical architectures --
they only differ in which dataset (`configs/train/*.json`'s `meta_csv`)
they're trained on.

**custom_cnn.json caveat**: `DermascanCNN` expects 28x28 input (see
`src/models/custom_cnn.py`), unlike every other backbone here (224x224).
There's no matching `configs/train/custom_cnn.json` for this reason --
`src/train/Trainer` hardcodes 224x224 transforms. This config exists for
architecture reference; the original script (confirmed unused downstream)
lives at `legacy/train_model_custom_cnn.py`.

## configs/train/*.json

| file | matches (original script) | split | loss | LR schedule |
|---|---|---|---|---|
| `resnet_100k.json` | `train_resnet_100K.py` | full 7-class | focal, class-weighted | steplr |
| `resnet_1m.json` | `train_resnet_1M.py` | full 7-class | focal, class-weighted | plateau |
| `resnet_6m.json` | `train_resnet_6M.py` | full 7-class | plain CE, unweighted | steplr |
| `densenet_1m.json` | `train_densenet_1M.py` | full 7-class | focal, class-weighted | plateau |
| `vit_1m.json` | `train_vit_1M.py` | full 7-class | focal, class-weighted | plateau |
| `gate_1m.json` | `train_gate_model.py` | binary gate | CE, cancer-weighted 3x | none |
| `specialist_1m.json` | `train_specialist_mel_bkl.py` | mel/bkl filtered | CE, mel-weighted 2x | none |

**resnet_6m fidelity note**: the original `train_resnet_6M.py` selected/
checkpointed the "best" model by lowest VAL LOSS. Every preset in this
project (including this one) selects by recall instead (`priority_classes`
in the JSON, empty = macro recall) -- `resnet_6m.json` reproduces the same
architecture, data, and optimizer settings, but not bit-for-bit identical
checkpoint selection. This only affects the earliest/experimental variant;
the production-relevant presets (`resnet_1m`, `gate_1m`, `specialist_1m`)
match their originals exactly.

**No preset for `train_resnet.py` (the un-suffixed base script)**: it used
a 2-way (train/val only) split with no held-out test set, superseded in
every later script by the 3-way split every preset above uses. Kept as-is
at `legacy/train_resnet_v1_base.py`.

**No presets for the gate/specialist v2/v3 retraining experiments**
mentioned in project notes (row-capped augmentation, `--extra_augment`):
those used a modified data-augmentation step that isn't part of the
`train_gate_model.py` / `train_specialist_mel_bkl.py` code actually in this
repo. `src/data/augment.py` would need a row-capping option added before a
matching preset could reproduce them.

## Adjusting one thing at a time

```bash
# same as gate_1m.json, but cancer weight 4.0 instead of 3.0
python -m src.train.cli --preset gate_1m --set 'extra_class_weight={"cancer": 4.0}'

# same as specialist_1m.json, but resnet50 backbone instead of densenet121
python -m src.train.cli --preset specialist_1m --model-set backbone=resnet50

# same as resnet_1m.json, but plain CE instead of focal loss
python -m src.train.cli --preset resnet_1m --set loss_type=ce
```

Or just copy a preset file, edit the one field you care about, and pass
`--preset your_new_name` (save it as `configs/train/your_new_name.json`).

# Yukarin: train the first stage model for voice conversion
This repository is refactoring the training code for the first stage model of
[Bcome Yukarin: Convert your voice to favorite voice](https://github.com/Hiroshiba/become-yukarin).

[Japanese README](./README_jp.md)

## Supported environment
* Linux OS
* Python 3.6

## Preparation
### Installation required libraries
```bash
pip install -r requirements.txt
```

### How to run code (preliminary knowledge)
To run a Python script in this repository, you should set the environment variable `PYTHONPATH` to find the `yukarin` library.
For example, you can run `scripts/foo.py` with the following command:

```bash
PYTHONPATH=`pwd` python scripts/foo.py
```

## Create dataset
### Prepare voice data
Put input/target voice data in two directories (ex. `input_wav` and `target_wav`).
These data should be same file names.

### Create acoustic feature
Create input/target acoustic feature files from each voice data.

```bash
python scripts/extract_acoustic_feature.py \
    -i './input_wav/*' \
    -o './input_feature/'

python scripts/extract_acoustic_feature.py \
    -i './target_wav/*' \
    -o './target_feature/'
```

### Align data
Align input and target acoustic features in time direction.
In the following example, create the alignment data between `input_feature` and `target_feature` into `aligned_indexes`.

```bash
python scripts/extract_align_indexes.py \
    -i1 './input_feature/*.npy' \
    -i2 './target_feature/*.npy' \
    -o './aligned_indexes/'
```

### Calculate frequency statistics
Calculate frequency statistics for input and target voice data.
Statistics are needed for voice pitch conversion.

```bash
python scripts/extract_f0_statistics.py \
    -i './input_feature/*.npy' \
    -o './input_statistics.npy'

python scripts/extract_f0_statistics.py \
    -i './target_feature/*.npy' \
    -o './target_statistics.npy'
```

## Train
### Create the training config file `config.json`
Modify `input_glob`, `target_glob` and `indexes_glob` in `sample_config.json`, then can train.

### Train

```bash
python train.py \
    sample_config.json \
    ./model_stage1/
```

## Test
Put the test input voice data in a directory (ex. `test_wav`), and run `voice_change.py`.

```bash
python scripts/voice_change.py \
    --model_dir './model_stage1' \
    --config_path './model_stage1/config.json' \
    --input_statistics 'input_statistics.npy' \
    --target_statistics 'target_statistics.npy' \
    --output_sampling_rate 24000 \
    --disable_dataset_test \
    --test_wave_dir './test_wav/' \
    --output_dir './output/'
```

## Advanced: with second stage model
[Become Yukarin](https://github.com/Hiroshiba/become-yukarin)'s [Second Stage Model](https://github.com/Hiroshiba/become-yukarin#second-stage-model)
can improve the quality of the converted voice.

### Train
Train the second stage model referring to [Second Stage Model](https://github.com/Hiroshiba/become-yukarin#second-stage-model) in [Become Yukarin](https://github.com/Hiroshiba/become-yukarin).

### Test
Put the test input voice data in a directory (ex. `test_wav`), and run `voice_change_with_second_stage.py`.

```bash
python scripts/voice_change_with_second_stage.py \
    --voice_changer_model_dir './model_stage1' \
    --voice_changer_config './model_stage1/config.json' \
    --super_resolution_model './model_stage2/' \
    --super_resolution_config './model_stage2/config.json' \
    --input_statistics 'input_statistics.npy' \
    --target_statistics 'target_statistics.npy' \
    --out_sampling_rate 24000 \
    --disable_dataset_test \
    --dataset_target_wave_dir '' \
    --test_wave_dir './test_wav' \
    --output_dir './output/'
```

## License
[MIT License](./LICENSE)

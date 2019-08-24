# Yukarin: ディープラーニング声質変換の第１段階の学習コード
このリポジトリは、[Become Yukarin: 誰でも好きなキャラの声になれるリポジトリ](https://github.com/Hiroshiba/become-yukarin)の、
第１段階の学習コードを拡張したものです。

[English README](./README.md)

## 推奨環境
* Linux OS
* Python 3.6

## 準備
### 必要なライブラリのインストール
```bash
pip install -r requirements.txt
```

### コードの実行方法（予備知識）
このリポジトリのコードを実行するには、`yukarin`ライブラリをパス（PYTHONPATH）に通す必要があります。
例えば`scripts/foo.py`を実行するには、以下のように書いて、パスを通します。

```bash
PYTHONPATH=`pwd` python scripts/foo.py
```

## データ作成
### 音声データを用意する
入力音声データと、目標音声データを大量に用意し、別々のディレクトリ（例：`input_wav`と`target_wav`）に配置します。
ファイル名は同じである必要があります。

### 音響特徴量を切り出す
入力と目標の音声データそれぞれの音響特徴量ファイルを出力します。

```bash
python scripts/extract_acoustic_feature.py \
    -i './input_wav/*' \
    -o './input_feature/'

python scripts/extract_acoustic_feature.py \
    -i './target_wav/*' \
    -o './target_feature/'
```

### データを揃える（アライメントする）
入力と目標の音響特徴量を時間方向に揃えます。
次の例では、`input_feature`と`target_feature`のアライメントデータを`aligned_indexes`に出力します。

```bash
python scripts/extract_align_indexes.py \
    -i1 './input_feature/*.npy' \
    -i2 './target_feature/*.npy' \
    -o './aligned_indexes/'
```

### 周波数の統計量を求める
声の高さの変換に必要な周波数の統計量を、入力・目標データそれぞれに対して求めます。

```bash
python scripts/extract_f0_statistics.py \
    -i './input_feature/*.npy' \
    -o './input_statistics.npy'

python scripts/extract_f0_statistics.py \
    -i './target_feature/*.npy' \
    -o './target_statistics.npy'
```

## 学習
### 学習用の設定ファイル`config.json`を作る
`sample_config.json`の`input_glob`、`target_glob`、`indexes_glob`を変更すればとりあえず動きます。

### 学習する

```bash
python train.py \
    sample_config.json \
    ./model_stage1/
```

## テスト
テスト用の入力音声データをディレクトリ（例：`test_wav`）に配置し、`voice_change.py`を実行します。

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

## 発展：第２段階モデル
[Become Yukarin](https://github.com/Hiroshiba/become-yukarin)の[第２段階モデル](https://github.com/Hiroshiba/become-yukarin#%E7%AC%AC%EF%BC%92%E6%AE%B5%E9%9A%8E%E3%81%AE%E5%AD%A6%E7%BF%92)を使えば、
変換結果の音声の品質を上げることができます。

### 学習する
[Become Yukarin](https://github.com/Hiroshiba/become-yukarin)の[第２段階の学習](https://github.com/Hiroshiba/become-yukarin#%E7%AC%AC%EF%BC%92%E6%AE%B5%E9%9A%8E%E3%81%AE%E5%AD%A6%E7%BF%92)を参考に、
第２段階モデルを学習します。

### テスト
テスト用の入力音声データをディレクトリ（例：`test_wav`）に配置し、`voice_change_with_second_stage.py`を実行します。

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

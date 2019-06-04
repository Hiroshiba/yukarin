# yukarin
ディープラーニング声質変換の第１段階モデルの学習コード。

## 推奨環境
* Unix系のPython3.6.3

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
1. 音声データを用意する
入力音声データと、目標音声データを大量に用意し、別々のディレクトリ（例：`input_wav`と`target_wav`）に配置します。
ファイル名は揃えるか、もしくは[glob](https://docs.python.org/ja/3/library/glob.html)の順序が同じになるようにします。

2. 音響特徴量を切り出す
入力と目標の音声データそれぞれの音響特徴量ファイルを出力します。

```bash
python scripts/extract_acoustic_feature.py \
    -i './input_wav/*' \
    -o './input_feature/'
```

3. データを揃える（アライメントする）
入力と目標の音声データを時間方向に揃えます。
次の例では、`input_dir`と`target_dir`のアライメントデータを`aligned_indexes`に出力します。

```bash
python scripts/extract_align_indexes.py \
    -i1 './input_feature/*' \
    -i2 './target_feature/*' \
    -o './aligned_indexes/'
```

4. 周波数の統計量を求める
声の高さの変換に必要な、周波数の統計量を入力・目標音声データそれぞれに対して求めます。

```bash
python scripts/extract_acoustic_feature.py \
    -i './input_feature/*' \
    -o './input_statistics.npy'
```

## 学習
1. 学習用の設定ファイル`config.json`を作る
`sample_config.json`の`input_glob`、`target_glob`、`indexes_glob`を変更すればとりあえず動きます。

2. 学習する

```bash
python scripts/train,py \
    config.json \
    ./model_stage1/
```

3. 第２段階モデルを学習する
[become-yukarin](https://github.com/Hiroshiba/become-yukarin)の[第２段階の学習](https://github.com/Hiroshiba/become-yukarin#%E7%AC%AC%EF%BC%92%E6%AE%B5%E9%9A%8E%E3%81%AE%E5%AD%A6%E7%BF%92)を参考に、
第２段階モデルを学習します。

## テスト
テスト用の入力音声データをディレクトリ（例：`test_wav`）に配置し、`voice_change.py`を実行します。

```bash
python scripts/voice_change.py \
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

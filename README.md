# plamo3_benchmark

`pfnet/plamo-3-nict-8b-base` を OpenVINO IR に変換し、`plamo3-ov`
コマンドでテキスト生成と簡易チャットを実行するための CLI です。

CPU / GPU 向けは OpenVINO GenAI 互換の stateful KV-cache IR を作ります。
NPU 向けは static shape / int32 入力の stateful KV-cache IR を作ります。

## Features

- Hugging Face の PLaMo 3 NICT 8B Base を OpenVINO IR に変換
- CPU / GPU 向け stateful KV-cache 変換
- NPU 向け static shape / int32 入力の stateful KV-cache 変換
- `fp32` / `fp16` / `int8` / `int4` の weight format
- NNCF による INT8 / INT4 weight compression
- PLaMo 3 tokenizer の保存と OpenVINO tokenizer IR 変換
- 位置引数、ファイル、標準入力からのプロンプト入力
- 生成と対話チャット
- `model_load`、first token time、total inference time、生成トークン数、tokens/sec の表示

## Requirements

- Python 3.10 以上
- `uv`
- Hugging Face アカウント
- `pfnet/plamo-3-nict-8b-base` へのアクセス権

PLaMo 3 NICT 8B Base は gated repo です。先に Hugging Face のモデルページで
アクセス申請またはライセンス同意を済ませてください。

https://huggingface.co/pfnet/plamo-3-nict-8b-base

このモデルは Hugging Face の custom code を使います。CLI は既定で
`trust_remote_code=True` を指定します。

## Setup

```powershell
uv sync
```

Hugging Face 認証が必要な場合は、どちらかでログインします。

```powershell
uv run huggingface-cli login
```

```powershell
$env:HF_TOKEN="<your-token>"
```

モデルファイルのダウンロードには Hugging Face Xet Storage を使うため、依存に
`hf-xet` を含めています。

## Quick Start

CPU / GPU 向けに変換します。

```powershell
uv run plamo3-ov convert --output-dir ov-plamo3 --weight-format fp16
```

生成します。

```powershell
uv run plamo3-ov generate "これからの人工知能技術は" --model ov-plamo3 --device CPU --max-new-tokens 128
```

チャットします。

```powershell
uv run plamo3-ov chat --model ov-plamo3 --device CPU --max-new-tokens 128
```

NPU 向けに変換する場合は `--target-device NPU` を指定します。

```powershell
uv run plamo3-ov convert --output-dir ov-plamo3-npu-int4 --target-device NPU --weight-format int4 --max-seq-len 512 --force
uv run plamo3-ov generate "これからの人工知能技術は" --model ov-plamo3-npu-int4 --device NPU --max-new-tokens 128
```

## Convert

```powershell
uv run plamo3-ov convert --output-dir ov-plamo3 --weight-format fp16
```

主なオプション:

- `--model`: 変換元モデル。既定は `pfnet/plamo-3-nict-8b-base`
- `--output-dir`: 変換後の OpenVINO model directory
- `--weight-format`: `fp32`、`fp16`、`int8`、`int4`
- `--target-device`: 変換ターゲットの目安。既定は `CPU`
- `--max-seq-len`: NPU 変換時の固定KV-cache長。NPUで省略した場合は512
- `--force`: 既存の `openvino_model.xml` があっても再変換
- `--local-files-only`: Hugging Face にアクセスせず、ローカル cache またはローカル model directory だけを使う
- `--trust-remote-code` / `--no-trust-remote-code`: custom code の許可

### CPU / GPU / AUTO

`--target-device NPU` を指定しない場合は、stateful KV-cache IR を作ります。
この IR は `input_ids`、`attention_mask`、`position_ids`、`beam_idx` を持ち、
KV cache は OpenVINO の `ReadValue` / `Assign` state として内部化されます。

```powershell
uv run plamo3-ov convert --output-dir ov-plamo3-fp16 --weight-format fp16
uv run plamo3-ov convert --output-dir ov-plamo3-int8 --weight-format int8
uv run plamo3-ov convert --output-dir ov-plamo3-int4 --weight-format int4
```

通常の INT8 / INT4 compression は NNCF の asymmetric mode を使います。

- `int8`: `INT8_ASYM`
- `int4`: `INT4_ASYM`

### NPU

`--target-device NPU` を指定した場合は、static shape / int32 入力の stateful KV-cache IR を作ります。
KV cache は OpenVINO の `ReadValue` / `Assign` state として内部化されます。

```powershell
uv run plamo3-ov convert --output-dir ov-plamo3-npu-int4 --target-device NPU --weight-format int4 --max-seq-len 512 --force
```

NPU 変換では次のように保存します。

- `input_ids`: `int32`、shape `[1, 1]`
- `attention_mask`: `int32`、shape `[1, max_seq_len]`
- `position_ids`: `int32`、shape `[1, 1]`
- `beam_idx`: `int32`、shape `[1]`
- KV cache: OpenVINO state、shape `[1, kv_heads, max_seq_len, head_dim]`
- `fp32` 指定時も保存形式は `fp16`
- NPU では常に `INT4_SYM`
- `group_size=-1` の channel-wise compression
- `ratio=1.0`
- embedding / last layer などの fallback は NNCF のNPU互換な既定に任せます

推論時はプロンプトも生成トークンも1トークンずつ流し、固定長KV stateを更新します。

### Existing Output Directory

既存ディレクトリに `openvino_model.xml` がある場合、`convert` はモデル本体を再利用し、
tokenizer と config を補完します。

再利用時には現在のオプションと既存 IR の形式を確認します。weight format、NPU 用 shape、
stateful layout が合わない場合は `--force` を付けて再変換してください。

```powershell
uv run plamo3-ov convert --output-dir ov-plamo3 --weight-format int4 --force
```

モデルがすでに Hugging Face cache にある環境でネットワーク確認を避けたい場合:

```powershell
uv run plamo3-ov convert --output-dir ov-plamo3 --weight-format fp16 --local-files-only
```

## Generate

```powershell
uv run plamo3-ov generate "これからの人工知能技術は" --model ov-plamo3
```

プロンプトは位置引数、ファイル、標準入力のいずれかで渡せます。同時指定はできません。

```powershell
uv run plamo3-ov generate --prompt-file prompt.txt --model ov-plamo3
Get-Content prompt.txt | uv run plamo3-ov generate --stdin --model ov-plamo3
```

主なオプション:

- `--model`: OpenVINO model directory。既定は `ov-plamo3`
- `--device`: OpenVINO device string。既定は `CPU`
- `--max-new-tokens`: 生成トークン数。既定は 128
- `--temperature`: direct fallback 用 sampling temperature。既定は 0.8
- `--top-p`: direct fallback 用 nucleus sampling。既定は 0.95
- `--top-k`: direct fallback 用 top-k sampling。既定は 50
- `--stream` / `--no-stream`: direct fallback の逐次表示
- `--model-cache` / `--no-model-cache`: OpenVINO compiled model cache の有効化。既定は有効
- `--model-cache-dir`: compiled model cache directory。既定は `<model>/.openvino_cache/<device>`
- `--apply-chat-template` / `--no-apply-chat-template`: `generate` で chat template を適用するか

OpenVINO compiled model cache は既定で有効です。NPU では初回実行時に compile した blob を
`<model>/.openvino_cache/<device>` に保存し、2回目以降の `model_load` を短縮します。
cache を分けたい場合は `--model-cache-dir` を指定し、無効化したい場合は `--no-model-cache` を指定します。

OpenVINO GenAI の `LLMPipeline` が使える IR では GenAI 経路を優先します。現在の実装では
GenAI 経路は greedy generation です。`temperature`、`top-p`、`top-k`、`stream` は
direct fallback で効きます。

## Chat

```powershell
uv run plamo3-ov chat --model ov-plamo3 --device CPU --max-new-tokens 128
```

チャットはモデルを一度ロードし、同じセッション内で使い回します。

`pfnet/plamo-3-nict-8b-base` のような Base モデルでは、CLI 側で
`System:` / `User:` / `Assistant:` 形式のプロンプトを組み立てます。
次の発話へ流れ込みにくいよう、会話区切りの stop string も指定します。

チャット中のコマンド:

- `/exit` または `/quit`: 終了
- `/reset`: 会話履歴をクリア

システムプロンプトを指定する場合:

```powershell
uv run plamo3-ov chat --model ov-plamo3 --system "日本語で簡潔に答えてください。"
```

ターン数を制限する場合:

```powershell
uv run plamo3-ov chat --model ov-plamo3 --max-turns 3
```

## Inference Path

推論時はまず OpenVINO GenAI 互換性を確認します。

GenAI 経路を使う条件:

- `openvino_tokenizer.xml` と `openvino_detokenizer.xml` がある
- `openvino_model.xml` が `beam_idx` input を持つ
- logits の sequence dimension が dynamic

条件を満たす場合は `openvino_genai.LLMPipeline` を使います。条件を満たさない場合は
OpenVINO Core と Hugging Face tokenizer を使う direct fallback に切り替えます。

direct fallback は次の IR を扱います。

- CPU / GPU 向け stateful IR: OpenVINO InferRequest の state を reset しながら逐次生成
- NPU 向け static stateful IR: int32 入力で OpenVINO InferRequest の固定長KV stateを reset しながら逐次生成

## Conversion Details

このリポジトリの変換経路は `optimum-cli` を使いません。Hugging Face model を読み込み、
`openvino.convert_model` で OpenVINO IR を作ります。

変換まわりのコードは役割ごとに分けています。

- `model_convert.py`: CLI から呼ばれる変換フロー全体
- `model_download.py`: Hugging Face access check、tokenizer/model loading
- `model_export.py`: PLaMo 3 固有の PyTorch wrapper と OpenVINO export
- `model_artifacts.py`: model/tokenizer/config/metadata の保存
- `quantization.py`: target 判定と NNCF weight compression

CPU / GPU 向け stateful 変換では `torch.export` で次の wrapper を変換します。

- PLaMo 3 の GQA attention を変換時だけ K/V head expansion できるよう patch
- `past.*` / `present.*` を flatten した KV-cache input/output として作成
- `apply_make_stateful_transformation` で KV cache を `ReadValue` / `Assign` state に変換
- sliding window attention は cache を切り詰めず、位置ベース mask で表現

NPU 向け変換では `StaticStatefulKV` wrapper を使い、`input_ids`、`attention_mask`、
`position_ids` は int32、`beam_idx` は int32 の固定長KV-cache IRを作ります。

## Tokenizer

PLaMo 3 の custom `Plamo3Tokenizer` は `openvino_tokenizers` で直接変換できないため、
内部の Unigram 語彙から Hugging Face fast tokenizer を再構築して OpenVINO tokenizer IR
に変換します。

既知の差分:

- `break_around_repeated_chars_threshold` の分割 heuristic は再現していません
- detokenizer は、デコード結果が 2 個以上の空白で始まる場合に先頭の空白を 1 つ落とします

Hugging Face tokenizer 自体も model directory に保存するため、OpenVINO tokenizer IR が使えない場合でも
direct fallback は Hugging Face tokenizer で動作できます。

## Output Files

変換後の directory には主に次のファイルが入ります。

- `openvino_model.xml`
- `openvino_model.bin`
- `openvino_tokenizer.xml`
- `openvino_tokenizer.bin`
- `openvino_detokenizer.xml`
- `openvino_detokenizer.bin`
- `config.json`
- `generation_config.json`
- tokenizer files
- `plamo3_ov_conversion.json`

`plamo3_ov_conversion.json` には weight format、target device、compression mode、
static shape / input dtype、KV-cache layout などの変換 metadata を保存します。

## Metrics

各生成後、stderr にメトリクスを表示します。

```text
[metrics] model_load: 12.326s | time_to_first_token: 0.123s | total_inference: 4.567s | output_tokens: 128 | tokens/sec: 28.02
```

- `model_load`: tokenizer / IR load / OpenVINO compile を含む generator 初期化時間。compiled model cache が効くと短くなります
- `time_to_first_token`: 生成開始から最初の token までの時間
- `total_inference`: `generate()` 呼び出し内の推論全体時間。model load は含みません
- `output_tokens`: 生成 token 数
- `tokens/sec`: first token 以降の生成速度

## Troubleshooting

CPU / GPU 向けの古い IR を使っていて `beam_idx` がないと言われる場合:

```powershell
uv run plamo3-ov convert --output-dir ov-plamo3 --weight-format fp16 --force
```

weight format を変える場合:

```powershell
uv run plamo3-ov convert --output-dir ov-plamo3 --weight-format int4 --force
```

NPU 用の sequence length を変える場合:

```powershell
uv run plamo3-ov convert --output-dir ov-plamo3-npu-int4 --target-device NPU --weight-format int4 --max-seq-len 1024 --force
```

Hugging Face への接続確認を避けたい場合:

```powershell
uv run plamo3-ov convert --output-dir ov-plamo3 --weight-format fp16 --local-files-only
```

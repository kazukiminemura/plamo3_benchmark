# plamo3_benchmark

`pfnet/plamo-3-nict-8b-base` を OpenVINO で動かすための
Python CLI です。Hugging Face 上の PLaMo 3 NICT 8B Base を OpenVINO IR に変換し、
テキスト生成や簡易チャットを実行できます。

## できること

- Hugging Face の PLaMo 3 NICT 8B Base を OpenVINO IR へ変換
- KV cache 付き IR での逐次推論
- `fp32` / `fp16` / `int8` / `int4` の weight format を指定
- CPU / GPU / NPU / AUTO などの OpenVINO device string で推論
- 位置引数、ファイル、標準入力からプロンプトを投入
- CLI 上での対話チャット
- 生成ごとの FTTT、生成トークン数、合計時間、tokens/sec の表示

## 前提

- Python 3.10 以上
- `uv`
- Hugging Face アカウント
- `pfnet/plamo-3-nict-8b-base` へのアクセス権

PLaMo 3 NICT 8B Base は gated repo です。先にモデルページでアクセス申請または
ライセンス同意を済ませてください。

https://huggingface.co/pfnet/plamo-3-nict-8b-base

このモデルは Hugging Face の custom code を使うため、CLI は既定で
`trust_remote_code=True` を指定します。利用前にモデルライセンスとコードの内容を
確認してください。

## アーキテクチャ

```mermaid
flowchart TD
    User["User / PowerShell"] --> CLI["plamo3-ov CLI<br/>cli.py"]

    CLI --> ConvertCmd["convert command"]
    CLI --> GenerateCmd["generate command"]
    CLI --> ChatCmd["chat command"]

    ConvertCmd --> ModelConvert["model_convert.py"]
    ModelConvert --> HFModel["Hugging Face<br/>pfnet/plamo-3-nict-8b-base"]
    ModelConvert --> HFTokenizer["Hugging Face tokenizer<br/>Plamo3Tokenizer"]
    ModelConvert --> OVConvert["openvino.convert_model"]
    ModelConvert --> GQAPatch["GQA SDPA patch<br/>K/V head expansion"]
    ModelConvert --> NNCF["NNCF weight compression<br/>fp16 / int8 / int4"]
    ModelConvert --> Artifact["OpenVINO model directory<br/>openvino_model.xml/bin<br/>tokenizer/config files"]

    HFModel --> OVConvert
    GQAPatch --> OVConvert
    OVConvert --> NNCF
    NNCF --> Artifact
    HFTokenizer --> Artifact

    GenerateCmd --> Inference["inference.py"]
    ChatCmd --> Inference
    Inference --> CoreFallback["OpenVINO Core<br/>HF tokenizer"]
    Artifact --> CoreFallback
    CoreFallback --> Metrics
    Metrics --> User
```

## セットアップ

```powershell
uv sync
```

モデルファイルのダウンロードには Hugging Face Xet Storage を使うため、依存に
`hf-xet` を含めています。

Hugging Face 認証が必要な場合は、どちらかの方法でログインします。

```powershell
uv run huggingface-cli login
```

```powershell
$env:HF_TOKEN="<your-token>"
```

## クイックスタート

まず OpenVINO 形式へ変換します。

```powershell
uv run plamo3-ov convert --output-dir ov-plamo3 --weight-format fp16
```

NPU で動かす場合は、変換時に NPU 向けの固定 shape IR と int32 token 入力に寄せます。

```powershell
uv run plamo3-ov convert --output-dir ov-plamo3-npu-int4 --target-device NPU --weight-format int4 --max-seq-len 512 --force
uv run plamo3-ov generate "これからの人工知能技術は" --model ov-plamo3-npu-int4 --device NPU --max-new-tokens 128
```

変換したモデルで生成します。

```powershell
uv run plamo3-ov generate "これからの人工知能技術は" --model ov-plamo3 --max-new-tokens 128
```

チャットを始める場合:

```powershell
uv run plamo3-ov chat --model ov-plamo3 --device CPU --max-new-tokens 128
```

## OpenVINO 形式への変換

この CLI は `optimum-cli` を使わず、`openvino.convert_model` で
OpenVINO 用のモデルディレクトリを作ります。推論は OpenVINO Core と
Hugging Face tokenizer を組み合わせて実行します。

```powershell
uv run plamo3-ov convert --output-dir ov-plamo3 --weight-format fp16
```

主なオプション:

- `--model`: 変換元モデル。既定は `pfnet/plamo-3-nict-8b-base`
- `--output-dir`: 変換後のモデルディレクトリ
- `--weight-format`: `fp32`、`fp16`、`int8`、`int4`
- `--max-seq-len`: 変換時に使う固定シーケンス長
- `--target-device`: 変換先デバイスの目安。`NPU` を指定すると固定 shape と int32 入力で変換
- `--kv-cache` / `--no-kv-cache`: KV cache 付き IR を作るかどうか。既定は無効。CPU で incremental inference を使いたい場合だけ `--kv-cache` を指定してください
- `--force`: 既存の `openvino_model.xml` があっても再変換
- `--local-files-only`: Hugging Face にアクセスせず、ローカル cache またはローカルモデルディレクトリだけを使う
- `--trust-remote-code` / `--no-trust-remote-code`: custom code の許可

`int8` は通常 NNCF の `INT8_ASYM`、`int4` は通常 NNCF の `INT4_ASYM` weight compression を
OpenVINO IR に適用します。`--target-device NPU` の場合は ASYM を使わず、`int8` は
`INT8_SYM`、`int4` は `INT4_SYM`、`ratio=1.0`、`group_size=-1` で圧縮します。

```powershell
uv run plamo3-ov convert --output-dir ov-plamo3-int8 --weight-format int8 --max-seq-len 512
uv run plamo3-ov convert --output-dir ov-plamo3-int4 --weight-format int4 --max-seq-len 512
```

`--no-kv-cache` で `--max-seq-len` を省略した場合は 512 を使います。OpenVINO GPU では
KV-cache graph が token を反復することがあるため、GPU で使うモデルは既定の no-KV 変換を使ってください。

`--target-device NPU` を指定した場合、NPU plugin が受けやすいように変換後の
`input_ids` / `attention_mask` は int32、shape は `[1, --max-seq-len]` に固定されます。
`--max-seq-len` を省略した場合は 512 を使います。`--weight-format fp32` を指定しても
NPU 向けには FP16 保存へ切り替えます。

NPU 用 int4 を作る場合:

```powershell
uv run plamo3-ov convert --output-dir ov-plamo3-npu-int4 --target-device NPU --weight-format int4 --max-seq-len 512 --force
```

Hugging Face への HEAD request が接続リセットになる環境で、モデルがすでに cache 済みの場合は
`--local-files-only` を付けるとネットワーク確認を避けられます。

```powershell
uv run plamo3-ov convert --output-dir ov-plamo3-npu-int4 --target-device NPU --weight-format int4 --max-seq-len 512 --force --local-files-only
```

既存ディレクトリに `openvino_model.xml` がある場合、`convert` はモデル本体を再利用し、
tokenizer と config を補完します。既存の別形式 IR を `int8` または `int4` に置き換える場合は、
Windows のファイルロックを避けるため `--force` を付けるか、別の出力ディレクトリを
使ってください。

```powershell
uv run plamo3-ov convert --output-dir ov-plamo3 --weight-format int8 --max-seq-len 512 --force
uv run plamo3-ov convert --output-dir ov-plamo3 --weight-format int4 --max-seq-len 512 --force
```

## テキスト生成

```powershell
uv run plamo3-ov generate "これからの人工知能技術は" --model ov-plamo3
```

デバイスは `--device` で指定できます。OpenVINO の device string をそのまま渡します。

```powershell
uv run plamo3-ov generate "これからの人工知能技術は" --model ov-plamo3 --device CPU
uv run plamo3-ov generate "これからの人工知能技術は" --model ov-plamo3-int8 --device GPU
uv run plamo3-ov generate "これからの人工知能技術は" --model ov-plamo3-int4 --device AUTO:GPU,CPU
uv run plamo3-ov generate "これからの人工知能技術は" --model ov-plamo3-npu-int4 --device NPU
```

ファイルまたは標準入力からプロンプトを渡すこともできます。

```powershell
uv run plamo3-ov generate --prompt-file prompt.txt --model ov-plamo3
Get-Content prompt.txt | uv run plamo3-ov generate --stdin --model ov-plamo3
```

生成オプション:

- `--max-new-tokens 128`
- `--temperature 0.8`
- `--top-p 0.95`
- `--top-k 50`
- `--stream` / `--no-stream`
- `--skip-prompt` / `--no-skip-prompt`

OpenVINO GenAI 経路は `gemma4_demo.py` と同じく greedy decoding で生成します。
`--temperature` / `--top-p` / `--top-k` / `--stream` は direct fallback でのみ使います。

## チャット

CLI 上で対話するには `chat` を使います。起動時にモデルを一度だけロードし、同じ
セッション内の各ターンで再利用します。

```powershell
uv run plamo3-ov chat --model ov-plamo3-int4 --device GPU --max-new-tokens 128
```

チャット中のコマンド:

- `/exit` または `/quit`: 終了
- `/reset`: 会話履歴をクリア

システムプロンプトを付ける場合:

```powershell
uv run plamo3-ov chat --model ov-plamo3 --system "日本語で簡潔に答えてください。"
```

## 変換ルートの注意点

PLaMo 3 は Hugging Face の custom code モデルです。このリポジトリの変換ルートは
実験的です。

この CLI は `optimum-cli` を使わず、OpenVINO 本体の `openvino.convert_model` で
PLaMo 3 を変換します。

PLaMo 3 の GQA attention は、変換時だけ K/V heads を明示的に展開して OpenVINO の
`ScaledDotProductAttention` に渡します。

推論では OpenVINO GenAI の `LLMPipeline` を優先して使います。GenAI 互換ではない
古い IR の場合だけ、OpenVINO Core と Hugging Face tokenizer を組み合わせた
direct fallback で生成します。fallback では `--max-seq-len` で変換した full-context IR の
固定長の範囲内で生成します。CPU 用に KV cache 付き IR を作る場合は、`--kv-cache` と
`--force` を付けて再変換してください。長い応答が必要な場合は、たとえば次のように
大きめの `--max-seq-len` で再変換してください。

```powershell
uv run plamo3-ov convert --output-dir ov-plamo3-int8 --weight-format int8 --max-seq-len 1024 --force
```

## 出力されるメトリクス

各生成後、stderr に次の形式でメトリクスを表示します。

```text
[metrics] model_load: 12.326s | time_to_first_token: 0.123s | output_tokens: 128 | tokens/sec: 28.02
```

- `model_load`: GenAI pipeline のロード時間
- `time_to_first_token`: first token time
- `output_tokens`: 生成トークン数
- `tokens/sec`: first token 以降の 1 秒あたりの生成トークン数

#!/bin/bash
# pii-shot を Mac にセットアップします（専用の仮想環境に入れるので、他の Python 環境には影響しません）
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$HOME/.pii-shot"
BIN_DIR="$HOME/.local/bin"

# Python 3.10 以上を探す
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    if "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then PY="$c"; break; fi
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10 以上が見つかりません。次のコマンドで入れてから再実行してください:"
  echo "  brew install python@3.12"
  exit 1
fi
echo "使用する Python: $($PY --version)"

mkdir -p "$APP_DIR" "$BIN_DIR"
cp "$HERE/pii_shot.py" "$APP_DIR/pii_shot.py"
"$PY" -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install -q --upgrade pip
echo "依存パッケージをインストール中（数分かかります）..."
"$APP_DIR/venv/bin/pip" install -q -r "$HERE/requirements.txt"

cat > "$BIN_DIR/pii-shot" <<EOF
#!/bin/bash
exec "$APP_DIR/venv/bin/python" "$APP_DIR/pii_shot.py" "\$@"
EOF
chmod +x "$BIN_DIR/pii-shot"

echo "NER モデルを事前にダウンロードします（初回のみ、約1GB）..."
"$APP_DIR/venv/bin/python" -c "from transformers import pipeline; pipeline('token-classification', model='tsmatz/xlm-roberta-ner-japanese')" >/dev/null 2>&1 \
  && echo "モデルの準備ができました。" \
  || echo "※ モデルの事前ダウンロードに失敗しました。初回実行時に再試行されます。"

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc"
     echo "PATH に ~/.local/bin を追加しました。新しいターミナルを開くか 'source ~/.zshrc' を実行してください。" ;;
esac

echo ""
echo "完了しました。使い方: pii-shot --help"

if [ -d "$HOME/.pii-mask" ] || [ -f "$BIN_DIR/pii-mask" ]; then
  echo ""
  echo "旧名 pii-mask のインストールが残っています。不要なら次のコマンドで削除できます:"
  echo "  rm -rf ~/.pii-mask ~/.local/bin/pii-mask"
  echo "（登録語ファイル ~/.pii-mask-words.txt は pii-shot でもそのまま読み込まれます）"
fi

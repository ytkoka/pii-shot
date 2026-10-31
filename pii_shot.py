#!/usr/bin/env python3
"""
pii_shot.py — スクリーンショット内の個人情報を黒塗りするローカルCLI

処理の流れ:
  1. OCR (RapidOCR / PaddleOCR系モデル) で文字と文字ごとの座標を取得
  2. 個人情報の判定
       - 正規表現: メール / URL / 電話番号 / 郵便番号 / 住所 / カード番号 / IPアドレス
       - 敬称・ラベル: 「〇〇様」「〇〇さん」「差出人: 〇〇」など
       - 日本語NER (tsmatz/xlm-roberta-ner-japanese): 人名など
       - ユーザー指定の語 (--mask)
  3. 該当文字の領域を塗りつぶして保存

すべてローカルで動作します（NERモデルは初回のみ Hugging Face からダウンロード）。
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

# --------------------------------------------------------------------------
# データ構造
# --------------------------------------------------------------------------


@dataclass
class Line:
    text: str
    # 文字ごとのボックス (x0, y0, x1, y1)。len(char_boxes) == len(text)
    char_boxes: list[tuple[float, float, float, float]]


@dataclass
class Hit:
    line: int
    start: int
    end: int
    label: str
    source: str
    text: str = ""


@dataclass
class Options:
    use_ner: bool = True
    ner_model: str = "tsmatz/xlm-roberta-ner-japanese"
    ner_tags: set[str] = field(default_factory=lambda: {"PER"})
    ner_threshold: float = 0.6
    mask_words: list[str] = field(default_factory=list)
    allow_words: list[str] = field(default_factory=list)
    mask_orgs: bool = False


# --------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------

_ocr_engine = None


def get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        import logging

        logging.getLogger("RapidOCR").setLevel(logging.WARNING)
        from rapidocr import RapidOCR

        _ocr_engine = RapidOCR()
        logging.getLogger("RapidOCR").setLevel(logging.WARNING)
    return _ocr_engine


def _box_xyxy(pts) -> tuple[float, float, float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def run_ocr(img_path: str) -> list[Line]:
    res = get_ocr()(img_path, return_word_box=True, return_single_char_box=True)
    if res.txts is None:
        return []
    lines: list[Line] = []
    for i, txt in enumerate(res.txts):
        line_box = _box_xyxy(res.boxes[i])
        words = res.word_results[i] if res.word_results is not None else None
        lines.append(Line(txt, _char_boxes(txt, words, line_box)))
    return lines


def _char_boxes(txt, words, line_box):
    """OCRの単語/文字ボックスを、行テキストの1文字ずつのボックスに対応させる。

    英数字は単語単位でまとまって返るので、単語ボックスを文字数で按分する。
    対応付けに失敗した場合は行ボックスを文字数で按分する。
    """
    n = len(txt)
    out: list[tuple[float, float, float, float] | None] = [None] * n
    if words:
        pos = 0
        for w in words:
            wtxt, _score, pts = w
            if not wtxt:
                continue
            idx = txt.find(wtxt, pos)
            if idx < 0:
                continue
            x0, y0, x1, y1 = _box_xyxy(pts)
            L = len(wtxt)
            step = (x1 - x0) / L
            for k in range(L):
                out[idx + k] = (x0 + step * k, y0, x0 + step * (k + 1), y1)
            pos = idx + L
    # 空白などの未対応文字は前後から補完、全滅なら行ボックスを按分
    if all(b is None for b in out):
        x0, y0, x1, y1 = line_box
        step = (x1 - x0) / max(n, 1)
        return [(x0 + step * k, y0, x0 + step * (k + 1), y1) for k in range(n)]
    for k in range(n):
        if out[k] is None:
            prev = next((out[j] for j in range(k - 1, -1, -1) if out[j]), None)
            nxt = next((out[j] for j in range(k + 1, n) if out[j]), None)
            ref = prev or nxt
            if prev and nxt:
                out[k] = (prev[2], min(prev[1], nxt[1]), nxt[0], max(prev[3], nxt[3]))
            else:
                out[k] = ref
    return out  # type: ignore[return-value]


# --------------------------------------------------------------------------
# 検出: 正規表現
# --------------------------------------------------------------------------

KANJI_KANA = r"一-龥々〆ヵヶぁ-んァ-ヴー"

REGEX_RULES: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    ("URL", re.compile(r"(?:https?://|www\.)[^\s<>\"'（）()]+")),
    # 電話番号: 0X0-XXXX-XXXX / 03-XXXX-XXXX / 0120-XXX-XXX / +81 ...
    (
        "PHONE",
        re.compile(
            r"(?<![\d\-])(?:\+81[\s\-]?\d{1,4}|0\d{1,4})[\s\-－ー‐]?\(?\d{1,4}\)?[\s\-－ー‐]?\d{3,4}(?![\d\-])"
        ),
    ),
    ("POSTAL", re.compile(r"〒?\s?(?<![\d\-])\d{3}[\-－ー‐]\d{4}(?![\d\-])")),
    (
        "ADDRESS",
        re.compile(
            r"(?:北海道|東京都|(?:京都|大阪)府|(?:青森|岩手|宮城|秋田|山形|福島|茨城|栃木|群馬|埼玉|千葉|神奈川|新潟|富山|石川|福井|山梨|長野|岐阜|静岡|愛知|三重|滋賀|兵庫|奈良|和歌山|鳥取|島根|岡山|広島|山口|徳島|香川|愛媛|高知|福岡|佐賀|長崎|熊本|大分|宮崎|鹿児島|沖縄)県)"
            r"[^\s、。,，|｜<>（）()]{2,60}"
        ),
    ),
    ("CARD", re.compile(r"(?<!\d)(?:\d{4}[\s\-]){3}\d{1,4}(?!\d)|(?<!\d)\d{15,16}(?!\d)")),
    ("IP", re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")),
]

# 敬称の前の名前。「皆様」「お客様」などは除外する
HONORIFIC = re.compile(rf"([{KANJI_KANA}A-Za-z]{{1,10}})\s?(?:様|さま|さん|殿|氏|くん|君|ちゃん|先生)")
HONORIFIC_STOP = {
    "皆", "皆さん", "お客", "客", "各位", "担当", "担当者", "ご担当", "ご担当者", "関係者", "みな",
    "みなさん", "お父", "お母", "奥", "お子", "神", "王", "殿", "お疲れ", "ご苦労", "お世話", "ご覧",
    "おかげ", "お陰", "ご馳走", "ご無沙汰", "お待ち遠", "皆々", "利用者", "会員", "ユーザー",
}

# 「差出人: 山田太郎」のようなラベルの後ろの名前
LABELED_NAME = re.compile(
    rf"(?:差出人|送信者|送信元|宛先|宛名|受取人|氏名|名前|お名前|担当者?|作成者|更新者|申請者|承認者|From|To|Cc|CC|Bcc|Name)\s*[:：]\s*"
    rf"([{KANJI_KANA}A-Za-z][{KANJI_KANA}A-Za-z\s・.]{{0,30}}?)(?=\s*(?:<|＜|\(|（|様|さん|,|，|、|$|\||｜))"
)


def detect_regex(lines: list[Line]) -> list[Hit]:
    hits: list[Hit] = []
    for li, ln in enumerate(lines):
        t = ln.text
        for label, pat in REGEX_RULES:
            for m in pat.finditer(t):
                hits.append(Hit(li, m.start(), m.end(), label, "regex"))
        for m in HONORIFIC.finditer(t):
            name = m.group(1)
            # 「お世話になっております。山田様」→ 末尾の連続した名前部分だけを使う
            if name in HONORIFIC_STOP or any(name.endswith(s) and len(s) >= 2 for s in HONORIFIC_STOP):
                continue
            if name.startswith(("お", "ご")) and len(name) <= 3:
                continue
            hits.append(Hit(li, m.start(1), m.end(1), "PERSON", "honorific"))
        for m in LABELED_NAME.finditer(t):
            s, e = m.start(1), m.end(1)
            while e > s and t[e - 1].isspace():
                e -= 1
            if e > s:
                hits.append(Hit(li, s, e, "PERSON", "label"))
    return hits


# --------------------------------------------------------------------------
# 検出: 日本語NER
# --------------------------------------------------------------------------

_ner = None

NER_STOP = {
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "january", "february", "march", "april", "june", "july", "august", "september", "october",
    "november", "december",
    "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}


def get_ner(model_name: str):
    global _ner
    if _ner is None:
        import logging
        import warnings

        warnings.filterwarnings("ignore")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        # macOS で torch が出す "Redirects are currently not supported" を抑止
        logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
        import contextlib
        import io

        with contextlib.redirect_stderr(io.StringIO()):
            import torch  # noqa: F401  （import 時の警告を捨てる）
            from transformers import pipeline
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()

        _ner = pipeline("token-classification", model=model_name, aggregation_strategy="simple")
    return _ner


def detect_ner(lines: list[Line], opt: Options) -> list[Hit]:
    if not lines:
        return []
    ner = get_ner(opt.ner_model)
    texts = [ln.text for ln in lines]
    results = ner(texts)
    hits: list[Hit] = []
    for li, ents in enumerate(results):
        for e in ents:
            tag = e.get("entity_group", "")
            if tag not in opt.ner_tags or float(e.get("score", 0)) < opt.ner_threshold:
                continue
            s, en = int(e["start"]), int(e["end"])
            while s < en and texts[li][s].isspace():
                s += 1
            word = texts[li][s:en].strip()
            # 誤検出の除外:
            #  - 1文字だけの行（アイコンが「田」「日」などと読まれたもの）や1文字の人名
            #  - 月・曜日の略称（GitHub の草グラフやカレンダーの見出しなど）
            if len(texts[li].strip()) <= 1 or len(word) <= 1:
                continue
            if word.lower().rstrip(".") in NER_STOP:
                continue
            if en > s:
                hits.append(Hit(li, s, en, "PERSON" if tag == "PER" else tag, "ner"))
    return hits


# --------------------------------------------------------------------------
# 検出: ユーザー指定語
# --------------------------------------------------------------------------


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def detect_words(lines: list[Line], words: list[str]) -> list[Hit]:
    """空白の有無を無視して一致させる（OCRは空白を落とすことがあるため）"""
    hits: list[Hit] = []
    for li, ln in enumerate(lines):
        # 空白を除いた文字列と元のインデックスの対応表
        idx = [i for i, c in enumerate(ln.text) if not c.isspace()]
        compact = "".join(ln.text[i] for i in idx)
        low = compact.lower()
        for w in words:
            w2 = _norm(w).lower()
            if not w2:
                continue
            p = low.find(w2)
            while p >= 0:
                hits.append(Hit(li, idx[p], idx[p + len(w2) - 1] + 1, "CUSTOM", "word"))
                p = low.find(w2, p + 1)
    return hits


def apply_allow(lines: list[Line], hits: list[Hit], allow: list[str]) -> list[Hit]:
    if not allow:
        return hits
    allow_n = {_norm(a).lower() for a in allow}
    return [h for h in hits if _norm(lines[h.line].text[h.start : h.end]).lower() not in allow_n]


# --------------------------------------------------------------------------
# 塗りつぶし
# --------------------------------------------------------------------------


def hit_rect(ln: Line, h: Hit, pad: int):
    boxes = ln.char_boxes[h.start : h.end]
    # OCRの文字境界は数px ずれることがあるので、横方向は文字の高さに比例して広げる（隠し過ぎ側に倒す）
    height = max(b[3] for b in boxes) - min(b[1] for b in boxes)
    pad_x = max(pad, 0.3 * height)
    x0 = min(b[0] for b in boxes) - pad_x
    y0 = min(b[1] for b in boxes) - pad
    x1 = max(b[2] for b in boxes) + pad_x
    y1 = max(b[3] for b in boxes) + pad
    return (int(x0), int(y0), int(x1 + 0.999), int(y1 + 0.999))


def merge_hits(hits: list[Hit]) -> list[Hit]:
    """同じ行の重なる範囲をまとめる"""
    hits = sorted(hits, key=lambda h: (h.line, h.start, -h.end))
    out: list[Hit] = []
    for h in hits:
        if out and out[-1].line == h.line and h.start <= out[-1].end:
            last = out[-1]
            if h.end > last.end:
                last.end = h.end
            if h.label not in last.label.split("+"):
                last.label += "+" + h.label
            continue
        out.append(Hit(h.line, h.start, h.end, h.label, h.source))
    return out


def render(img: Image.Image, lines: list[Line], hits: list[Hit], style: str, color, pad: int) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for h in hits:
        r = hit_rect(lines[h.line], h, pad)
        r = (max(r[0], 0), max(r[1], 0), min(r[2], out.width), min(r[3], out.height))
        if r[2] <= r[0] or r[3] <= r[1]:
            continue
        if style == "fill":
            draw.rectangle(r, fill=color)
        else:  # mosaic + blur（文字の形が残らないよう強めに）
            region = out.crop(r)
            w, h_ = region.size
            small = region.resize((max(1, w // 12), max(1, h_ // 12)), Image.BILINEAR)
            region = small.resize((w, h_), Image.NEAREST).filter(ImageFilter.GaussianBlur(3))
            out.paste(region, r[:2])
    return out


def render_preview(img: Image.Image, lines: list[Line], hits: list[Hit], pad: int) -> Image.Image:
    """確認用: 元画像に検出範囲を赤枠で、OCRで読めた行を薄い青枠で表示"""
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for ln in lines:
        if ln.char_boxes:
            xs0 = min(b[0] for b in ln.char_boxes)
            ys0 = min(b[1] for b in ln.char_boxes)
            xs1 = max(b[2] for b in ln.char_boxes)
            ys1 = max(b[3] for b in ln.char_boxes)
            draw.rectangle((xs0, ys0, xs1, ys1), outline=(120, 170, 255), width=1)
    for h in hits:
        r = hit_rect(lines[h.line], h, pad)
        draw.rectangle(r, outline=(230, 0, 0), width=3)
        draw.text((r[0], max(r[1] - 12, 0)), h.label, fill=(230, 0, 0))
    return out


# --------------------------------------------------------------------------
# クリップボード (macOS)
# --------------------------------------------------------------------------


def clipboard_image_to_file() -> str:
    from PIL import ImageGrab

    im = ImageGrab.grabclipboard()
    if not isinstance(im, Image.Image):
        sys.exit("クリップボードに画像がありません。⌘+Shift+Ctrl+4 などで画像をコピーしてから実行してください。")
    fd, path = tempfile.mkstemp(suffix=".png", prefix="pii_shot_clip_")
    os.close(fd)
    im.save(path)
    return path


def file_to_clipboard(path: str) -> None:
    if sys.platform != "darwin":
        print("※ クリップボードへの書き戻しは macOS のみ対応です", file=sys.stderr)
        return
    p = str(Path(path).resolve()).replace('"', '\\"')
    script = f'set the clipboard to (read (POSIX file "{p}") as «class PNGf»)'
    subprocess.run(["osascript", "-e", script], check=True)


# --------------------------------------------------------------------------
# メイン
# --------------------------------------------------------------------------


def process(path: str, opt: Options, args) -> tuple[str, list[Hit]]:
    img = Image.open(path)
    lines = run_ocr(path)
    hits = detect_regex(lines)
    if opt.mask_words:
        hits += detect_words(lines, opt.mask_words)
    if opt.use_ner:
        hits += detect_ner(lines, opt)
    hits = apply_allow(lines, hits, opt.allow_words)
    hits = merge_hits(hits)
    for h in hits:
        h.text = lines[h.line].text[h.start : h.end]

    src = Path(path)
    outdir = Path(args.outdir) if args.outdir else src.parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = src.stem if not args.clipboard else "clipboard"
    out_path = outdir / f"{stem}_masked.png"
    color = tuple(int(args.color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    render(img, lines, hits, args.style, color, args.pad).save(out_path)
    if args.preview:
        render_preview(img, lines, hits, args.pad).save(outdir / f"{stem}_preview.png")
    if args.show_ocr:
        print(f"--- OCR: {path}")
        for ln in lines:
            print("  ", ln.text)
    return str(out_path), hits


def resolve_path(p: str) -> str:
    """よくある入力ミスを吸収してパスを解決する。

    - "…\\ …" のようにクォートとバックスラッシュを併用した場合の余分な "\\"
    - macOS のファイル名の Unicode 正規化（NFD）との不一致
    - 端末の折り返しなどで紛れ込んだ空白・不可視文字
    """
    import unicodedata

    if Path(p).exists():
        return p
    cands = [p.replace("\\ ", " ").replace("\\", "")]
    for c in list(cands):
        cands += [unicodedata.normalize("NFC", c), unicodedata.normalize("NFD", c)]
    for c in cands:
        if Path(c).expanduser().exists():
            return str(Path(c).expanduser())
    # 同じフォルダ内で、空白・不可視文字・正規化の違いを無視して一致するファイルを探す
    base = cands[0]
    parent = Path(base).expanduser().parent
    key = lambda s: re.sub(r"[\s​  ]", "", unicodedata.normalize("NFC", s))
    if parent.is_dir():
        target = key(Path(base).name)
        for f in parent.iterdir():
            if key(f.name) == target:
                return str(f)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="pii-shot",
        description="スクリーンショットの個人情報（氏名・メール・電話番号・住所など）をローカルで黒塗りします。",
    )
    ap.add_argument("images", nargs="*", help="入力画像（複数可）")
    ap.add_argument("-c", "--clipboard", action="store_true", help="クリップボードの画像を処理し、結果をクリップボードに戻す")
    ap.add_argument("-o", "--outdir", help="出力先フォルダ（既定: 入力と同じ場所 / クリップボード時は ~/Desktop）")
    ap.add_argument("-p", "--preview", action="store_true", help="検出範囲を赤枠で示した確認用画像も出力")
    ap.add_argument("-m", "--mask", action="append", default=[], help="必ず隠す語（例: 自分の名前・社名）。複数指定可")
    ap.add_argument("-a", "--allow", action="append", default=[], help="隠さない語（誤検出の除外）。複数指定可")
    ap.add_argument("--words-file", help="必ず隠す語を1行1語で書いたファイル（既定: ~/.pii-shot-words.txt があれば読む）")
    ap.add_argument("--no-ner", action="store_true", help="NERモデルを使わない（正規表現と敬称ルールのみ。高速）")
    ap.add_argument("--orgs", action="store_true", help="組織名（会社名など）も隠す")
    ap.add_argument("--places", action="store_true", help="地名も隠す（住所以外の地名を含む）")
    ap.add_argument("--ner-model", default="tsmatz/xlm-roberta-ner-japanese", help="Hugging Face の token-classification モデル")
    ap.add_argument("--threshold", type=float, default=0.6, help="NERのしきい値（既定 0.6。下げると拾いやすく、誤検出も増える）")
    ap.add_argument("--style", choices=["fill", "mosaic"], default="fill", help="塗りつぶし方（既定 fill）")
    ap.add_argument("--color", default="#000000", help="塗りつぶし色（既定 #000000）")
    ap.add_argument("--pad", type=int, default=2, help="塗りつぶしの余白px（既定 2）")
    ap.add_argument("--show-ocr", action="store_true", help="OCRで読み取った全文を表示（取りこぼし確認用）")
    ap.add_argument("-q", "--quiet", action="store_true", help="検出結果の一覧を表示しない")
    args = ap.parse_args(argv)

    if not args.images and not args.clipboard:
        ap.print_help()
        return 1

    words = list(args.mask)
    if args.words_file:
        word_files = [Path(args.words_file).expanduser()]
    else:
        # 旧名 (pii-mask) の登録語ファイルも読む
        word_files = [Path("~/.pii-shot-words.txt").expanduser(), Path("~/.pii-mask-words.txt").expanduser()]
    for wf in word_files:
        if wf.exists():
            words += [w.strip() for w in wf.read_text(encoding="utf-8").splitlines() if w.strip() and not w.startswith("#")]

    tags = {"PER"}
    if args.orgs:
        tags |= {"ORG", "ORG-O", "ORG-P"}
    if args.places:
        tags |= {"LOC"}
    opt = Options(
        use_ner=not args.no_ner,
        ner_model=args.ner_model,
        ner_tags=tags,
        ner_threshold=args.threshold,
        mask_words=words,
        allow_words=args.allow,
    )

    inputs = list(args.images)
    clip_tmp = None
    if args.clipboard:
        clip_tmp = clipboard_image_to_file()
        inputs = [clip_tmp]
        if not args.outdir:
            args.outdir = str(Path("~/Desktop").expanduser())

    for p in inputs:
        p = resolve_path(p)
        if not Path(p).exists():
            print(f"見つかりません: {p}", file=sys.stderr)
            print("  ヒント: ファイルを Finder からターミナルにドラッグ＆ドロップすると、正しいパスが入力されます。", file=sys.stderr)
            continue
        out, hits = process(p, opt, args)
        if not args.quiet:
            print(f"■ {p if not clip_tmp else 'クリップボード'} → {out}  （{len(hits)}箇所）")
            for h in hits:
                print(f"   [{h.label}] {h.text}")
        if clip_tmp:
            file_to_clipboard(out)
            print("マスク済み画像をクリップボードにコピーしました。そのまま貼り付けできます。")
    if clip_tmp:
        os.remove(clip_tmp)
    print("※ 自動検出は完全ではありません。送信前に必ず目視で確認してください。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

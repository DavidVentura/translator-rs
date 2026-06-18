#!/usr/bin/env bash
# End-to-end Galician sabela lexicon build. Idempotent: cached downloads/clone
# survive a reboot under $WORK, so a crash only re-runs the extraction itself.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$HOME/gl_sabela_build}"
AHO="$WORK/aHoTTS"
GLDIR="$AHO/ahotts/dicts/gl/cotovia/lang/gl"
mkdir -p "$WORK"

step() { printf '\n=== %s ===\n' "$1"; }

step "clone aHoTTS"
[ -d "$AHO" ] || git clone --depth 1 https://github.com/hitz-zentroa/aHoTTS "$AHO"

step "fetch sabela voice + install stub"
mkdir -p "$AHO/ahotts/voices/gl/sabela" "$AHO/ahotts/voices/gl/stub" "$AHO/output"
[ -f "$AHO/ahotts/voices/gl/sabela/vits.onnx" ] || \
  curl -sL "https://huggingface.co/HiTZ/TTS-gl_sabela/resolve/main/vits.onnx" \
       -o "$AHO/ahotts/voices/gl/sabela/vits.onnx"
cp "$HERE/stub_vits.onnx" "$AHO/ahotts/voices/gl/stub/vits.onnx"

step "build interposer (host-local toolchain)"
gcc -shared -fPIC -O2 -I"$HERE" "$HERE/ort_intercept.c" -o "$HERE/ort_intercept.so" -ldl

step "fetch Galician hunspell"
[ -f "$WORK/gl.aff" ] || curl -sL "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/gl/gl_ES.aff" -o "$WORK/gl.aff"
[ -f "$WORK/gl.dic" ] || curl -sL "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/gl/gl_ES.dic" -o "$WORK/gl.dic"

step "build wordlist (cotovia dicts + hunspell expansion)"
bash "$HERE/build_wordlist.sh" "$GLDIR" "$WORK/words_dict.txt"
python3 "$HERE/expand_hunspell.py" --aff "$WORK/gl.aff" --dic "$WORK/gl.dic" --out "$WORK/words_expanded.txt"
sort -u "$WORK/words_dict.txt" "$WORK/words_expanded.txt" > "$WORK/words_all.txt"
echo "union wordlist: $(wc -l <"$WORK/words_all.txt")"

step "extract word -> ids lexicon (stub voice; JOBS=${JOBS:-4} cores)"
JOBS="${JOBS:-4}" bash "$HERE/parallel_extract.sh" "$AHO" "$WORK/words_all.txt" "$WORK/gl_lexicon.txt"

step "compress + report"
zstd -19 -q -f "$WORK/gl_lexicon.txt" -o "$WORK/gl_lexicon.txt.zst" 2>/dev/null || \
  echo "(zstd cli missing; raw lexicon at $WORK/gl_lexicon.txt)"
echo "entries: $(wc -l <"$WORK/gl_lexicon.txt")"
echo "raw:  $(wc -c <"$WORK/gl_lexicon.txt") bytes"
[ -f "$WORK/gl_lexicon.txt.zst" ] && echo "zstd: $(wc -c <"$WORK/gl_lexicon.txt.zst") bytes"

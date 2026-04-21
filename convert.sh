set -eu
find samples -name '*.wav' -print0 | while IFS="" read -r -d '' f; do
  rel="${f#samples/}"
  out="samples_ogg/${rel%.wav}.ogg"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ]; then
    # echo "skipping $f -> $out"
    continue
  fi
  echo "converting $f -> $out"
  (
    ffmpeg -nostdin -nostats -loglevel error -hide_banner -i "$f" -c:a libopus -b:a 48k -vbr on -compression_level 10 "$out"
  ) &
  sleep 0.1
done

wait

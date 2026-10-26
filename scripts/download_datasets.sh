export HF_ENDPOINT=https://hf-mirror.com

# StreamingBench
# sudo apt install aria2
bash datasets/hfd.sh mjuicem/StreamingBench --dataset --local-dir ./datasets/StreamingBench --include "StreamingBench/Real_Time_Visual_Understanding.csv"
bash datasets/hfd.sh mjuicem/StreamingBench --dataset --local-dir ./datasets/StreamingBench --include "Real-Time Visual Understanding_*.zip"
bash scripts/unzip_streamingbench.sh
pushd ./datasets/StreamingBench >/dev/null
mkdir -p Real-Time_Visual_Understanding
shopt -s nullglob
sample_dirs=(sample_*)
if (( ${#sample_dirs[@]} > 0 )); then
  mv "${sample_dirs[@]}" ./Real-Time_Visual_Understanding/
fi
shopt -u nullglob
popd >/dev/null

# OVO-Bench
bash datasets/hfd.sh JoeLeelyf/OVO-Bench --dataset --local-dir ./datasets/OVO-Bench --include "src_videos.tar.part*"
wget -O datasets/OVO-Bench/ovo_bench_new.json  https://raw.githubusercontent.com/JoeLeelyf/OVO-Bench/main/data/ovo_bench_new.json
bash scripts/unzip_ovobench.sh

# Video-MME
bash datasets/hfd.sh lmms-lab/Video-MME --dataset --local-dir ./datasets/Video-MME --include "videomme/test-00000-of-00001.parquet"
bash datasets/hfd.sh lmms-lab/Video-MME --dataset --local-dir ./datasets/Video-MME --include "videos_chunked_*.zip"
bash scripts/unzip_videomme.sh

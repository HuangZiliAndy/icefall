#!/usr/bin/env bash
# Download the AMI data used by this recipe and by the SSL recipe
# egs/librispeech/SSL/zipformer_causal_fbank:
#
#   <ami_dir>/<meeting>/audio/<meeting>.Headset-<ch>.wav   IHM (openslr 16)
#   <ami_dir>/ami_public_manual_1.6.2/                     NXT manual annotations
#   <mix_dir>/<meeting>.Mix-Headset.wav                    official IHM-Mix audio
#                                                          (Edinburgh mirror only)
#
# All downloads are resumable: re-run to continue.
#
# usage: local/download_ami.sh <ami_dir> <mix_dir>

set -eou pipefail

ami_dir=$1
mix_dir=$2
mkdir -p $ami_dir $mix_dir

# 1. Individual headset audio
if [ ! -f $ami_dir/.headset.done ]; then
  url=https://openslr.trmal.net/resources/16/headset.tar.gz
  wget -c --tries=20 --timeout=60 --progress=dot:giga $url -O $ami_dir/headset.tar.gz.part
  mv $ami_dir/headset.tar.gz.part $ami_dir/headset.tar.gz
  tar --skip-old-files -xzf $ami_dir/headset.tar.gz -C $ami_dir
  touch $ami_dir/.headset.done
fi
echo "headset wavs: $(find $ami_dir -name '*.Headset-*.wav' | wc -l)"

# 2. Manual annotations
if [ ! -d $ami_dir/ami_public_manual_1.6.2 ]; then
  url=https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip
  wget -c --tries=20 --timeout=60 $url -O $ami_dir/ami_public_manual_1.6.2.zip
  unzip -q -d $ami_dir/ami_public_manual_1.6.2 $ami_dir/ami_public_manual_1.6.2.zip
fi

# 3. Official Mix-Headset audio of every meeting that has headset audio
url=https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus
for m in $(ls $ami_dir/*/audio/*.Headset-0.wav | xargs -n1 basename | sed 's/\.Headset-0\.wav$//'); do
  f=$mix_dir/$m.Mix-Headset.wav
  [ -f $f ] && continue
  if curl -sS -f -C - --retry 10 --retry-delay 5 --connect-timeout 30 \
       -o $f.part "$url/$m/audio/$m.Mix-Headset.wav"; then
    mv $f.part $f
  else
    echo "FAILED $m"
  fi
done
echo "Mix-Headset wavs: $(ls $mix_dir/*.Mix-Headset.wav | wc -l)"

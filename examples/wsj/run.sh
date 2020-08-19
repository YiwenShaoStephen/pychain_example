#!/bin/bash
# Copyright (c) Yiwen Shao

# Apache 2.0

. ./cmd.sh
set -e -o pipefail

stage=0
ngpus=1 # num GPUs for multiple GPUs training within a single node; should match those in $free_gpu
free_gpu= # comma-separated available GPU ids, eg., "0" or "0,1"; automatically assigned if on CLSP grid

# data related
rootdir=data
dumpdir=data/dump   # directory to dump full features
langdir=data/lang   # directory for language models
graphdir=data/graph # directory for chain graphs (FSTs)
wsj0=
wsj1=
if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
  wsj0=/export/corpora5/LDC/LDC93S6B
  wsj1=/export/corpora5/LDC/LDC94S13B
fi

# Data splits
train_set=train_si284
valid_set=test_dev93
test_set=test_eval92

# feature configuration
do_delta=false

# Model options
unit=phone # phone/char
type=mono # mono/bi

affix=

. ./path.sh
. ./utils/parse_options.sh

dir=exp/tdnn_${type}${unit}${affix:+_$affix}
lang=$langdir/lang_${type}${unit}_e2e
graph=$graphdir/${type}${unit}

if [ ${stage} -le 0 ]; then
  echo "Stage 0: Data Preparation"
  local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.?
  srcdir=data/local/data
  for x in $train_set $valid_set $test_set; do
    mkdir -p $rootdir/$x
    cp $srcdir/${x}_wav.scp $rootdir/$x/wav.scp || exit 1;
    cp $srcdir/$x.txt $rootdir/$x/text || exit 1;
    cp $srcdir/$x.spk2utt $rootdir/$x/spk2utt || exit 1;
    cp $srcdir/$x.utt2spk $rootdir/$x/utt2spk || exit 1;
    utils/filter_scp.pl $rootdir/$x/spk2utt $srcdir/spk2gender > $rootdir/$x/spk2gender || exit 1;
  done
fi

if [ $stage -le 1 ]; then
  echo "Stage 1: Feature Extraction"
  ./prepare_feat.sh --wsj0 $wsj0 \
		    --wsj1 $wsj1 \
		    --train_set $train_set \
		    --valid_set $valid_set \
		    --test_set $test_set \
		    --dumpdir $dumpdir \
		    --rootdir $rootdir
fi

if [ ${stage} -le 2 ]; then
  echo "Stage 2:  Dictionary and LM Preparation"
  ./prepare_lang.sh --langdir $langdir \
		    --unit $unit \
		    --wsj1 $wsj1
fi

if [ $stage -le 3 ]; then
  echo "Stage 3: Graph Preparation"
  ./prepare_graph.sh --train_set $train_set \
		     --valid_set $valid_set \
		     --rootdir $rootdir \
		     --graphdir $graphdir \
		     --langdir $langdir \
		     --type $type \
		     --unit $unit
fi

if [ ${stage} -le 4 ]; then
  echo "Stage 4: Dump Json Files"
  train_wav=$rootdir/$train_set/wav.scp
  train_dur=$rootdir/$train_set/utt2dur
  train_feat=$dumpdir/$train_set/feats.scp
  train_fst=$graph/$train_set/num.scp
  train_text=$rootdir/$train_set/text
  train_utt2num_frames=$rootdir/$train_set/utt2num_frames

  valid_wav=$rootdir/$valid_set/wav.scp
  valid_dur=$rootdir/$valid_set/utt2dur
  valid_feat=$dumpdir/$valid_set/feats.scp
  valid_fst=$graph/$valid_set/num.scp
  valid_text=$rootdir/$valid_set/text
  valid_utt2num_frames=$dumpdir/$valid_set/utt2num_frames

  test_wav=$rootdir/$test_set/wav.scp
  test_dur=$rootdir/$test_set/utt2dur
  test_feat=$dumpdir/$test_set/feats.scp
  test_text=$rootdir/$test_set/text
  test_utt2num_frames=$dumpdir/$test_set/utt2num_frames
  
  asr_prep_json.py --wav-files $train_wav \
		   --dur-files $train_dur \
		   --feat-files $train_feat \
		   --numerator-fst-files $train_fst \
		   --text-files $train_text \
		   --num-frames-files $train_utt2num_frames \
		   --output data/train_${type}${unit}.json
  asr_prep_json.py --wav-files $valid_wav \
		   --dur-files $valid_dur \
		   --feat-files $valid_feat \
		   --numerator-fst-files $valid_fst \
		   --text-files $valid_text \
		   --num-frames-files $valid_utt2num_frames \
		   --output data/valid_${type}${unit}.json
  asr_prep_json.py --wav-files $test_wav \
		   --dur-files $test_dur \
		   --feat-files $test_feat \
		   --text-files $test_text \
		   --num-frames-files $test_utt2num_frames \
		   --output data/test_${type}${unit}.json
fi

num_targets=$(tree-info $graph/tree | grep num-pdfs | awk '{print $2}')

if [ ${stage} -le 5 ]; then
  echo "Stage 5: Model Training"
  opts=""
  mkdir -p $dir/logs
  log_file=$dir/logs/train.log
  python3 train.py \
    --train data/train_${type}${unit}.json \
    --valid data/valid_${type}${unit}.json \
    --den-fst $graph/normalization.fst \
    --epochs 20 \
    --dropout 0.2 \
    --wd 0.01 \
    --optimizer adam \
    --lr 0.001 \
    --scheduler plateau \
    --gamma 0.5 \
    --hidden-dims 384 384 384 384 384 \
    --curriculum 1 \
    --num-targets $num_targets \
    --seed 1 \
    --exp $dir 2>&1 | tee $log_file
fi

if [ ${stage} -le 6 ]; then
  echo "Stage 6: Dumping Posteriors for Test Data"
  path=$dir/$checkpoint
  log_file=$dir/logs/dump_$test_set.log
  result_file=$test_set/posteriors.ark
  mkdir -p $dir/$test_set
  $cuda_cmd $log_file test.py \
	    --test data/test_${type}${unit}.json \
	    --exp $dir \
	    --model model_best.pth.tar \
	    --results $result_file
fi

if [ ${stage} -le 7 ]; then
  echo "Stage 7: Trigram LM Decoding"
  decode_dir=$dir/decode/$test_set/bd_tgpr
  mkdir -p $decode_dir
  latgen-faster-mapped --acoustic-scale=1.0 --beam=15 --lattice-beam=8 \
		       --word-symbol-table="$graph/graph_bd_tgpr/words.txt" \
		       $graph/0.trans_mdl $graph/graph_bd_tgpr/HCLG.fst \
		       ark:$dir/$test_set/posteriors.ark \
		       "ark:|lattice-scale --acoustic-scale=10.0 ark:- ark:- | gzip -c >$decode_dir/lat.1.gz" \
		       2>&1 | tee $dir/logs/decode_$test_set.log
fi


if [  $stage -le 8 ]; then
  echo "Stage 8: Forthgram LM rescoring"
  oldlang=$langdir/$unit/lang_${unit}_test_bd_tgpr
  newlang=$langdir/$unit/lang_${unit}_test_bd_fgconst
  oldlm=$oldlang/G.fst
  newlm=$newlang/G.carpa
  oldlmcommand="fstproject --project_output=true $oldlm |"
  olddir=$dir/decode/$test_set/bd_tgpr
  newdir=$dir/decode/$test_set/fgconst
  mkdir -p $newdir
  $train_cmd $dir/logs/rescorelm_$test_set.log \
	     lattice-lmrescore --lm-scale=-1.0 \
	     "ark:gunzip -c ${olddir}/lat.1.gz|" "$oldlmcommand" ark:- \| \
	     lattice-lmrescore-const-arpa --lm-scale=1.0 \
	     ark:- "$newlm" "ark,t:|gzip -c>$newdir/lat.1.gz"
fi

if [ ${stage} -le 9 ]; then
  echo "Stage 9: Computing WER"
  for lmtype in bd_tgpr fgconst; do
    local/score_kaldi_wer.sh $rootdir/$test_set $graph/graph_bd_tgpr $dir/decode/$test_set/$lmtype
    echo "Best WER for $dataset with $lmtype:"
    cat $dir/decode/$test_set/$lmtype/scoring_kaldi/best_wer
  done
fi

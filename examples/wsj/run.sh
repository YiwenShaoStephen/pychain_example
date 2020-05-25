#!/bin/bash
# Copyright (c) Yiwen Shao

# Apache 2.0

set -e -o pipefail

stage=0
ngpus=1 # num GPUs for multiple GPUs training within a single node; should match those in $free_gpu
free_gpu= # comma-separated available GPU ids, eg., "0" or "0,1"; automatically assigned if on CLSP grid

# E2E model related
affix=
train_set=train_si284
valid_set=test_dev93
test_set=test_eval92
treedir=data/graph  # contain numerator fst

# data related
dumpdir=data/dump   # directory to dump full features
wsj0=
wsj1=
if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
  wsj0=/export/corpora5/LDC/LDC93S6B
  wsj1=/export/corpora5/LDC/LDC94S13B
fi

# feature configuration
do_delta=false

. ./path.sh
. ./utils/parse_options.sh

dir=exp/tdnn${affix:+_$affix}

if [ ${stage} -le 0 ]; then
  echo "Stage 0: Data Preparation"
  ./prepare_data.sh --train_set $train_set \
		    --valid_set $valid_set \
		    --test_set $test_set \
		    --dumpdir $dumpdir \
		    --wsj0 $wsj0 \
		    --wsj1 $wsj1 \
		    --do_delta $do_delta
fi

if [ $stage -le 1 ]; then
  echo "Stage 1: Graph Preparation"
  ./prepare_graph.sh --train_set $train_set \
		     --valid_set $valid_set \
		     --treedir $treedir
fi

if [ ${stage} -le 2 ]; then
  echo "Stage 2: Dump Json Files"
  train_feat=$dumpdir/$train_set/feats.scp
  train_fst=$treedir/$train_set/num.scp
  train_text=data/$train_set/text
  train_utt2num_frames=$dumpdir/$train_set/utt2num_frames
  
  valid_feat=$dumpdir/$valid_set/feats.scp
  valid_fst=$treedir/$valid_set/num.scp
  valid_text=data/$valid_set/text
  valid_utt2num_frames=$dumpdir/$valid_set/utt2num_frames

  test_feat=$dumpdir/$test_set/feats.scp
  test_text=data/$test_set/text
  test_utt2num_frames=$dumpdir/$test_set/utt2num_frames
  
  asr_prep_json.py --feat-files $train_feat --numerator-fst-files $train_fst --text-files $train_text --utt2num-frames-files $train_utt2num_frames --output data/train.json
  asr_prep_json.py --feat-files $valid_feat --numerator-fst-files $valid_fst --text-files $valid_text --utt2num-frames-files $valid_utt2num_frames --output data/valid.json
  asr_prep_json.py --feat-files $test_feat --text-files $test_text --utt2num-frames-files $test_utt2num_frames --output data/test.json
fi

num_targets=$(tree-info $treedir/tree | grep num-pdfs | awk '{print $2}')

if [ ${stage} -le 3 ]; then
  echo "Stage 3: Model Training"
  opts=""
  valid_subset=valid
  mkdir -p $dir/logs
  log_file=$dir/logs/train.log
  python3 train.py \
    --train data/train.json \
    --valid data/valid.json \
    --den-fst $treedir/normalization.fst \
    --epochs 15 \
    --wd 0.1 \
    --optimizer adam \
    --lr 0.001 \
    --scheduler plateau \
    --gamma 0.5 \
    --arch TDNN \
    --hidden-dims 256 256 256 256 256 \
    --curriculum 1 \
    --num-targets $num_targets \
    --seed 1 \
    --exp $dir 2>&1 | tee $log_file
fi

if [ ${stage} -le 4 ]; then
  echo "Stage 4: Dumping"
  path=$dir/$checkpoint
  log_file=$dir/logs/dump_$test_set.log
  result_file=$test_set/posteriors.ark
  mkdir -p $dir/$test_set
  test.py \
    --test data/test.json \
    --exp $dir \
    --model model_best.pth.tar \
    --results $result_file 2>&1 | tee $log_file
fi

if [ ${stage} -le 5 ]; then
  echo "Stage 5: Decoding"
  decode_dir=$dir/decode/$test_set/bd_tgpr
  mkdir -p $decode_dir
  latgen-faster-mapped --acoustic-scale=1.0 --beam=15 --lattice-beam=8 \
		       --word-symbol-table="$treedir/graph_bd_tgpr/words.txt" \
		       $treedir/0.trans_mdl $treedir/graph_bd_tgpr/HCLG.fst \
		       ark:$dir/$test_set/posteriors.ark \
		       "ark:|lattice-scale --acoustic-scale=10.0 ark:- ark:- | gzip -c >$decode_dir/lat.1.gz" \
		       2>&1 | tee $dir/logs/decode_$test_set.log
fi


if [  $stage -le 6 ]; then
  echo "Stage 6: LM rescoring"
  oldlang=data/lang_nosp_test_bd_tgpr
  newlang=data/lang_nosp_test_bd_fgconst
  oldlm=$oldlang/G.fst
  newlm=$newlang/G.carpa
  oldlmcommand="fstproject --project_output=true $oldlm |"
  olddir=$dir/decode/$test_set/bd_tgpr
  newdir=$dir/decode/$test_set/rescore
  mkdir -p $newdir
  $train_cmd $dir/logs/rescorelm_$test_set.log \
	     lattice-lmrescore --lm-scale=-1.0 \
	     "ark:gunzip -c ${olddir}/lat.1.gz|" "$oldlmcommand" ark:- \| \
	     lattice-lmrescore-const-arpa --lm-scale=1.0 \
	     ark:- "$newlm" "ark,t:|gzip -c>$newdir/lat.1.gz"
fi

if [ ${stage} -le 7 ]; then
  echo "Stage 7: Computing WER"
  for lmtype in bd_tgpr rescore; do
    local/score_kaldi_wer.sh data/$test_set $treedir/graph_bd_tgpr $dir/decode/$test_set/$lmtype
    echo "Best WER for $dataset with $lmtype:"
    cat $dir/decode/$test_set/$lmtype/scoring_kaldi/best_wer
  done
fi

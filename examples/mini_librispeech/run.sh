#!/bin/bash
# Copyright (c) Yiwen Shao

# Apache 2.0

. ./cmd.sh
set -euo pipefail

# Change this location to somewhere where you want to put the data.
data=./corpus/

data_url=www.openslr.org/resources/31
lm_url=www.openslr.org/resources/11

# data related
rootdir=data
dumpdir=data/dump   # directory to dump full features
langdir=data/lang   # directory for language models
graphdir=data/graph # directory for chain graphs (FSTs)


# Data splits
train_set=train_clean_5
valid_set=dev_clean_2
test_set=dev_clean_2

# feature configuration
do_delta=false

# Model options
unit=phone # phone/char
type=mono # mono/bi

affix=
stage=0
. ./path.sh
. ./utils/parse_options.sh

dir=exp/tdnn_${type}${unit}${affix:+_$affix}
lang=$langdir/lang_${type}${unit}_e2e
graph=$graphdir/${type}${unit}


mkdir -p $data

for part in dev-clean-2 train-clean-5; do
  local/download_and_untar.sh $data $data_url $part
done

if [ $stage -le -1 ]; then
  local/download_lm.sh $lm_url $data data/local/lm
fi

if [ $stage -le 0 ]; then
  echo "Stage 0: Data Preparation"
  # format the data as Kaldi data directories
  for part in dev-clean-2 train-clean-5; do
    # use underscore-separated names in data directories.
    local/data_prep.sh $data/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
  done
fi

if [ $stage -le 1 ]; then
  echo "Stage 1: Feature Extraction"
  ./prepare_feat.sh --train_set $train_set \
		    --valid_set $valid_set \
		    --dumpdir $dumpdir \
		    --rootdir $rootdir
fi

if [ $stage -le 2 ]; then
  echo "Stage 2: Dictionary and LM Preparation"

  local/prepare_dict.sh --stage 3 --nj 30 --cmd "$train_cmd" \
    data/local/lm data/local/lm data/local/dict_nosp

  utils/prepare_lang.sh data/local/dict_nosp \
    "<UNK>" data/local/lang_tmp_nosp data/lang_nosp

  local/format_lms.sh --src-dir data/lang_nosp data/local/lm
  # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
  utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz \
    data/lang_nosp data/lang_nosp_test_tglarge
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
fi

num_targets=$(tree-info $graph/tree | grep num-pdfs | awk '{print $2}')

if [ ${stage} -le 5 ]; then
  echo "Stage 5: Model Training"
  opts=""
  mkdir -p $dir/logs
  log_file=$dir/logs/train.log
  python3 train.py
    --train data/train_${type}${unit}.json \
    --valid data/valid_${type}${unit}.json \
    --den-fst $graph/normalization.fst \
    --epochs 40 \
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
  log_file=$dir/logs/dump_$test_set.log
  result_file=$test_set/posteriors.ark
  mkdir -p $dir/$test_set
  python3 test.py \
	  --test data/valid_${type}${unit}.json \
	 --model model_best.pth.tar \
	 --results $result_file \
	 --exp $dir 2>&1 | tee $log_file
fi

if [ ${stage} -le 7 ]; then
  echo "Stage 7: Trigram LM Decoding"
  decode_dir=$dir/decode/$test_set/tgsmall
  mkdir -p $decode_dir
  latgen-faster-mapped --acoustic-scale=1.0 --beam=15 --lattice-beam=8 \
		       --word-symbol-table="$graph/graph_tgsmall/words.txt" \
		       $graph/0.trans_mdl $graph/graph_tgsmall/HCLG.fst \
		       ark:$dir/$test_set/posteriors.ark \
		       "ark:|lattice-scale --acoustic-scale=10.0 ark:- ark:- | gzip -c >$decode_dir/lat.1.gz" \
		       2>&1 | tee $dir/logs/decode_$test_set.log
fi


if [  $stage -le 8 ]; then
  echo "Stage 8: Forthgram LM rescoring"
  oldlang=$langdir/lang_nosp_test_tgsmall
  newlang=$langdir/lang_nosp_test_tglarge
  oldlm=$oldlang/G.fst
  newlm=$newlang/G.carpa
  oldlmcommand="fstproject --project_output=true $oldlm |"
  olddir=$dir/decode/$test_set/tgsmall
  newdir=$dir/decode/$test_set/tglarge
  mkdir -p $newdir
  $train_cmd $dir/logs/rescorelm_$test_set.log \
	     lattice-lmrescore --lm-scale=-1.0 \
	     "ark:gunzip -c ${olddir}/lat.1.gz|" "$oldlmcommand" ark:- \| \
	     lattice-lmrescore-const-arpa --lm-scale=1.0 \
	     ark:- "$newlm" "ark,t:|gzip -c>$newdir/lat.1.gz"
fi

if [ ${stage} -le 9 ]; then
  echo "Stage 9: Computing WER"
  for lmtype in tglarge; do
    local/score_kaldi_wer.sh $rootdir/$test_set $graph/graph_tgsmall $dir/decode/$test_set/$lmtype
    echo "Best WER for $test_set with $lmtype:"
    cat $dir/decode/$test_set/$lmtype/scoring_kaldi/best_wer
  done
fi

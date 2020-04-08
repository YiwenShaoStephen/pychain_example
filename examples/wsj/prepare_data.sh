#!/bin/bash
# Copyright (c) Yiwen Shao

# Apache 2.0

# data related
dumpdir=data/dump   # directory to dump full features
wsj0=
wsj1=
if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
  wsj0=/export/corpora5/LDC/LDC93S6B
  wsj1=/export/corpora5/LDC/LDC94S13B
fi

train_set=train_si284
valid_set=test_dev93
test_set=test_eval92
train_subset_size=200
stage=0

# feature configuration
do_delta=false

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh


if [ ${stage} -le 0 ]; then
  echo "$0: Stage 0: Data Preparation"
  local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.?
  local/wsj_prepare_dict.sh --dict-suffix "_nosp"
  utils/prepare_lang.sh data/local/dict_nosp \
                        "<SPOKEN_NOISE>" data/local/lang_tmp_nosp data/lang_nosp
  local/wsj_format_data.sh --lang-suffix "_nosp"
  echo "Done formatting the data."

  local/wsj_extend_dict.sh --dict-suffix "_nosp" $wsj1/13-32.1
  utils/prepare_lang.sh data/local/dict_nosp_larger \
                        "<SPOKEN_NOISE>" data/local/lang_tmp_nosp_larger \
                        data/lang_nosp_bd
  local/wsj_train_lms.sh --dict-suffix "_nosp"
  local/wsj_format_local_lms.sh --lang-suffix "_nosp"
  echo "Done exteding the dictionary and formatting LMs."
fi

train_feat_dir=${dumpdir}/${train_set}; mkdir -p ${train_feat_dir}
train_subset_feat_dir=${dumpdir}/${train_set}_${train_subset_size}; mkdir -p ${train_subset_feat_dir}
valid_feat_dir=${dumpdir}/${valid_set}; mkdir -p ${valid_feat_dir}
test_feat_dir=${dumpdir}/${test_set}; mkdir -p ${test_feat_dir}
if [ ${stage} -le 1 ]; then
  echo "$0: Stage 1: Feature Generation"
  echo "extracting MFCC features for the training data"
  for x in $train_set $valid_set $test_set; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 20 \
                       --mfcc-config conf/mfcc_hires.conf data/${x}
    # compute global CMVN
    compute-cmvn-stats scp:data/${x}/feats.scp data/${x}/cmvn.ark
  done

  dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
    data/${train_set}/feats.scp data/${train_set}/cmvn.ark ${train_feat_dir}/log ${train_feat_dir}
  dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
    data/${valid_set}/feats.scp data/${valid_set}/cmvn.ark ${valid_feat_dir}/log ${valid_feat_dir}
  dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
    data/${test_set}/feats.scp data/${test_set}/cmvn.ark ${test_feat_dir}/log ${test_feat_dir}

  # randomly select a subset of train set for optional diagnosis
  utils/subset_data_dir.sh data/${train_set} ${train_subset_size} data/${train_set}_${train_subset_size}
  utils/filter_scp.pl data/${train_set}_${train_subset_size}/utt2spk ${train_feat_dir}/feats.scp \
    > ${train_subset_feat_dir}/feats.scp
fi

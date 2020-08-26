#!/bin/bash
# Copyright (c) Yiwen Shao

# Apache 2.0

# data related
rootdir=data
dumpdir=data/dump   # directory to dump full features

train_set=train_clean_5
valid_set=dev_clean_2

train_subset_size=0
stage=0

# feature configuration
do_delta=false

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh


if [ ${stage} -le 0 ]; then
  echo "Extracting MFCC features"
  for x in $train_set $valid_set; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 20 \
                       --mfcc-config conf/mfcc_hires.conf $rootdir/${x}
    # compute global CMVN
    compute-cmvn-stats scp:$rootdir/${x}/feats.scp $rootdir/${x}/cmvn.ark
  done
fi

train_feat_dir=${dumpdir}/${train_set}; mkdir -p ${train_feat_dir}
valid_feat_dir=${dumpdir}/${valid_set}; mkdir -p ${valid_feat_dir}

if [ ${stage} -le 1 ]; then
  echo "Dumping Features with CMVN"
  dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
    $rootdir/${train_set}/feats.scp $rootdir/${train_set}/cmvn.ark ${train_feat_dir}/log ${train_feat_dir}
  dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
    $rootdir/${valid_set}/feats.scp $rootdir/${valid_set}/cmvn.ark ${valid_feat_dir}/log ${valid_feat_dir}

fi

# randomly select a subset of train set for optional diagnosis
if [ $train_subset_size -gt 0 ]; then
  train_subset_feat_dir=${dumpdir}/${train_set}_${train_subset_size}; mkdir -p ${train_subset_feat_dir}
  utils/subset_data_dir.sh $rootdir/${train_set} ${train_subset_size} $rootdir/${train_set}_${train_subset_size}
  utils/filter_scp.pl $rootdir/${train_set}_${train_subset_size}/utt2spk ${train_feat_dir}/feats.scp \
		      > ${train_subset_feat_dir}/feats.scp
fi

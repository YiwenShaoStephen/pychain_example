#!/bin/bash
# Copyright (c) Yiwen Shao

# Apache 2.0

# data related
langdir=data/lang
lmdir=data/local/nist_lm
unit=phone
stage=0
wsj1=

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

if [ -f $langdir/$unit/lang_$unit/L.fst ];then
  echo "$0: Dictionary and LM has already been prepared"
  exit 0
fi

if [ $stage -le 0 ]; then
  echo "$0: Dictionary and LM Preparation"
  [[ -f data/local/dict_phone/lexicon.txt ]] || local/wsj_prepare_dict.sh --dict-suffix "_phone"
  if [ $unit == "char" ]; then
    local/wsj_prepare_char_dict.sh --phone_dir data/local/dict_phone
  fi
  utils/prepare_lang.sh data/local/dict_${unit} \
			"<SPOKEN_NOISE>" data/local/lang_tmp_${unit} data/lang_${unit}
fi

if [ $stage -le 1 ]; then
  echo "$0: Preparing language models for test"
  for lm_suffix in bg tgpr tg bg_5k tgpr_5k tg_5k; do
    test=data/lang_${unit}_${lm_suffix}
    
    mkdir -p $test
    cp -r data/lang_${unit}/* $test || exit 1;
    
    gunzip -c $lmdir/lm_${lm_suffix}.arpa.gz | \
      arpa2fst --disambig-symbol=#0 \
               --read-symbol-table=$test/words.txt - $test/G.fst
    
    utils/validate_lang.pl --skip-determinization-check $test || exit 1;
  done
fi

if [ $stage -le 2 ]; then
  echo "$0: Extending Dictionary"
  if [ $unit == "phone" ]; then
    local/wsj_extend_dict.sh --dict-suffix "_phone" $wsj1/13-32.1
  else
    local/wsj_extend_char_dict.sh $wsj1/13-32.1 data/local/dict_char \
				  data/local/dict_char_larger
  fi
  
  utils/prepare_lang.sh data/local/dict_${unit}_larger \
			"<SPOKEN_NOISE>" data/local/lang_tmp_${unit}_larger \
			data/lang_${unit}_bd
fi

if [ $stage -le 3 ]; then
  echo "$0: Training Language Model"
  local/wsj_train_lms.sh --dict-suffix "_${unit}"
  local/wsj_format_local_lms.sh --lang-suffix "_${unit}"
fi

echo "Move all lang_* to $langdir/$unit"
mkdir -p $langdir/$unit
mv data/lang_* $langdir/$unit/
echo "Done exteding the dictionary and formatting LMs."

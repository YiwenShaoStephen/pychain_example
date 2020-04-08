#!/bin/bash
# Copyright (c) Yiwen Shao
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e -o pipefail

stage=0
train_set=train_si284
valid_set=test_dev93

lang=data/lang_e2e
treedir=data/graph

nj=10

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

if [ $stage -le 0 ]; then
  echo "$0: Stage 0: Phone LM estimating"
  rm -rf $lang
  cp -r data/lang_nosp $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo

  echo "Estimating a phone language model for the denominator graph..."
  mkdir -p $treedir/log
  $train_cmd $treedir/log/make_phone_lm.log \
             cat data/$train_set/text \| \
             steps/nnet3/chain/e2e/text_to_phones.py --between-silprob 0.1 \
             data/lang_nosp \| \
             utils/sym2int.pl -f 2- data/lang_nosp/phones.txt \| \
             chain-est-phone-lm --num-extra-lm-states=2000 \
             ark:- $treedir/phone_lm.fst
fi

echo "$0: Graph generation..."
if [ $stage -le 1 ]; then
  prepare_e2e.sh --nj $nj --cmd "$train_cmd" \
		 --shared-phones true \
		 data/$train_set $lang $treedir
  echo "Making denominator graph..."
  $train_cmd $treedir/log/make_den_fst.log \
	     chain-make-den-fst $treedir/tree $treedir/0.trans_mdl \
	     $treedir/phone_lm.fst \
	     $treedir/den.fst $treedir/normalization.fst
fi

if [ $stage -le 2 ]; then
  echo "Making HCLG full graph..."
  utils/lang/check_phones_compatible.sh \
    data/lang_nosp_test_bd_tgpr/phones.txt $lang/phones.txt
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_nosp_test_bd_tgpr \
    $treedir $treedir/graph_bd_tgpr || exit 1;
fi

if [ $stage -le 3 ]; then
  echo "Making numerator graph..."
  lex=$lang/L.fst
  oov_sym=`cat $lang/oov.int` || exit 1;
  for x in $train_set $valid_set; do
    sdata=data/$x/split$nj;
    [[ -d $sdata && $data/$x/feats.scp -ot $sdata ]] || split_data.sh data/$x $nj || exit 1;
    $train_cmd JOB=1:$nj $treedir/$x/log/compile_graphs.JOB.log \
    	       compile-train-graphs $scale_opts --read-disambig-syms=$lang/phones/disambig.int \
    	       $treedir/tree $treedir/0.mdl $lex \
    	       "ark:sym2int.pl --map-oov $oov_sym -f 2- $lang/words.txt < $sdata/JOB/text|" \
    	       "ark,scp:$treedir/$x/fst.JOB.ark,$treedir/$x/fst.JOB.scp" || exit 1;
    $train_cmd JOB=1:$nj $treedir/$x/log/make_num_fst.JOB.log \
    	       chain-make-num-fst-e2e $treedir/0.trans_mdl $treedir/normalization.fst \
    	       scp:$treedir/$x/fst.JOB.scp ark,scp:$treedir/$x/num.JOB.ark,$treedir/$x/num.JOB.scp
    for id in $(seq $nj); do cat $treedir/$x/num.$id.scp; done > $treedir/$x/num.scp
  done
fi

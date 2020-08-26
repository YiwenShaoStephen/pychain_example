MAIN_ROOT=$PWD/../..
KALDI_ROOT=$MAIN_ROOT/tools/kaldi

# BEGIN from kaldi path.sh
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sctk/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
# END

export PATH=$MAIN_ROOT:$MAIN_ROOT/tools:$PATH
export LD_LIBRARY_PATH=$MAIN_ROOT/tools/pychain/openfst/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$MAIN_ROOT:$MAIN_ROOT/tools:$MAIN_ROOT/tools/pychain:$PYTHONPATH
export PYTHONUNBUFFERED=1

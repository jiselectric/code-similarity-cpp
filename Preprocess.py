from Utils.Preprocess_utils import *
from transformers import *

def call_preprocess():
    ''' Set args. '''
    args = preprocess_set_args()

    ''' Set seed. '''
    set_seed(args.seed)

    ''' Make directories to store pre-processed data. '''
    preprocess_mkdir(args)

    ''' Create Tokenizer. '''
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    tokenizer.truncation_side = args.truncation_side

    ''' Pre-process data & returns data-frame. '''
    train_df = make_df(args)
    train_df_bm25 = make_pairs(train_df, tokenizer)


if __name__ == "__main__":
    call_preprocess()
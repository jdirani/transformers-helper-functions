import pandas as pd
import numpy as np
import os, pickle
import torch

def get_sentence_pdist(INPUT_TEXT, model, tokenizer):
    '''
    Get probabiliy distribution over the whole vocab at each word of the sentence INPUT_TEXT.

    This function is only compatible with maked models (e.g. BertForMaskedLM).

    e.g.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')


    Parameters
    -----------
    INPUT_TEXT : str
                Must include "<mask>" at the location of the queried word.
                e.g. 'He went to the <mask> to buy a load of bread.'

    tokenizer : transformers tokenizer object.

    model : transformers model object. Has to be a masked model.

    RETURNS
    -----------
    tokens : np.array
             Tokenized sentence, minus special tokens (e.g. [CLS], [EOS])

    pdist : np.array
            Probability distribution over tokenzier.vocab, for each token

    '''
    model.eval()

    # mask_token = tokenizer.mask_token # define the mask token for this specific model
    # INPUT_TEXT = INPUT_TEXT.replace('<mask>', mask_token) # replace the <mask> with corresponding model token mask

    input_ids = tokenizer.encode(INPUT_TEXT, return_tensors="pt")
    # mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

    with torch.no_grad(): # for faster computation
        token_logits = model(input_ids)[0] # get the logits of all the tokens

    pdist = torch.softmax(token_logits, 1).squeeze() # convert logit to pdist: pdist at the mask, one for each of the words in tokenizer.vocab
    tokens = np.array(tokenizer.convert_ids_to_tokens(input_ids[0]))

    # remove special tokens (e.g. CLS, EOS, etc..)
    mask_special_ids = [False if i in tokenizer.all_special_ids else True for i in input_ids[0]]
    pdist = pdist[mask_special_ids]
    tokens = tokens[mask_special_ids]

    return tokens, pdist

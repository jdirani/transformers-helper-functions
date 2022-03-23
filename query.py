import torch
import numpy as np
from transformers import *
from scipy.special import softmax
from scipy.stats import rankdata

# INPUT_TEXT = 'He could tell she was mad by the sound of her <mask>.'
# query = 'voice'
#
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForMaskedLM.from_pretrained('bert-base-uncased')


def masked_model_query(INPUT_TEXT, query, tokenizer, model, return_log_proba=True, print_top_pred=True):

    '''
    Queries the model for a specific word at a masked location. Returns the log probability and rank of the query word.
    This function is only compatible with maked models (e.g. BertForMaskedLM).

    e.g.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')


    Parameters
    -----------
    INPUT_TEXT : str
                Must include "<mask>" at the location of the queried word.
                e.g. 'He went to the <mask> to buy a load of bread.'

    query : str
            The word to query the model. Case sensitive.
            e.g. "bakery"

    tokenizer : transformers tokenizer object.

    model : transformers model object. Has to be a masked model.

    return_log_proba : bool, default = True
                       If false, return simple 0-1 probability

    print_top_pred : bool, default = True
                     Prints out the top prediction of the model.

    Returns
    -----------
    query_proba :  probability of the query word

    query_rank : rank of the query word
                 (e.g. 3 = 3rd most probable word according to the model)


    '''

    model.eval()

    mask_token = tokenizer.mask_token # define the mask token for this specific model
    INPUT_TEXT = INPUT_TEXT.replace('<mask>', mask_token) # replace the <mask> with corresponding model token mask

    input_ids = tokenizer.encode(INPUT_TEXT, return_tensors="pt")
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

    with torch.no_grad(): # for faster computation
        token_logits = model(input_ids)[0] # get the logits of all the tokens
    mask_token_logits = token_logits[0, mask_token_index, :] # subset to logits of mask token only
    probs = torch.softmax(mask_token_logits, 1).squeeze() # convert logit to probas: probas at the mask, one for each of the words in tokenizer.vocab
                                                          # squeeze to get rid of extra dim
    if model.config.architectures[0] == 'RobertaForMaskedLM':
        query = 'Ġ' + query # for some reason, roberta adds this symbol before words

    # get proba of query word
    query_index = tokenizer.convert_tokens_to_ids(query) # get index of query word (index in the vocab)
    query_proba = probs[query_index].item() # get proba by indexing using query_index
    # get rank of query word
    rank_probs = len(probs) - rankdata(probs).astype(int) # rank the probs form highest to lowest (highest=0, lowest=len(tokenizer.vocab_size)-1)
    query_rank = rank_probs[query_index] + 1 # get the rank of the query word

    # print top prediction from the model
    if print_top_pred == True:
        top_index = torch.argmax(probs).item()
        top_token = tokenizer.convert_ids_to_tokens([top_index])[0]
        top_proba = torch.max(probs).item()

        if model.config.architectures[0] == 'RobertaForMaskedLM':
            top_token = top_token[1:] # last brackets to remove 'Ġ'

        print('Top prediction: %s, prob=%2f, log_prob=%2f' %(top_token, top_proba, np.log(top_proba)))


    if return_log_proba == True:
        return np.log(query_proba), query_rank
    else:
        return query_proba, query_rank




def autoreg_model_query(INPUT_TEXT, query, tokenizer, model, return_log_proba=True, print_top_pred=True):

    '''
    Queries the model for a specific word at the end of the input sentence. Returns the log probability and rank of the query word.
    This function is only compatible with GPT and GPT2.

    e.g.
    # For GPT2:
    pretrained_weights = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # for GPT:
    pretrained_weights2 = 'openai-gpt'
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')


    Parameters
    -----------
    INPUT_TEXT : str
                Must NOT include a "<mask>". Query is by default at the end of the input sentence.
                e.g. 'He went to the bakery to buy a load of'

    query : str
            The word to query the model. Case sensitive.
            e.g. "bread"

    tokenizer : transformers tokenizer object

    model : transformers model object.

    return_log_proba : bool, default = True
                       If false, return simple 0-1 probability

    print_top_pred : bool, default = True
                     Prints out the top prediction of the model.

    Returns
    -----------
    query_proba :  probability of the query word

    query_rank : rank of the query word
                 (e.g. 3 = 3rd most probable word according to the model)


    '''

    model.eval()

    # if GPT2
    if model.config.architectures[0] == 'GPT2LMHeadModel':
        bos_token = tokenizer.special_tokens_map['bos_token']
        eos_token = tokenizer.special_tokens_map['eos_token']
        INPUT_TEXT = bos_token + ' ' + INPUT_TEXT + ' ' + eos_token
        input_ids = torch.tensor(tokenizer.encode(INPUT_TEXT))
        query = 'Ġ' + query # must add special character to query

    # if GPT
    elif model.config.architectures[0] == 'OpenAIGPTLMHeadModel':
        tokenized = tokenizer.tokenize(INPUT_TEXT) + ['<unk>'] # must add special tokens after tokenizing
        input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokenized))
        query = query + '</w>' # must add special character to query


    with torch.no_grad(): # for faster computation
        token_logits = model(input_ids)[0]

    mask_token_index = -2 # second to last token, last being the eos
    mask_token_logits = token_logits[mask_token_index]
    probs = torch.softmax(mask_token_logits, 0)

    # get proba of query word
    query_index = tokenizer.convert_tokens_to_ids(query) # get index of query word (index in the vocab)
    query_proba = probs[query_index].item() # get proba by i

    # get rank of query word
    rank_probs = len(probs) - rankdata(probs).astype(int) # rank the probs form highest to lowest (highest=0, lowest=len(tokenizer.vocab_size)-1)
    query_rank = rank_probs[query_index] + 1 # get the rank of the query word

    # print top prediction from the model
    if print_top_pred == True:
        top_index = torch.argmax(probs).item()
        top_token = tokenizer.convert_ids_to_tokens([top_index])[0]

        # remove special characters (gpt2:Ġ, gpt: </w>)
        if model.config.architectures[0] == 'GPT2LMHeadModel':
            top_token = top_token[1:]
        elif model.config.architectures[0] == 'OpenAIGPTLMHeadModel':
            top_token = top_token[:-4]

        top_proba = torch.max(probs).item()
        print('Top prediction: %s, prob=%2f, log_prob=%2f' %(top_token, top_proba, np.log(top_proba)))


    if return_log_proba == True:
        return np.log(query_proba), query_rank
    else:
        return query_proba, query_rank

from transformers import *
import pandas as pd
import numpy as np
import torch

def masked_model_predict(INPUT_TEXT, TOP_N, tokenizer, model):
    '''

    Get the top N predictions of the model for the LAST word of the sentence
    as well as their corresponding probabilities.
    This function is only compatible with maked models (e.g. BertForMaskedLM).

    Parameters
    -----------
    INPUT_TEXT : str
                Input sentence. Must include "<mask>".
                e.g. "She could tell he was mad by the sound of her <mask>."

    TOP_N : int
           Top N predictions to return.

    tokenizer : transformers tokenizer object

    model : transformers model object
           e.g. pretrained_weights = 'bert-base-uncased'
                tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
                model = BertForMaskedLM.from_pretrained(pretrained_weights)

    Returns
    -----------
    out_preds: list of tuples
               Top N predictions and their corresponding probabilities

    '''

    model.eval()  # Sets the module in evaluation mode.

    mask_token = tokenizer.mask_token
    INPUT_TEXT = INPUT_TEXT.replace('<mask>', mask_token) # replace the <mask> with corresponding model token mask

    input_ids = tokenizer.encode(INPUT_TEXT, return_tensors="pt")
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

    with torch.no_grad(): # for faster computation
        token_logits = model(input_ids)[0]
    mask_token_logits = token_logits[0, mask_token_index, :]
    probs = torch.softmax(mask_token_logits, 1)

    top_N_tokens_idx = torch.topk(mask_token_logits, TOP_N, dim=1).indices[0].tolist()
    top_N_tokens = [tokenizer.decode([token]).strip() for token in top_N_tokens_idx]
    top_N_probs  = probs[0, top_N_tokens_idx].tolist()

    out_preds = list(zip(top_N_tokens, top_N_probs))

    return out_preds



def masked_model_get_distribution(INPUT_TEXT, proba_at, TOP_N, tokenizer, model):
    '''
    Varies slightly from masked_model_predict(): no need to add a mask, add the
    word at which predictions ar required using proba_at.
    This was written to easily get the distribution accross the vocab for any word
    in the sentence, although it works exactly like masked_model_predict().

    This function is only compatible with maked models (e.g. BertForMaskedLM).

    Parameters
    -----------
    INPUT_TEXT : str
                Input sentence. Must NOT include "<mask>".
                e.g. "She could tell he was mad by the sound of her voice."

    proba_at : str
               Set which word to get the probas at.
               e.g. 'sound'

    TOP_N : int
           Top N predictions to return.

    tokenizer : transformers tokenizer object

    model : transformers model object
           e.g. pretrained_weights = 'bert-base-uncased'
                tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
                model = BertForMaskedLM.from_pretrained(pretrained_weights)

    Returns
    -----------
    top_N_probs

    top_N_tokens

    '''

    model.eval()  # Sets the module in evaluation mode.

    input_ids = tokenizer.encode(INPUT_TEXT, return_tensors="pt")
    query_token_index = torch.where(input_ids == tokenizer.encode(proba_at)[1])[1]

    with torch.no_grad(): # for faster computation
        token_logits = model(input_ids)[0]
    query_token_logits = token_logits[0, query_token_index, :]
    probs = torch.softmax(query_token_logits, 1)

    top_N_tokens_idx = torch.topk(query_token_logits, TOP_N, dim=1).indices[0].tolist()
    top_N_tokens = [tokenizer.decode([token]).strip() for token in top_N_tokens_idx]
    top_N_probs  = probs[0, top_N_tokens_idx].tolist() # Distribution

    return top_N_probs, top_N_tokens



def autoreg_model_predict(INPUT_TEXT, TOP_N, tokenizer, model, padding=True):

    '''
    Get the top N predictions of the model for the last word of the sentence
    as well as their corresponding probabilities.
    Use this function with causal/unidirectional language modeling (e.g. gpt, gpt2).

    Parameters
    -----------
    INPUT_TEXT : str
                Input sentence. No need to add a mask, predicts last (mising) word.
                e.g. 'I went to the bakery to buy some'

    TOP_N : int
           Top N predictions to return.

    tokenizer : transformers tokenizer object

    model : transformers model object
           e.g. pretrained_weights = 'gpt2'
                tokenizer = GPT2Tokenizer.from_pretrained(pretrained_weights)
                model = GPT2LMHeadModel.from_pretrained(pretrained_weights)

    padding : bool
              Add bos and eos padding.
                   - For GPT set to True
                   - For GPT2 set to False

    Returns
    -----------
    out_preds: list of tuples
               Top N predictions and their corresponding probabilities

    '''

    model.eval() #Sets the module in evaluation mode.

    # -------- First, get top 5 predictions
    input_ids = torch.tensor([tokenizer.encode(INPUT_TEXT)])

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        predictions = outputs[0]
        logits = outputs[0][0]

    # get the predicted next N sub-word
    idx_top_N_tokens = torch.topk(predictions[0, -1, :], TOP_N).indices.tolist()
    top_N_tokens = [tokenizer.decode([token]).strip() for token in idx_top_N_tokens]

    # -------- Second caclulate their probabilities
    probabilities = []
    if padding == True:
        bos = tokenizer.bos_token
        eos = tokenizer.eos_token
        sentences =  [bos + ' ' + INPUT_TEXT + top_N_tokens[i] + ' ' + eos for i in range(len(top_N_tokens))]
    else:
        sentences = [INPUT_TEXT + top_N_tokens[i] + ' ' + '<|endoftext|>' for i in range(len(top_N_tokens))]

    for sentence in sentences:
        input_ids = torch.tensor([tokenizer.encode(sentence)])

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs[0][0]
            probs = torch.softmax(logits,1)
            index = len(input_ids[0])-2
            token_id = input_ids[0][index]
            probability = probs[index-1][token_id].item()
            probabilities.append(probability)

    # -------- Format into list of tuples
    out_preds = list(zip(top_N_tokens, probabilities))
    return out_preds



def autoreg_model_get_distribution(INPUT_TEXT, proba_at, TOP_N, tokenizer, model, padding=True):

    '''
    Varies slightly from autoreg_model_predict()
    This was written to easily get the distribution accross the vocab for any word
    in the sentence, although it works exactly like autoreg_model_predict().

    Use this function with causal/unidirectional language modeling (e.g. gpt, gpt2).

    Parameters
    -----------
    INPUT_TEXT : str
                Input sentence. No mask, write full sentence.
                e.g. 'I went to the bakery to buy some bread.'

    proba_at : str
               Set which word to get the probas at.
               e.g. 'bakery'

    TOP_N : int
           Top N predictions to return.

    tokenizer : transformers tokenizer object

    model : transformers model object
           e.g. pretrained_weights = 'gpt2'
                tokenizer = GPT2Tokenizer.from_pretrained(pretrained_weights)
                model = GPT2LMHeadModel.from_pretrained(pretrained_weights)

    padding : bool
              Add bos and eos padding.
                   - For GPT set to True
                   - For GPT2 set to False

    Returns
    -----------
    probabilities

    top_N_tokens

    '''


    # --------- First, edit input by splitting it at the proba_at word. This is done because the model predicts the last word in the sentence.
    # Split the input into seperate words
    INPUT_TEXT = INPUT_TEXT.split(' ')
    # Edit to have the dots as a seperate word, result is ['dog', 'ate', 'bone', '.']
    INPUT_TEXT_clean = []
    for w in INPUT_TEXT:
        if '.' in w:
            w = w.strip('.')
            INPUT_TEXT_clean.append(w)
            INPUT_TEXT_clean.append('.')
        else:
            INPUT_TEXT_clean.append(w)
    INPUT_TEXT_clean = np.array(INPUT_TEXT_clean) # as numpy
    split_here = np.where(INPUT_TEXT_clean == proba_at)[0][0] # determine where to split the sentence
    # create cleaned input, overwrite INPUT_TEXT
    INPUT_TEXT = ' '.join(INPUT_TEXT_clean[:split_here])



    model.eval() #Sets the module in evaluation mode.

    # -------- Second, get top 5 predictions
    input_ids = torch.tensor([tokenizer.encode(INPUT_TEXT)])

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        predictions = outputs[0]
        logits = outputs[0][0]

    # get the predicted next N sub-word
    idx_top_N_tokens = torch.topk(predictions[0, -1, :], TOP_N).indices.tolist()
    top_N_tokens = [tokenizer.decode([token]).strip() for token in idx_top_N_tokens]

    # -------- Third caclulate their probabilities
    probabilities = []
    if padding == True:
        bos = tokenizer.bos_token
        eos = tokenizer.eos_token
        sentences =  [bos + ' ' + INPUT_TEXT + top_N_tokens[i] + ' ' + eos for i in range(len(top_N_tokens))]
    else:
        sentences = [INPUT_TEXT + top_N_tokens[i] + ' ' + '<|endoftext|>' for i in range(len(top_N_tokens))]

    for sentence in sentences:
        input_ids = torch.tensor([tokenizer.encode(sentence)])

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs[0][0]
            probs = torch.softmax(logits,1)
            index = len(input_ids[0])-2
            token_id = input_ids[0][index]
            probability = probs[index-1][token_id].item()
            probabilities.append(probability)


    return probabilities, top_N_tokens




def xlnet_model_predict(INPUT_TEXT, TOP_N, tokenizer, model, PADDING_TEXT=None):

    '''

    Get the top N predictions of the model for the last word of the sentence
    as well as their corresponding probabilities.
    Use this function with  permutation models (e.g. XLnet).
    XL models required random padding (PADDING_TEXT) for the permutation to work.

    Parameters
    -----------
    INPUT_TEXT : str
                Input sentence.

    PADDING_TEXT : str
                   Random padding text necessary for the function to work.
                   If None, uses the default random padding (xlnet_model_pred?? for details)

    TOP_N : int
           Top N predictions to return.

    tokenizer : transformers tokenizer object

    model : transformers model object


    Returns
    -----------
    out_preds: list of tuples
               Top N predictions and their corresponding probabilities

    '''

    # sets default padding text
    if PADDING_TEXT == None:
        PADDING_TEXT = '''In 1991, the remains of Russian Tsar Nicholas II and his family
        (except for Alexei and Maria) are discovered.
        The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
        remainder of the story. 1883 Western Siberia,
        a young Grigori Rasputin is asked by his father and a group of men to perform magic.
        Rasputin has a vision and denounces one of the men as a horse thief. Although his
        father initially slaps him for making such an accusation, Rasputin watches as the
        man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
        the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
        with people, even a bishop, begging for his blessing. <eod>'''

    model.eval()  # Sets the module in evaluation mode.

    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    mask_token = tokenizer.mask_token
    INPUT_TEXT = bos + ' ' + INPUT_TEXT + mask_token + ' ' + eos
    tokenize_input = tokenizer.tokenize(PADDING_TEXT + INPUT_TEXT)
    tokenize_text = tokenizer.tokenize(INPUT_TEXT)

    words = []
    lp = []
    for max_word_id in range((len(tokenize_input) - len(tokenize_text)), (len(tokenize_input))):
        sent = tokenize_input

        input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(sent)])

        perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
        perm_mask[:, :, max_word_id:] = 1.0

        target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)
        target_mapping[0, 0, max_word_id] = 1.0

        with torch.no_grad():
            outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
            next_token_logits = outputs[0]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]

        predicted_prob = torch.softmax(next_token_logits[0], 1)
        tokens_id = [torch.topk(next_token_logits[0], TOP_N, dim=1).indices[0].tolist()]
        words.append([tokenizer.decode([token]).strip() for token in tokens_id[0]])
        lp.append(predicted_prob[0][tokens_id[0]].tolist())

    out_preds = list(zip(words[-2], lp[-2]))

    return out_preds

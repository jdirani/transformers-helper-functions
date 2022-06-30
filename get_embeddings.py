import torch
from transformers import BertTokenizer, BertModel

# # Load pre-trained model tokenizer (vocabulary)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
# # Model: hidden_states must be set to True.
# model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
#
#
# text = "dog ate bone"


def get_CLS_embedding(text, tokenizer, model):
    '''
    text : str
           Input text. Single string (not list of strings) so tjat we can handle
           sentences of varying length.
           e.g. 'the dog is barking'

    tokenizer : transformers tokenizer
                e.g. BertTokenizer.from_pretrained('bert-base-uncased')

    model : transformers model, with hidden_states == True
            e.g. BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
    '''

    tokenized_text = tokenizer.encode(text, add_special_tokens=True)     # Add the special tokens and tokenize
    input_ids = torch.tensor([tokenized_text]) # add a dimension and convert to tensor

    model.eval()
    with torch.no_grad(): # minimizes computations, no need for backprop
        last_hidden_states = model(input_ids)
    CLS_features = last_hidden_states[0][0,0,:] # Here taking the first (and only) sentence, first token (the CLS token), all hidden outoputs (768)

    return CLS_features.numpy()



def get_words_embeddings(text, tokenizer, model, method='concatenate'):
    '''

    Get contextulized embeddings for each word in the text.
    Can use BertModel, not necessarily BertForMaskedLM.

    text : str
           Input text.

    tokenizer : transformers tokenizer
                e.g. BertTokenizer.from_pretrained('bert-base-uncased')

    model : transformers model, with hidden_states == True
            e.g. BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)

    method: str ('concatenate', 'sum', 'last')
            concatenate: concatenates last 4 hidden layers
            sum: sums last 4 hidden layers
            last: get last hidden layer


    '''

    # ============= Tokenize the text

    # Add the special tokens.
    marked_text = "[CLS] " + text + " [SEP]"
    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)
    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # # Display the words with their indeces.
    # for tup in zip(tokenized_text, indexed_tokens):
    #     print('{:<12} {:>6,}'.format(tup[0], tup[1]))
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])


    # ================ Get the hidden layers
    model.eval()

    with torch.no_grad(): # minimizes computations, no need for backprop
        outputs = model(tokens_tensor)
        hidden_states = outputs[2]

    # ================ Format layers
    # Get rid of the first dimension (the tuple)
    token_embeddings = torch.stack(hidden_states, dim=0)
    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # Swap dimensions 0 and 1: in order to be able to loop through tokens/words easily
    token_embeddings = token_embeddings.permute(1,0,2)

    # ========== get word vectors
    vectors = []

    for token in token_embeddings:  # For each token in the sentence
        if method == 'concatenate':
            vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
        elif method == 'sum':
            vec = torch.sum(token[-4:], dim=0)
        elif method == 'last':
            vec = token[-1]
        else:
            raise ValueError("%s: Invalid 'method' argument."%method)

        vectors.append(vec)

    return tokenized_text, vectors



def get_sentence_embeddings(text, tokenizer, model, method='average'):

    '''
        method:
            CLS: classification layer for use scikit-learn in classification tasks (logistic reg)
            average_last: average the last hidden layer of each token
            average_second: average the second to last hidden layer of each token
    '''

    # ============= Tokenize the text
    tokens_tensor = torch.tensor(tokenizer.encode(text)).unsqueeze(0)


    # ================ Get the hidden layers
    model.eval()

    with torch.no_grad(): # minimizes computations, no need for backprop
        outputs = model(tokens_tensor)
        hidden_states = outputs[2]

    # ================ Format layers
    # Get rid of the first dimension (the tuple)
    token_embeddings = torch.stack(hidden_states, dim=0)
    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # Swap dimensions 0 and 1: in order to be able to loop through tokens/words easily
    token_embeddings = token_embeddings.permute(1,0,2)

    # `token_embeddings` is a [N x 13 x 768] tensor.
                # nb of tokens, nb of layers, nb of features

    # ========== get sentence vectors
    vectors = []

    for token in token_embeddings:  # For each token in the sentence
        if method == 'average_second':
            # `token_vecs` is a tensor with shape [22 x 768]
            token_vecs = hidden_states[-2][0]
            # Calculate the average of all 22 token vectors.
            sentence_embedding = torch.mean(token_vecs, dim=0)
        elif method == 'average_last':
            # `token_vecs` is a tensor with shape [22 x 768]
            token_vecs = hidden_states[-1][0]
            # Calculate the average of all 22 token vectors.
            sentence_embedding = torch.mean(token_vecs, dim=0)
        elif method == 'CLS':
            last_hidden_states = token_embeddings[:,-1,:]
            cls_layer = last_hidden_states[0,:]
        else:
            raise ValueError("%s: Invalid 'method' argument."%method)

    return sentence_embedding

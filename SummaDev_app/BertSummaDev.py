import nltk
from nltk import sent_tokenize
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import transformers 
from transformers import BertForSequenceClassification, AutoModelForSequenceClassification, BertTokenizer, AdamW, BertConfig
nltk.download('punkt')

#BertSummaDev
def BertSummaDev(text):
    
    #load necessary objects to spit the text in sentences and tokenize on a word level
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    BERT_model = BertForSequenceClassification.from_pretrained('Greg1901/BertSummaDev_AFD')

    #convert to list of sentences
    sents = sent_tokenize(text)
    sentences = []
    for i in sents: 
        sentences.append(str(i))
    
    # Obtaining the longest sentence's length
    max_len = 0

    for sent in sentences:

        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)

        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))

    
    
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    
    # For every sentence...
    for sent in sentences:
    # `encode_plus` will:
    #  (1) Tokenize the sentence.
    #  (2) Prepend the `[CLS]` token to the start.
    #  (3) Append the `[SEP]` token to the end.
    #  (4) Map tokens to their IDs.
    #  (5) Pad or truncate the sentence to `max_length`
    #  (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_len,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                      )
    
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
    
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    batch_size = 32
    prediction_data = TensorDataset(input_ids, attention_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # Put model in evaluation mode
    BERT_model.eval()

    # Tracking variables 
    predictions = []

    # Predict 
    for batch in prediction_dataloader:

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch
  
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            result = BERT_model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask,
                        return_dict=True)

        logits = result.logits

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
  
        # Store predictions and true labels
        predictions.append(logits)

    # Combine the results across all batches. 
    flat_predictions = np.concatenate(predictions, axis=0)
    
    # Exctracting columns 1 which scores sentences for their probability of being selected
    summary_score_arr = flat_predictions[:, 1]
    #converting it to a list
    summary_score_list = summary_score_arr.tolist()
    
    # Defining a list with the indexes corresponding to the 2 higher scores 
    summary_index = [summary_score_list.index(x) for x in sorted(summary_score_list, reverse=True)[:2]]
    
    # Extracting the text correponding to these indexs
    #creating an empty list
    summary = []

    #looping in the summary indexes 
    for index in summary_index:
        #apending the sentences of theses indexes to the summary list
        summary.append(sentences[index])

    result = ' '.join(summary)

    return result


#Web dev
def main():
    #Setting the layout 
    st.title ('Welcome to SummaDev, a summarising interface for development papers.')
    st.subheader('This was built in the context of a master thesis.')
    text = st.text_area('Please enter your text below.')

    if st.button('Summarise'):
        result_summary = BertSummaDev(text)

        st.write(result_summary)

if __name__ == '__main__':
    main()





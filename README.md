# copy_generation

This repo is about generating copies by finetuning kogpt from kakaobrain

# Data

We gathered our data with some public advertisements, corporate advertisements, some slogans, and naver powerlinks etc...

# About train code

We used many hugginface modules to make the training step more efficient and easy.

### model and tokenizer

We used GPT2LMHeadModel from huggingface to load the pretrained kogpt_trinity model and PreTrainedTokenizerFast from huggingface to load the pretrained kogpt_trinity tokenizer. 
We added 6 additional tokens, category_token='\[CAT\]', company_token='\[COMP\]', brand_token='\[BRAND\]', name_token='\[NAME\]', keyword_token='\[KEY\]', copy_token='\[COPY\]'. If needed, few hundreds of tokens could be added.

### dataset and data collator

To train a advertisement copy, we thought the data format should look like '{input data} -> {output data}' and the data collator should be DataCollatorForLanguageModeling. 
We tried DataCollatorForSeq2Seq which looks like questioning-answering task dataset, '{input data}', '{output data}' in a different sentence and model receives only the input data and tries to generate the output data, 
but since the {input data} is too short, the information was too weak to create {output data} and had bad results. So, by using DataCollatorForLanguageModeling and the dataset seen before, we trained the model as training the language model.

### Trainer

Since the model argument size is about 15Gb and costs much more GPU memory on training, we tried to optimize the GPU memory as possible. Also, we had 4 A100 gpus to train, so running on multi-gpu environment efficiently was very important. So, we used **deepspeed** module, wich is supported in transformers Trainer module, which is faster and more efficient than using torch.distributed. By using deepspeed, batch size 8 per device and gradient_accumulation_steps 16 was affordable, so we trained our model on batch size 512(4*8*16).

### Details

As an augmentation, we randomly deleted the tokens in the dataset. Each tokens except [NAME] and [COPY] were added to the data with a probability of 0.8. Data argumentation was done once every epoch.

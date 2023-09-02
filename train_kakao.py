import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
import json
from datasets import Dataset
from transformers import TrainerCallback
import random

block_size = 64

tokenizer = AutoTokenizer.from_pretrained('kakao_dictionary')

# model = AutoModelForCausalLM.from_pretrained(
#   'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b').cuda()
model = AutoModelForCausalLM.from_pretrained(
  'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b',  # or float32 version: revision=KoGPT6B-ryan1.5b
  pad_token_id=tokenizer.eos_token_id,
  torch_dtype='auto'
)
print('model load')

# def gen_dataset(path):
#     with open(path, 'r') as f:
#         data = f.read()
#     json_data = json.loads(data)
#     output = []
#     for data in json_data:
#         output_dic = {}
# #         output_seq = ''
#         lst = ['[CAT]', '[COMP]', '[BRAND]', '[NAME]', '[KEY]', '[COPY]']
#         for i in lst:
#             output_dic[i] = data[i]
#         output.append(output_dic)
#     return output

def random_select(lst, prob_dic = {'[CAT]':0.8, '[COMP]':0.8, '[BRAND]':0.8, '[NAME]':1, '[KEY]':0.8, '[COPY]':1}):
    rslt = []
    for i in lst:
        seq = ''
        for key in ['[CAT]', '[COMP]', '[BRAND]', '[NAME]','[KEY]', '[COPY]']:
            if random.random() <= prob_dic[key] and i[key]:
                seq+=f'{key}{i[key]}'.strip()
            else:
                seq+=key
        rslt.append(f'[BOS]{seq}[EOS]')
    return rslt

def add_label(examples):
    examples['labels'] = examples['input_ids'].copy()
    return examples

def gen_label_encoded_dataset(lst, tokenizer):
    dataset = random_select(lst)
    dataset = Dataset.from_dict({'input_ids':dataset})
    encoded_dataset = dataset.map(lambda examples: tokenizer(examples["input_ids"], truncation=True, padding='max_length', max_length=64), batched=True)
    label_encoded_dataset = encoded_dataset.map(lambda examples: add_label(examples))
    print(f'generated dataset, size={len(label_encoded_dataset)}')
    return label_encoded_dataset

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length', max_length=64, return_tensors='pt')

data_path = 'final.json'

dataset = Dataset.from_json(data_path)
dataset = dataset.train_test_split(test_size=0.1)
training_args = TrainingArguments(
    resume_from_checkpoint=True,
    output_dir="./results_kakao_final",
    deepspeed='ds_config_zero3.json',
    num_train_epochs=20,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    logging_dir='./logs',
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=16,
    save_strategy='epoch',
    local_rank=-1
)

class RandomizeDatasetCallback(TrainerCallback):
    """
    Trigger re-computing subset for dataset Examples-proportional mixing, see `dataset::ProportionMixingDataset`

    A hack that modifies the train dataset, pointed by Trainer's dataloader
    """
    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        self.trainer.ori_train_dataset = self.trainer.train_dataset
        self.trainer.ori_eval_dataset = self.trainer.eval_dataset
        print('initialize dataset...')
        self.trainer.train_dataset = gen_label_encoded_dataset(self.trainer.ori_train_dataset, self.trainer.data_collator.tokenizer)
        self.trainer.eval_dataset = gen_label_encoded_dataset(self.trainer.ori_eval_dataset, self.trainer.data_collator.tokenizer)
    def on_epoch_begin(self, args: TrainingArguments, state, control, **kwargs):
        print('reinitializing dataset on epoch start...')
        self.trainer.train_dataset = gen_label_encoded_dataset(self.trainer.ori_train_dataset, self.trainer.data_collator.tokenizer)
        self.trainer.eval_dataset = gen_label_encoded_dataset(self.trainer.ori_eval_dataset, self.trainer.data_collator.tokenizer)
#     def on_evaluate(self, args: TrainingArguments, state, contril, **kwargs):
#         with torch.no_grad():
#             self.trainer.
#             self.trainer.model.generate(tokens, temperature=0.8, max_length=64)


class MyTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_callback(RandomizeDatasetCallback(trainer=self))
        
        
trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator
)
trainer.train()
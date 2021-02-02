from transformers import TrainingArguments, BertConfig, BertTokenizer, LineByLineTextDataset, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, pipeline
import math 


training_args = TrainingArguments(
    output_dir='temp_trainer',
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    evaluation_strategy='epoch',
    num_train_epochs=5,
    load_best_model_at_end=True,
)

tokenizer = BertTokenizer('data/self_vocab.txt', max_len=10)  #, return_tensors='pt'
config = BertConfig(vocab_size=len(tokenizer))
model = AutoModelForMaskedLM.from_config(config)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="data/ceshi.csv",
    block_size=10,
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= dataset,
    eval_dataset= dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,

)
train_result = trainer.train()
trainer.save_model()  # Saves the tokenizer too for easy upload
print("train_result:", train_result)

# Evaluation
eval_output = trainer.evaluate()
perplexity = math.exp(eval_output["eval_loss"])
print("perplexity", perplexity)

# inference
fill_mask = pipeline(
    "fill-mask",
    model= training_args.output_dir,
    tokenizer=tokenizer
)

inf_result = fill_mask("11 3 [MASK]")
print('----inf_result---:', inf_result)

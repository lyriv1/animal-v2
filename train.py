import os

import torch

from transformers import (
    TrainingArguments,
    LlavaNextProcessor, 
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig,
    AutoTokenizer, 
    AutoProcessor,
    Qwen2VLForConditionalGeneration
)

import evaluate

from qwen_vl_utils import process_vision_info

from peft import LoraConfig, get_peft_model

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_quantization_config,
    get_peft_config
)


from datasets import Dataset,DatasetDict,Image as Image_from_datasets, load_dataset, load_from_disk


from data_process_utils import data_process
from config import dataset_config, model_config as mc
#----------------------数据加载与处理----------------------

# train_res_json=data_process.img_label_process(dataset_config.train_data_path,dataset_config.edit_uuid_flag,'train')
# test_res_json=data_process.img_label_process(dataset_config.test_data_path,dataset_config.edit_uuid_flag,'test')


# data_process.edit_json(train_res_json,dataset_config.train_json_path,'train')
# data_process.edit_json(test_res_json,dataset_config.test_json_path,'test')

# train_dataset = Dataset.from_json("./temp/train.json")
# test_dataset = Dataset.from_json("./temp/test.json")

# dataset = DatasetDict({
#     'train': train_dataset,
#     'test': test_dataset
# })


# dataset = dataset.cast_column("images", Image_from_datasets())
# # os.remove("data.hf")


# dataset.save_to_disk("./data.hf")

# 加载数据集
dataset = load_from_disk('./data.hf')


metric = evaluate.load("accuracy")

#------------------------------------------超参数配置-----------------------------------------------




parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig)) #命令行中读取配置
script_args, training_args, model_config = parser.parse_args_and_config()
training_args.gradient_checkpointing_kwargs = dict(use_reentrant = False)
training_args.remove_unused_columns = False
training_args.dataset_kwargs = {"skip_prepare_dataset": True}


training_args.gradient_checkpointing = True  
training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
# training_args.per_device_train_batch_size = 1  # 减小批次大小
# training_args.per_device_eval_batch_size = 1
# training_args.gradient_accumulation_steps = 8  # 增加梯度累积步数
# training_args.bf16 = True
# # training_args.fp16 = True  # 使用混合精度训练
# training_args.optim = "adamw_8bit"  # 使用8-bit优化器
# training_args.remove_unused_columns = False
training_args.dataset_kwargs = {"skip_prepare_dataset": True}
# training_args.logging_steps = 10
# training_args.save_steps = 100
training_args.evaluation_strategy = "epoch"
# training_args.eval_steps = 100
training_args.evaluation_strategy = 'no'  # 限制训练步数


# 量化配置
torch_dtype = (
    model_config.torch_dtype
    if model_config.torch_dtype in ["auto", None]
    else getattr(torch, model_config.torch_dtype)
)

quantization_config = get_quantization_config(model_config)

model_kwargs = dict(
    revision = model_config.model_revision,
    attn_implementation = model_config.attn_implementation,
    torch_dtype = torch_dtype,
    device_map = get_kbit_device_map() if quantization_config is not None else None,
    quantization_config = quantization_config,
)


#------------------------------------------模型加载-----------------------------------------------

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


processor = AutoProcessor.from_pretrained(
    mc.model_name,
    trust_remote_code = model_config.trust_remote_code
    )

processor.image_processor.size = {"height": 224, "width": 224}

model = Qwen2VLForConditionalGeneration.from_pretrained(
    mc.model_name, 
    trust_remote_code = model_config.trust_remote_code, 
    **model_kwargs
)

# model.print_trainable_parameters()


#------------------------------------------数据处理-----------------------------------------------

def collate_fn(examples):


    texts = [processor.apply_chat_template(example["messages"], tokenize = False) for example in examples]

    texts = [i.replace('<\s>','</s>') for i in texts]

    images = [example["images"] for example in examples]
        
    batch = processor(text = texts, images = images, return_tensors = "pt", padding = True)

    labels = batch["input_ids"].clone()

    labels[labels  ==  processor.tokenizer.pad_token_id] = -100  #

    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)

    labels[labels  ==  image_token_id] = -100

    batch["labels"] = labels

    return batch


#------------------------------------------评估函数-----------------------------------------------


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


#------------------------------------------开始训练-----------------------------------------------

trainer = SFTTrainer(
    model = model,
    args = training_args,
    data_collator = collate_fn,
    train_dataset = dataset['train'],
    eval_dataset = dataset['test'],
    processing_class = processor.tokenizer,
    compute_metrics=compute_metrics,
    # peft_config = lora_config
)


# 训练
train_result = trainer.train()

# 保存最终模型
trainer.save_model(training_args.output_dir)

# 保存训练指标
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

# 保存训练状态
trainer.save_state()


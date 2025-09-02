# import torch
# print(torch.__version__)

# print(torch.version.cuda)
# print(torch.backends.cudnn.version())

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())

# import os

# if not os.path.exists(os.path.dirname("./tem/e.json")):
#     os.makedirs(os.path.dirname("./tem/e.json"))

# print("./tem/e.json".split('/')[:-1])


import os
import random
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel, LoraConfig, TaskType


from data_process_utils import data_process
from config import dataset_config, model_config as mc



config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True,
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    mc.model_name, torch_dtype="auto", device_map="auto"
)
model = PeftModel.from_pretrained(model, model_id=mc.output_dir, config=config)
processor = AutoProcessor.from_pretrained(mc.model_name)

if not os.path.exists("./test-imgs"):
    os.makedirs("./test-imgs")
    print("成功创建目录：", "./test-imgs")
    raise ValueError("请先将测试图片放入 test-imgs 文件夹内后重启测试")
else:
    test_img = random.choice(os.listdir("./test-imgs"))

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": os.path.join("./test-imgs",test_img),
            },
            {"type": "text", "text": "this is a photo of"},
        ],
    }
]


text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")


generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("图片为：", test_img)
print(output_text)
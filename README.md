# 动物分类微调模型

## 技术栈

|          | 选择                                                         |
| -------- | ------------------------------------------------------------ |
| 模型：   | [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) |
| 数据集： | [图像分类·动物图像识别与分类](https://aistudio.baidu.com/datasetdetail/140388/0) |

## 文件结构

```shell
.                           # 原始模型
├── input                           # 数据集
├── test-img					  # 推理测试数据集
├── scripts                         # deepspeed配置
│   ├── deepspeed_zero3.yaml
│   └── zero3.json
├── temp                            # 数据集加载缓存
│   ├── validation.json
│   └── train.json
├── REDAME.md
├── config.py                       # 配置项
├── data_process_utils.py           # 数据集加载脚本
├── test.py                         # 测试脚本
├── train.py                        # 训练脚本
├── train.sh                        # 训练命令
```

## 数据集

- **训练集**: 14308张图片
- **测试集**: 3495张图片
- **动物类别**: 蝴蝶、猫、鸡、牛、狗、大象、马、蜘蛛、羊、松鼠

测试集无标签，可选择使用别的模型辅助打标签或从训练集剪切一部分用于测试，本项目选二

### 数据集结构

```shell
├── input
│   ├── train_data                  # 训练集
│   │   ├── butterfly
│   │   ├── cat
│   │   ├── chicken
│   │   ├── cow
│   │   ├── dog
│   │   ├── elephant
│   │   ├── horse
│   │   ├── ragno
│   │   ├── sheep
│   │   └── squirrel
│   └── test_data             # 验证集
│       ├── butterfly
│       ├── cat
│       ├── chicken
│       ├── cow
│       ├── dog
│       ├── elephant
│       ├── horse
│       ├── ragno
│       ├── sheep
│       └── squirrel
```

## 模型

[Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)

参考资料：
[多模态大模型应用实践（一）- 利用微调 LLaVA 实现高效酒店图片分类](https://aws.amazon.com/cn/blogs/china/multimodal-large-model-application-practice-part-one/)
[动物分类深度学习项目](https://github.com/WorthStudy/animal-classification/tree/main)
[Qwen2-VL多模态大模型微调实战](https://zhuanlan.zhihu.com/p/7144893529)

### 训练参数

```shell
accelerate launch train.py \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --output_dir lora-Qwen2-VL-2b \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --num_train_epochs 2 \
    --save_strategy "steps" \
    --save_steps 3000 \
    --learning_rate 1e-4 \
    --save_total_limit 3 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --per_device_eval_batch_size 1 \
    --logging_steps 10 \
    --logging_dir "./logs" \
    --report_to "tensorboard" 
```

## 训练过程

[Qwen2-VL-2b-Animal-Classification](https://swanlab.cn/@Lyriv/Qwen2-VL-2b-Animal-Classification/runs/t7qjdrhahszxbw0fy5i2j/chart)

## 如何开始微调

1. **安装依赖**: `pip install -r requirements.txt`
2. **下载训练集**：前往[图像分类·动物图像识别与分类](https://aistudio.baidu.com/datasetdetail/140388/0)，并按上文给出的结构进行整理
3. **执行训练**：`sh train.sh`
4. **推理测试**：`python -u test.py`
5. **查看训练过程**：`tensorboard --logdir=./logs`

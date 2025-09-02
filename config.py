
class dataset_config:
    edit_uuid_flag = False
    train_data_path = "./input/train_data"
    test_data_path = "./input/test_data"
    train_json_path = "./temp/train.json"
    test_json_path = "./temp/test.json"



class model_config:
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    output_dir = "lora-Qwen2-VL-2b"

    # peft_model_path = "./lora-16"
import pandas as pd
from tqdm import tqdm
import json
import os
from PIL import Image
from datasets import Dataset,DatasetDict,Image as Image_from_datasets, load_dataset, load_from_disk
import uuid
from transformers import  TrainingArguments
# from trl import SFTTrainer
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


def get_single_res(type,idx,img_path,data_type):
        res = {}


        image = Image.open(img_path)
        res['id'] = str(idx)
        res['image'] = img_path
        # if data_type == 'test':
        #     res["conversations"] = [{'from': 'human',
        #                             'value': 'Fill in the blank: this is a photo of a {}'}]

        # else:
        res["conversations"] = [{'from': 'human',
                                'value': '<image>\nFill in the blank: this is a photo of a {}, you should select the appropriate categories from the provided labels: [butterfly, cat, chicken, cow, dog, elephant, horse, ragno, sheep, squirrel]'},
                                {'from': 'gpt',
                                'value': 'this is a photo of {}.'.format(type)}]



        return res


class data_process:
    
    
    def img_label_process(path,edit_img_name_flag,data_type):
        # count = 0
        # for _, _, files in os.walk(path):
        #     count += len(files)
        
        res_json = []
        for root, ds, fs in os.walk(path,topdown=False):
            
            # waring linux中需要改这行，改为Linux的路径格式，即/
            animal_type=root.split('/')[-1]

            
            for f in fs:
                i=uuid.uuid4()
                full_name = os.path.join(root, f)
                if edit_img_name_flag:
                    img_type=f.split('.')[1]
                    
                    new_name=os.path.join(root,str(i)+'.'+img_type)
                    
                    # print(new_name)
                    if full_name!=new_name:
                        os.rename(full_name, new_name)
                        full_name=os.path.join(root,str(i)+'.'+img_type)
                # if data_type=="test":
                #     animal_type=""
                res = get_single_res(animal_type,str(i),full_name,data_type)
                if res!={}:
                    res_json.append(res)
                

        return res_json




    def edit_json(data,output_json,data_type):
        output_json_dir = os.path.dirname(output_json)
        if not os.path.exists(output_json_dir):
            os.makedirs(output_json_dir)
            print("成功创建目录：", output_json_dir)
        # if data_type == 'test':
        #     for i in data:
        #         i['images'] = i.pop('image')
        #         i['messages'] = []
        #         i['messages'].append({'content': [{'index': None, 'text': i['conversations'][0]['value'].replace('<image>\n','')+'\n', 'type': 'text'},
        #         {'index': 0, 'text': None, 'type': 'image'}],
        #         'role': 'user'})
        # else:
        for i in data:
            i['images'] = i.pop('image')
            i['messages'] = []
            i['messages'].append({'content': [{'index': None, 'text': i['conversations'][0]['value'].replace('<image>\n','')+'\n', 'type': 'text'},
            {'index': 0, 'text': None, 'type': 'image',"resized_height": 280,"resized_width": 280}],
            'role': 'user'})
            i['messages'].append({'content': [{'index': None, 'text': i['conversations'][1]['value'], 'type': 'text'}],
            'role': 'assistant'})

        with open(output_json, 'w') as file:
            json.dump(data, file, indent=4)
        return output_json

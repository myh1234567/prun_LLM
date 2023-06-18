import shutil
from transformers import AutoTokenizer
import json
import os

# 指定模型的路径
model_path = "bloom-560m"

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

def contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False

# 读取tokenizer.json文件
with open(f'{model_path}/tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_data = json.load(f)

# 筛选出包含中文字符的词汇，并重排ID
old_vocab = tokenizer_data['model']['vocab']
new_vocab = {}
new_id = 0
for word, id_ in old_vocab.items():
    if contains_chinese(tokenizer.decode(id_)):
        new_vocab[word] = new_id
        new_id += 1
print(new_vocab)
new_vocab_size = len(new_vocab)

# 筛选出包含中文字符的合并规则
new_merges = [merge for merge in tokenizer_data['model']['merges'] if contains_chinese(merge.split()[0]) or contains_chinese(merge.split()[1])]

# 更新tokenizer.json文件
tokenizer_data['model']['vocab'] = new_vocab
tokenizer_data['model']['vocab_size'] = new_vocab_size
tokenizer_data['model']['merges'] = new_merges

# 保存新的tokenizer.json
new_model_path = "bloom-myh"
os.makedirs(new_model_path, exist_ok=True)
with open(f'{new_model_path}/tokenizer.json', 'w', encoding='utf-8') as f:
    json.dump(tokenizer_data, f, ensure_ascii=False, indent=4)

# 遍历原始模型文件
for filename in os.listdir(model_path):
    file_path = os.path.join(model_path, filename)
    
    # 只复制非目录文件并排除tokenizer.json
    if os.path.isfile(file_path) and filename != 'tokenizer.json' and filename != 'pytorch_model.bin':
        shutil.copy(file_path, os.path.join(new_model_path, filename))
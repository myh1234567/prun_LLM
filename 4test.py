from pruners.vocabulary_pruner import BloomVocabularyPruner
import deepspeed

# 需要进行裁剪的模型路径
model_name_or_path = 'bloom-560m'
# 自己制作的词表的路
new_tokenizer_name_or_path = 'bloom-myh'
save_path = 'path-to-save'
pruner = BloomVocabularyPruner()
# 裁剪
# pruner.prune(model_name_or_path, new_tokenizer_name_or_path, save_path)


# 检查裁剪的模型与原模型是否一致
pruner.check(model_name_or_path, save_path, text='这些对方过后就哭')
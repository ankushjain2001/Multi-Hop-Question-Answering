# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import string
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from collections import Counter
from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import ByteLevelBPETokenizer
from transformers import LongformerTokenizerFast, TFLongformerModel, LongformerConfig



# Training Parameters
MAX_LEN = int(sys.argv[1])
BATCH_SIZE = int(sys.argv[2])
EPOCHS = int(sys.argv[3])
LR = float(sys.argv[4])
MODEL_NAME = str(sys.argv[5])
WEIGHT_NAME = str(sys.argv[6])

# Sample values
# MAX_LEN = 1024
# BATCH_SIZE = 8
# EPOCHS = 2
# LR = 3e-5
# MODEL_NAME = 'allenai/longformer-base-4096/'
# WEIGHT_NAME = 'weights' (.h5)



# Configs for model training - CHANGE ME
configuration = LongformerConfig()
save_path = "/scratch/aj2885/results/longformer-base-4096_"+WEIGHT_NAME[:-1]+"/"
data_path = "/scratch/aj2885/datasets/hotpotqa/"


# Tokenizer
slow_tokenizer = LongformerTokenizerFast.from_pretrained(MODEL_NAME)
if not os.path.exists(save_path):
    os.makedirs(save_path)
slow_tokenizer.save_pretrained(save_path)

# Load the fast tokenizer from saved file
tokenizer = ByteLevelBPETokenizer(save_path+"vocab.json",save_path+"merges.txt" ,lowercase=True)



# Import Data
with open(data_path+'train_em.json') as f:
    train_data = json.load(f)
with open(data_path+'val_em.json') as f:
    test_data = json.load(f)
    
    
    
# Dataset class with input tokens, attention mask and stant & end tokens
class HotpotQAExample:
    def __init__(self, question, context, start_char_idx, answer_text, all_answers):
        self.question = question
        self.context = context
        self.start_char_idx = start_char_idx
        self.answer_text = answer_text
        self.all_answers = all_answers
        self.skip = False

    def preprocess(self):
        context = self.context
        question = self.question
        answer_text = self.answer_text
        start_char_idx = self.start_char_idx

        # Clean context, answer and question
        context = " ".join(str(context).split())
        question = " ".join(str(question).split())
        answer = " ".join(str(answer_text).split())

        # Find end character index of answer in context
        end_char_idx = start_char_idx + len(answer)
        if end_char_idx >= len(context):
            self.skip = True
            return

        # Mark the character indexes in context that are in answer
        is_char_in_ans = [0] * len(context)
        for idx in range(start_char_idx, end_char_idx):
            is_char_in_ans[idx] = 1

        # Tokenize context
        tokenized_context = tokenizer.encode(context)

        # Find tokens that were created from answer characters
        ans_token_idx = []
        for idx, (start, end) in enumerate(tokenized_context.offsets):
            if sum(is_char_in_ans[start:end]) > 0:
                ans_token_idx.append(idx)

        if len(ans_token_idx) == 0:
            self.skip = True
            return

        # Find start and end token index for tokens from answer
        start_token_idx = ans_token_idx[0]
        end_token_idx = ans_token_idx[-1]

        # Tokenize question
        tokenized_question = tokenizer.encode(question)

        # Create inputs
        input_ids = tokenized_context.ids + tokenized_question.ids[1:]
        token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(
            tokenized_question.ids[1:]
        )
        attention_mask = [1] * len(input_ids)

        # Pad and create attention masks.
        # Skip if truncation is needed
        padding_length = MAX_LEN - len(input_ids)
        if padding_length > 0:  # pad
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        elif padding_length < 0:  # skip
            self.skip = True
            return

        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.context_token_to_char = tokenized_context.offsets



# Create examples
def create_hotpotqa_examples(raw_data):
    hotpotqa_examples = []
    for i in tqdm(range(len(raw_data))):
      item = raw_data[i]
      question = item["query"]
      answer_text = item["answer"]
      all_answers = None
      context = " ".join(item["supports"]).lower()
      start_char_idx = context.find(answer_text)
      hotpotqa_eg = HotpotQAExample(
          question, context, start_char_idx, answer_text, all_answers
      )
      hotpotqa_eg.preprocess()
      hotpotqa_examples.append(hotpotqa_eg)
    return hotpotqa_examples

# Create input targets
def create_inputs_targets(hotpotqa_examples):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
    }
    for item in hotpotqa_examples:
        if item.skip == False:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["attention_mask"],
    ]
    y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
    return x, y



# Create train set
train_hotpotqa_examples = create_hotpotqa_examples(train_data)
x_train, y_train = create_inputs_targets(train_hotpotqa_examples)
print(f"{len(train_hotpotqa_examples)} training points created.")



# Create validation set
eval_hotpotqa_examples = create_hotpotqa_examples(test_data)
x_eval, y_eval = create_inputs_targets(eval_hotpotqa_examples)
print(f"{len(eval_hotpotqa_examples)} evaluation points created.")



# Create model
def create_model():
    # Longformer encoder
    encoder = TFLongformerModel.from_pretrained('weights.h5')

    # QA Model - Reproducing HuggingFace like QA model architecture
    input_ids = layers.Input(shape=(MAX_LEN,), dtype=tf.int32)
    #token_type_ids = layers.Input(shape=(MAX_LEN,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(MAX_LEN,), dtype=tf.int32)
    embedding = encoder(
        input_ids, attention_mask=attention_mask
    )[0]

    start_logits = layers.Dense(1, name="start_logit", use_bias=False)(embedding)
    start_logits = layers.Flatten()(start_logits)

    end_logits = layers.Dense(1, name="end_logit", use_bias=False)(embedding)
    end_logits = layers.Flatten()(end_logits)

    start_probs = layers.Activation(keras.activations.softmax)(start_logits)
    end_probs = layers.Activation(keras.activations.softmax)(end_logits)

    model = keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=[start_probs, end_probs],
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = keras.optimizers.Adam(lr=LR)
    model.compile(optimizer=optimizer, loss=[loss, loss])
    return model



# Use TPU
use_tpu = False
if use_tpu:
    # Create distribution strategy
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    # Create model
    with strategy.scope():
        model = create_model()
else:
    model = create_model()
# Print model architecture summary
model.summary()



# Text normalizer for exact match
def normalize_text(text):
    # Remove the special seperator symbol 'Ġ'
    text = ' '.join(text.split('Ġ'))
    # Lowercase
    text = text.lower()
    # Remove punctuations
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)
    # Remove articles
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    text = re.sub(regex, " ", text)
    # Remove extra white space
    text = " ".join(text.split())
    return text


   
# Exact Match score
def exact_match_score(prediction, ground_truth):
    return (normalize_text(prediction) == normalize_text(ground_truth))



# F1, Precision & Recall
def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_text(prediction)
    normalized_ground_truth = normalize_text(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall



class ExactMatch(keras.callbacks.Callback):
    def __init__(self, x_eval, y_eval):
        self.x_eval = x_eval
        self.y_eval = y_eval

    def on_epoch_end(self, epoch, logs=None):
        pred_start, pred_end = self.model.predict(self.x_eval)
        em_count = 0
        f1_count = 0
        precision_count = 0
        recall_count = 0
        eval_examples_no_skip = [_ for _ in eval_hotpotqa_examples if _.skip == False]
        for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
            hotpotqa_eg = eval_examples_no_skip[idx]
            offsets = hotpotqa_eg.context_token_to_char
            start = np.argmax(start)
            end = np.argmax(end)
            if start >= len(offsets):
                continue
            pred_char_start = offsets[start][0]
            if end < len(offsets):
                pred_char_end = offsets[end][1]
                pred_ans = hotpotqa_eg.context[pred_char_start:pred_char_end]
            else:
                pred_ans = hotpotqa_eg.context[pred_char_start:]

            true_ans = hotpotqa_eg.answer_text
            
            # EM Score
            em_count += int(exact_match_score(pred_ans, true_ans))
            # FPR score
            f, p, r = f1_score(pred_ans, true_ans)
            f1_count += f
            precision_count += p
            recall_count += r
        
        # Print scores
        acc = em_count / len(self.y_eval[0])
        f1 = f1_count / len(self.y_eval[0])
        prec = precision_count / len(self.y_eval[0])
        rec = recall_count / len(self.y_eval[0])
        print(f"\nEpoch={epoch+1}, Exact Match Score={acc:.4f}, F1 Score={f1:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")

        # Save model checkpoint
        try:
            model.save_weights(save_path+'weights_'+WEIGHT_NAME+str(epoch+1)+".h5")
        except:
            print('Error occurred in saving model')
            pass



# Begin training
exact_match_callback = ExactMatch(x_eval, y_eval)
model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,  # For demonstration, 3 epochs are recommended
    verbose=1,
    batch_size=BATCH_SIZE,
    callbacks=[exact_match_callback],
)



# Save trained weights
model.save_weights(save_path+'weights_'+WEIGHT_NAME[:-1]+".h5")

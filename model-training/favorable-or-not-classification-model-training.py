import warnings
import io
import os
import torch
from tqdm.notebook import tqdm
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import pandas as pd
from sklearn.model_selection import train_test_split
from ml_things import plot_dict, plot_confusion_matrix, fix_text
from sklearn.metrics import classification_report, accuracy_score
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          AdamW,
                          get_linear_schedule_with_warmup
                          )
import random

warnings.filterwarnings("ignore")

set_seed(123)
epochs = 4
batch_size = 16
max_length = 256
data_path = 'input_data\\train_data\\train_data_json'
labels_ids = {'01.유리':0, '02.불리':1}
n_labels = len(labels_ids)

import json

# Json 파일에서 특정 키가 있는지 확인하는 함수 정의
def is_json_key_present(json, key):
    try:
        buf = json[key]
    except KeyError:
        return False
    return True


# Json에서 데이터를 가져오는 클래스
class JsonDataset(Dataset):
  def __init__(self, path, use_tokenizer): # 생성자 (초기화 함수)
    if not os.path.isdir(path):
      raise ValueError('Invalid `path` variable! Needs to be a directory')

    self.texts = []
    self.labels = []

    # 라벨에 따른 패스 설정 & 데이터 불러오기
    for label in ['01.유리', '02.불리']:
      sentiment_path = os.path.join(path, label)
      files_names = os.listdir(sentiment_path)

      for file_name in tqdm(files_names, desc=f'{label} files'):
        file_path = os.path.join(sentiment_path, file_name)

        with open(file_path, "r") as json_file:
          json_data  =  json.load(json_file)

        if(is_json_key_present(json_data,"clauseArticle")) :
          for data in json_data['clauseArticle'] :
            self.texts.append(fix_text(data))
            self.labels.append(label)
        else :
          print(file_path)

    self.n_examples = len(self.labels)
    return


  def __len__(self):
    return self.n_examples


  def __getitem__(self, item):
    return {'text':self.texts[item],
            'label':self.labels[item]}

# 데이터 전처리 클래스
class ClassificationCollator(object):

    # 토크나이저, 라벨 인코더, 최대 시퀀스 길이 설정 & 객체 초기화
    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):
        self.use_tokenizer = use_tokenizer
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        self.labels_encoder = labels_encoder
        return

    # 설정된 토크나이저, 라벨 인코더, 최대 시퀀스 길이를 이용해 데이터를 전처리
    def __call__(self, sequences):
        texts = [sequence['text'] for sequence in sequences]
        labels = [sequence['label'] for sequence in sequences]
        labels = [self.labels_encoder[label] for label in labels]

        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
        inputs.update({'labels':torch.tensor(labels)})

        return inputs

# 학습 진행 함수 (파이토치 사용 학습 -> 예측 결과 반환)
def train(dataloader, optimizer_, scheduler_):
    global model  # 전역 변수 model & 학습시킴

    predictions_labels = []
    true_labels = []
    total_loss = 0
    model.train()

    for batch in tqdm(dataloader, total=len(dataloader)):
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k: v.type(torch.long) for k, v in batch.items()}
        optimizer_.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_.step()
        scheduler_.step()
        logits = logits.detach().cpu().numpy()
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()

    avg_epoch_loss = total_loss / len(dataloader)

    # 실제 라벨, 예측 라벨, 평균 손실 반환
    return true_labels, predictions_labels, avg_epoch_loss

# 검증 데이터에 대한 예측 & 손실 평가
def validation(dataloader):
    global model

    predictions_labels = []
    true_labels = []
    total_loss = 0
    model.eval()

    for batch in tqdm(dataloader, total=len(dataloader)):
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k: v.type(torch.long) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss, logits = outputs[:2]
            logits = logits.detach().cpu().numpy()
            total_loss += loss.item()
            predict_content = logits.argmax(axis=-1).flatten().tolist()
            predictions_labels += predict_content

    avg_epoch_loss = total_loss / len(dataloader)
    # 실제 라벨, 예측 라벨, 평균 손실 반환
    return true_labels, predictions_labels, avg_epoch_loss

# 학습된 모델을 이용해 예측을 수행하는 함수
def test(inputStr):
  global model # 전역 변수 model & 예측 수행

  model.eval() # 평가 모드로 변경

  batch = {k:v.type(torch.long) for k,v in batch.items()}
  # 딕셔너리 형태로 전처리 수행

  with torch.no_grad(): # 그래디언트 계산 비활성화
    outputs = model(**batch) # 모델 예측 수행
    loss, logits = outputs[:2] # 로스 & 로짓(확률) 추출
    logits = logits.detach().cpu().numpy() # 로짓을 넘파이 배열로 변환
    predict_content = logits.argmax(axis=-1).flatten().tolist()

  return predict_content # 로짓이 가장 큰 값의 인덱스 반환

from transformers import BertTokenizer
from transformers import BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
model = BertForSequenceClassification.from_pretrained("monologg/kobert", num_labels=n_labels)

model.resize_token_embeddings(len(tokenizer))

# -- 전처리 객체 초기화 ---
Classificaiton_collator = ClassificationCollator(use_tokenizer=tokenizer,
                                                          labels_encoder=labels_ids,
                                                          max_sequence_len=max_length)

# --- pytorch dataset 생성 ---
print('Dealing with Train...')
print()
dataset = JsonDataset(path=data_path,use_tokenizer=tokenizer)
print()
print()
print('Created `train_dataset` with %d examples!'%len(dataset))

# --- pytorch dataset -> input 형식 ---
df = pd.DataFrame.from_records(dataset)

# 라벨별 데이터 분포 확인
df.groupby('label').size()
df1 = df[df['label'].str.startswith('01')]
df2 = df[df['label'].str.startswith('02')]

# 클래스별 데이터 수 확인 (변경 후)
print('--- TRAIN (After Deleting) ---')
print(df1.groupby('label').size())
print(df2.groupby('label').size())
print()

# train, valid 분리 (9 : 1)
df1_train, df1_valid = train_test_split(df1, test_size=0.1, random_state=123)
df2_train, df2_valid = train_test_split(df2, test_size=0.1, random_state=123)

# 라벨 연결
df_train = pd.concat([df1_train,df2_train])
df_valid = pd.concat([df1_valid,df2_valid])

# train, valid 데이터셋 분포 확인
print(len(df_train))
print(len(df_valid))

print()

# 딕셔너리 형태로 변환
train_dataset = df_train.to_dict('records')
valid_dataset = df_valid.to_dict('records')

# 데이터 생성 (전처리 객체 호출 & Data Load)
print('Dealing with Train...')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=Classificaiton_collator)
print('Created `train_dataloader` with %d batches!'%len(train_dataloader))

print()

print('Dealing with Validation...')
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=Classificaiton_collator)
print('Created `eval_dataloader` with %d batches!'%len(valid_dataloader))

# --- 데이터 확인 ---
# 학습데이터 확인
print('--- TRAIN ---')
print(df_train.groupby('label').size())

print()

# 검증데이터 확인
print('--- VALIDATION ---')
print(df_valid.groupby('label').size())

print()

# 전체데이터 확인
print('--- ALL ---')
print(df.groupby('label').size())


# --- 모델 학습 시작 전 setting ---
# optimizer 생성
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # default is 1e-8.
                  )

# 총 학습 스텝 계산) = 배치 수 * 에폭 수
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

# 학습 값 저장 딕셔너리(리스트) 초기화 (to plot loss & acc curve)
all_loss = {'train_loss':[], 'val_loss':[]}
all_acc = {'train_acc':[], 'val_acc':[]}

# --- 그리드 서치 시작 ---
from sklearn.model_selection import ParameterGrid

# Define hyperparameter grid for grid search
param_grid = {
    'lr': [2e-5, 3e-5, 4e-5, 5e-5],  # Learning rates to try
    'batch_size': [16, 32],     # Batch sizes to try
    'epochs': [3, 4, 5],        # Number of epochs to try
}

# List to store grid search results
grid_search_results = []

# Iterate over all hyperparameter combinations in the grid
for params in ParameterGrid(param_grid):
    # Print current parameter combination
    print("\nTraining with parameters:", params)

    # Update the model's learning rate, batch size, and epochs
    optimizer = AdamW(model.parameters(), lr=params['lr'], eps=1e-8)
    total_steps = len(train_dataloader) * params['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Perform training with the current parameter combination
    for epoch in tqdm(range(params['epochs'])):
        train_labels, train_predict, train_loss = train(train_dataloader, optimizer, scheduler)

    # Perform validation with the current parameter combination
    valid_labels, valid_predict, val_loss = validation(valid_dataloader)

    # Calculate validation accuracy
    val_acc = accuracy_score(valid_labels, valid_predict)

    # Store results
    grid_search_results.append({
        'params': params,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc
    })

# Print grid search results
for result in grid_search_results:
    print("Parameters:", result['params'])
    print("Train Loss:", result['train_loss'])
    print("Validation Loss:", result['val_loss'])
    print("Validation Accuracy:", result['val_acc'])
    print()

    
# Find the result with the highest validation accuracy
best_result = max(grid_search_results, key=lambda x: x['val_acc'])

# Get the best parameters
best_params = best_result['params']

# Print the best parameters
print("Best Parameters:", best_params)

print('hi')

# --- 모델 학습 시작 전 setting ---
# optimizer 생성
optimizer = AdamW(model.parameters(),
                  lr=best_params['lr'],
                  eps=1e-8)

batch_size = best_params['batch_size']
epochs = best_params['epochs']

# 총 학습 스텝 계산) = 배치 수 * 에폭 수
total_steps = len(train_dataloader) * best_params['epochs']

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

# 학습 값 저장 딕셔너리(리스트) 초기화 (to plot loss & acc curve)
all_loss = {'train_loss': [], 'val_loss': []}
all_acc = {'train_acc': [], 'val_acc': []}

# --- 모델 학습 ---
# Loop through each epoch.

for epoch in tqdm(range(epochs)):
  print()
  print()

  print('Training on batches...')
  # Perform one full pass over the training set.
  train_labels, train_predict, train_loss = train(train_dataloader, optimizer, scheduler)
  train_acc = accuracy_score(train_labels, train_predict)

  print()
  print()


  # Get prediction form model on validation data.
  print('Validation on batches...')
  valid_labels, valid_predict, val_loss = validation(valid_dataloader)
  val_acc = accuracy_score(valid_labels, valid_predict)

  print()
  print()


  # Print loss and accuracy values to see how training evolves.
  print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"%(train_loss, val_loss, train_acc, val_acc))

  print()
  print()


  # 저장 the loss value for plotting the learning curve.
  all_loss['train_loss'].append(train_loss)
  all_loss['val_loss'].append(val_loss)
  all_acc['train_acc'].append(train_acc)
  all_acc['val_acc'].append(val_acc)

  # --- 모델 시각화 ---
# Plot loss curves.
plot_dict(all_loss, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'])

# Plot accuracy curves.
plot_dict(all_acc, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'])

# --- 모델 평가 ---
true_labels, predictions_labels, avg_epoch_loss = validation(valid_dataloader)

# Create the evaluation report.
evaluation_report = classification_report(true_labels, predictions_labels, labels=list(labels_ids.values()), target_names=list(labels_ids.keys()))

# Show the evaluation report.
print(evaluation_report)

# Plot confusion matrix.
plot_confusion_matrix(y_true=true_labels, y_pred=predictions_labels,
                      classes=list(labels_ids.keys())
                      );

# Plot confusion matrix.
plot_confusion_matrix(y_true=true_labels, y_pred=predictions_labels,
                      classes=list(labels_ids.keys())
                      );

# 저장할 디렉토리와 파일명 정의
output_dir = "output"

# 모델 저장
model_to_save = model.module if hasattr(model, 'module') else model
torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'favorable_or_not_classification_model_state_dict.pkl'))

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import pytesseract
import numpy as np
import re
from transformers import BertTokenizer 
from transformers import BertForSequenceClassification, set_seed
from torch.utils.data import DataLoader, Dataset
import torch
import os
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import copy
from konlpy.tag import Kkma
import pandas as pd
from sentence_transformers import SentenceTransformer
import itertools

set_seed(123)

app = Flask(__name__)
CORS(app)

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# # 유불리 모델 초기화
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)

model_path = '.\\bert_classification_model_state_dict.pkl'
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(state_dict)
model.eval()

# 요약 모델 초기화
nltk.download('punkt')
tokenizer_summ = AutoTokenizer.from_pretrained('Youngwoo9/T5_Pyeongsan')
model_summ = AutoModelForSeq2SeqLM.from_pretrained('Youngwoo9/T5_Pyeongsan')
model_summ.eval()

# 요약 리스트 초기화
ad_sentences = [] 
disad_sentences = []
ad_sum_sentences = []
disad_sum_sentences = []

# 제목, 키워드 리스트 초기화
ad_title = []
ad_keywords = []


@app.route('/api/upload', methods=['POST'])
def handle_image_upload():
    if 'image' not in request.files:
        return jsonify({'error': '이미지를 제공하지 않았습니다.'}), 400
    image = request.files['image']
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray_image, lang='kor')


    print('추출된 텍스트:', text)
    response_data = {'text': text}
    
    return jsonify(response_data)


@app.route('/api/process', methods=['POST'])
def process_text():
    data = request.json
    sentences = data.get('sentences', [])
    results = copy.deepcopy(sentences)

    # sentences 전처리
    sentences = list(filter(lambda sentences: len(sentences) > 5, sentences))

    # 리스트 초기화
    ad_sentences.clear()
    disad_sentences.clear()
    ad_sum_sentences.clear()
    disad_sum_sentences.clear()
    ad_title.clear()
    ad_keywords.clear()

    # input data 전처리
    sentences_input = ["[CLS] " + str(s) + " [SEP]" for s in sentences]


    class ClassificationCollator(object):
        # 토크나이저, 라벨 인코더, 최대 시퀀스 길이 설정 & 객체 초기화
        def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):
            self.use_tokenizer = use_tokenizer
            self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
            self.labels_encoder = labels_encoder
            return

        # 설정된 토크나이저, 라벨 인코더, 최대 시퀀스 길이를 이용해 데이터를 전처리
        def __call__(self, sequences):
            texts = sequences
            inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)

            return inputs

    Classificaiton_collator = ClassificationCollator(use_tokenizer=tokenizer,
                                                          labels_encoder=2,
                                                          max_sequence_len=128)

    print('Dealing with Data...')
    sentences_input = DataLoader(sentences_input, collate_fn=Classificaiton_collator)
    print('Created `sentences` with %d batches!'%len(sentences_input))


    def predict_sentiment(tensor, model, tokenizer):
        inputs = tensor
        model.eval()

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        probabilities = torch.softmax(logits, dim=1)
        prob_1, prob_2 = probabilities[0]

        return prob_1.item(), prob_2.item()


    for tensor, text in zip(sentences_input, sentences) :
        print('들어온 문장 : ',text,'\n')
        # import pdb; pdb.set_trace()
        print()
        prob_1, prob_2 = predict_sentiment(tensor, model, tokenizer)
        print()
        print('유리확률 : ',prob_1,'\n')
        print()
        print()
        print('불리확률 : ',prob_2,'\n')
        print()
        if prob_1 > 0.95:
            ad_sentences.append(text)
        if prob_2 > 0.95:
            disad_sentences.append(text)
    print()
    print()
    print('유리 조항 리스트')
    print(ad_sentences)
    print()
    print()
    print('불리 조항 리스트')
    print(disad_sentences)
    print()
    print()
    print()
    prefix = "summarize: "

    for sentence in ad_sentences:
        if not sentence.endswith('.'):
            ad_sum_sentences.append(sentence)
            
        else:
            input_text = prefix + sentence.replace('"', '')

            inputs = tokenizer_summ(input_text, max_length=1024, truncation=True, return_tensors="pt")
            output = model_summ.generate(**inputs, num_beams=3, do_sample=True, min_length=1, max_length=len(sentence))
            decoded_output = tokenizer_summ.batch_decode(output, skip_special_tokens=True)[0]
            decoded_output = decoded_output.strip()

            if decoded_output:
                sentences_summ = nltk.sent_tokenize(decoded_output)

                if sentences_summ:
                    result = sentences_summ[0]
                    ad_sum_sentences.append(result)


    for sentence in disad_sentences:
        if not sentence.endswith('.'):
            disad_sum_sentences.append(sentence)

        else:
            input_text = prefix + sentence.replace('"', '')

            inputs = tokenizer_summ(input_text, max_length=1024, truncation=True, return_tensors="pt")
            output = model_summ.generate(**inputs, num_beams=3, do_sample=True, min_length=1, max_length=len(sentence))
            decoded_output = tokenizer_summ.batch_decode(output, skip_special_tokens=True)[0]
            decoded_output = decoded_output.strip()

            if decoded_output:
                sentences_summ = nltk.sent_tokenize(decoded_output)

                if sentences_summ:
                    result = sentences_summ[0]
                    disad_sum_sentences.append(result)
    print()
    print()
    print()
    print('요약된 유리 조항 리스트')
    print(ad_sum_sentences)
    print()
    print()
    print('요약된 불리 조항 리스트')
    print(disad_sum_sentences)
    print()
    print()
    print()
    
    #타이틀 모델
    try: 
        def extract_nouns(text):
            kkma = Kkma()
            nouns = kkma.nouns(text)
            filtered_nouns = [noun for noun in nouns if len(noun) > 1 and not (re.match(r'^\d+$', noun) or re.search(r'\d', noun))]
            return " ".join(filtered_nouns)
            

        # 약관 설명 XML 파일들을 읽어서 데이터프레임으로 만들어주는 함수
        def read_xml_files(xml_data_folder):
            data = []
            for filename in os.listdir(xml_data_folder):
                if filename.endswith(".xml"):
                    filepath = os.path.join(xml_data_folder, filename)
                    with open(filepath, 'r', encoding='utf-8') as file:  # 여기 수정함
                        xml_text = file.read()
                    data.append({'filename': filename, 'xml_text': xml_text})
            return pd.DataFrame(data)


        # 이미 추출한 텍스트 데이터 활용
        image_to_text =  ' '.join(sentences)

        # 명사 추출
        user_nouns = extract_nouns(image_to_text)

        # 약관 설명 XML 파일들을 읽어서 데이터프레임으로 만듦
        xml_data_folder = "./sample/Terms/rawData"

        legal_data = read_xml_files(xml_data_folder)  # read_xml_files 함수가 호출되어야 합니다.

        # TfidfVectorizer 적용
        stop_words = ['조건','기타', '설명', '제시', '제공', '약관', '정보','목적','경우','필요','책임','본인','국가','총칙','모든','약관의','내용','명시','이상','이하','사항','공지','사유', '안녕']  # 불용어 리스트에 추가
        tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
        tfidf_matrix = tfidf_vectorizer.fit_transform(legal_data['xml_text'])
        user_tfidf = tfidf_vectorizer.transform([user_nouns])

        # 유사도 측정하여 가장 유사한 약관 파일 찾음
        similarities = cosine_similarity(user_tfidf, tfidf_matrix)
        most_similar_index = similarities.argmax()
        most_similar_filename = legal_data.loc[most_similar_index, 'filename']

        # 파일 이름에서 앞의 3글자와 뒤의 6글자를 제외한 부분 추출
        parts = most_similar_filename.split("_")
        keyword = parts[1]
        # 출력
        print("약관명:", keyword)
        ad_title.append(keyword)

    except Exception as e:
            print(f"오류가 발생했습니다: {e}")
            return jsonify({'error': f"서버 안에서 발생한 내부 오류입니다: {e}"}), 150

    # import pdb; pdb.set_trace()

    # 키워드 추출
    try:
        # doc = sentences

        # Kkma를 사용하여 명사 추출
        kkma = Kkma()
        nouns = kkma.nouns(image_to_text)
        stop_words = ['조건','기타', '설명', '제시', '제공', '약관', '정보','목적','경우','필요','책임','본인','국가','총칙','모든','약관의','내용','명시','이상','이하','사항','공지','사유']  # 필요한 불용어를 추가해주세요

        nouns = list(filter(lambda noun: len(noun) > 1 and noun not in stop_words, nouns))

        keyword_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        doc_embedding = keyword_model.encode([image_to_text])
        candidate_embeddings = keyword_model.encode(nouns)
        def max_sum_sim(doc_embedding, candidate_embeddings, words, top_n, nr_candidates):
            distances = cosine_similarity(doc_embedding, candidate_embeddings)
            distances_candidates = cosine_similarity(candidate_embeddings, candidate_embeddings)

            words_idx = list(distances.argsort()[0][-nr_candidates:])
            words_vals = [nouns[index] for index in words_idx]
            distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

            min_sim = np.inf
            candidate = None
            for combination in itertools.combinations(range(len(words_idx)), top_n):
                sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
                if sim < min_sim:
                    candidate = combination
                    min_sim = sim

            return [words_vals[idx] for idx in candidate]

        # import pdb; pdb.set_trace()
        
        # '키워드 추출 모델' 호출하여 이미지에서 주요 키워드 추출
        top_keywords = max_sum_sim(doc_embedding, candidate_embeddings, nouns, top_n=6, nr_candidates=10)

        # 추출된 상위 6개 명사 출력
        print("주요 키워드:")
        print(top_keywords)
        ad_keywords.append(top_keywords)

    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        return jsonify({'error': f"서버 안에서 발생한 내부 오류입니다: {e}"}), 350
        
    print(f'title {ad_title}')
    print(f'keyword {ad_keywords}')
    
    print(results)


    return jsonify({'text': results})



@app.route('/process', methods=['GET'])
def process():
    response_data = {
        'positive_texts': ad_sentences,
        'negative_texts': disad_sentences,
        'sum_p_texts': ad_sum_sentences,
        'sum_n_texts': disad_sum_sentences,
        'title_texts' : ad_title,
        'keyword_texts' : ad_keywords
        }
    
    print(response_data)
    return jsonify(response_data)



if __name__ == "__main__":
    app.run('0.0.0.0', port=5000, threaded=True)
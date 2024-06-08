from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import numpy as np
from datasets import Dataset, load_dataset
import json
from huggingface_hub import login
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from multiprocessing import Pool
import pandas as pd
import re
import nltk
import ast
import os
nltk.download('punkt',quiet=True)
nltk.download('stopwords',quiet=True)
nltk.download('wordnet',quiet=True)
login("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")
llm = LLM(model="microsoft/Phi-3-mini-128k-instruct",gpu_memory_utilization=0.95,max_model_len=4096,tensor_parallel_size=2)  

dataset = load_dataset("taln-ls2n/kp20k",trust_remote_code=True)



tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_abstract(abstract):
    tokens = word_tokenize(abstract)
    processed_tokens = []
    for token in tokens:
        token = token.lower()  # Lowercasing
        if token not in stop_words:
            token = stemmer.stem(token)
            token = lemmatizer.lemmatize(token)
            processed_tokens.append(token)
    return ' '.join(processed_tokens)

def prepend_title_to_abstract(titles, abstracts):
    combined_abstracts = []
    for title, abstract in zip(titles, abstracts):
        combined_abstract = title + " " + abstract
        combined_abstracts.append(combined_abstract)
    return combined_abstracts

def calculate_kpp_s(keyphrase_tokens, token_info_list):
    product = 1.0
    for token_info in token_info_list:
        log_prob = token_info['log_prob']
        conditional_prob = np.exp(log_prob)
        product *= conditional_prob
    wc = len(keyphrase_tokens)
    if wc == 0:
        return 0.0  # Return 0 if the number of tokens is zero
    kpp = (product ** (-1 / wc))
    return kpp


titles = dataset['test']['title']
abstracts = dataset['test']['abstract']
ground_truth_keyphrases = dataset["test"]["keyphrases"]
abstracts_all = prepend_title_to_abstract(titles, abstracts)



total_data = {}
for idx, (title, abstract) in enumerate(zip(dataset['test']['title'],dataset['test']['abstract'])):
    total_data[idx] = {'title': title, 'abstract': abstract}

preprocessed_keyphrases=[]
for i in range(len(abstracts)):
    keyphrases_list= ground_truth_keyphrases[i]
    processed_keys = [preprocess_abstract(word) for word in keyphrases_list]
    preprocessed_keyphrases.append(processed_keys)


system_prompt = "You are an expert in finding the best keyphrases to represent the topical information of a scientific document just from reading its title and abstract."
user_prompt1 = "Generate present and absent keyphrases from the following title and abstract of a scientific document."
INS1 = "1. List the most appropriate and essential keyphrases. Each keyphrase should consist of only one to three words. Do not create too many keyphrases.\n2. Only respond with a list of comma-separated keyphrases in the following format: [\"keyphrase 1\", \"keyphrase 2\", ..., \"keyphrase n\"]"


def keyphrase_generator(title,abstract):
    
    
  
    prompt_try_2= f'''<|user|>\n{user_prompt1}\n**Title:** \"{title}\"\n\n**Abstract:** \"{abstract}\"\n\n **Instructions:** \n{INS1} <|end|>\n<|assistant|>\n'''
    
    prompt_try_3= f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_prompt1}\n**Title:** \"{title}\"\n\n**Abstract:** \"{abstract}\"\n\n **Instructions:** \n{INS1} <|end|>\n<|assistant|>\n["



    request = prompt_try_3
    sampling_params = SamplingParams(temperature=0,top_p=0.5,top_k=-1,min_p=0.4,logprobs=1,seed=42,max_tokens=500)
    outputs = llm.generate(request, sampling_params,use_tqdm=False)
    
    return outputs


def is_keyphrase_present(src, keyphrase):
    # Modified to check for whole words only
    presence = re.search(r'\b{}\b'.format(re.escape(keyphrase)), src, re.IGNORECASE) is not None
    return presence

def process_abstract(abstract_index):
    abstract = abstracts_all[abstract_index]
    processed_src = preprocess_abstract((abstract))
    processed_ground_truth = preprocessed_keyphrases[abstract_index]
    gt_present = []
    gt_absent = []
    for keyphrase_list in [processed_ground_truth]:
        for keyphrase in keyphrase_list:
            if is_keyphrase_present(processed_src, keyphrase):
                gt_present.append(keyphrase)
            else:
                gt_absent.append(keyphrase)  # Append the full keyphrase, not individual characters
    return gt_present, gt_absent




num_cores = 16  # Adjust number of cores as needed
with Pool(num_cores) as pool:
    results = pool.map(process_abstract, range(len(abstracts_all)))
    gt_present_lists, gt_absent_lists = zip(*results)



print("checking if GT present ,absent  length matches or not")
len(gt_absent_lists)==len(gt_present_lists)==len(abstracts)

print("gt_present_lists[13]")
print(gt_present_lists[13])
print()
print("gt_absent_lists[13]")
print(gt_absent_lists[13])
print("PRMU | Key Phrases")
print("------------------------------------")
for prmu_item, keyphrase in zip(dataset["test"]['prmu'][13], dataset["test"]['keyphrases'][13]):
    print(f"{prmu_item:<0} | {keyphrase}")
print()




all_pred_keyphrases = []
output_text = ""  # Initialize an empty string to hold the output

for i in range(len(total_data)):
    generated_keyphrases = keyphrase_generator(total_data[i]['title'],total_data[i]['abstract'])
    generated_text = generated_keyphrases[0].outputs[0].text
    log_probs = generated_keyphrases[0].outputs[0].logprobs
    if i % 100 == 0:
        print(f"{i} Documents Processed", flush=True)
    a='['+generated_keyphrases[0].outputs[0].text
    data_list=(a.strip('[]')).replace('"', '').replace('",', '').split(', ')
    all_pred_keyphrases.append(data_list)



    token_info_list = [
        {'token_id': token_id, 'log_prob': logprob_info.logprob, 'decoded_token': logprob_info.decoded_token}
        for token_probs in log_probs
        for token_id, logprob_info in token_probs.items()
    ]

    temp_keyphrase_tokens = []
    keyphrases = []
    keyphrase_kpp_s_values = []
    word_counts = []

    for token_info in token_info_list:
        decoded_token = token_info['decoded_token']
        if decoded_token.strip():
            temp_keyphrase_tokens.append(decoded_token)
        else:
            kpp_s = calculate_kpp_s(temp_keyphrase_tokens, token_info_list)
            word_count = len(temp_keyphrase_tokens)
            if word_count != 0:
                kpp_s_normalized = kpp_s / word_count
                keyphrases.append(' '.join(temp_keyphrase_tokens))
                keyphrase_kpp_s_values.append(kpp_s_normalized)
                word_counts.append(word_count)
                temp_keyphrase_tokens = []

    output_text += f"Keyphrases with their normalized KPP-s values and word counts for abstract:{i}\n"
    output_text += "---------------------------------------------------------------------\n"
    output_text += "| Keyphrase                        | KPP-s Value | Word Count |\n"
    output_text += "---------------------------------------------------------------------\n"
    for keyphrase, kpp_s, count in zip(keyphrases, keyphrase_kpp_s_values, word_counts):
        output_text += f"| {keyphrase:32s} | {kpp_s:.5f}   | {count:10d} |\n"
    output_text += "---------------------------------------------------------------------\n"

# Print the lists of generated keyphrases
output_text += "Lists of generated keyphrases:\n"
output_text += "---------------------------------------------------------------------\n"
output_text += str(all_pred_keyphrases)




output_dir = "/mnt/jayanth-llama-volume/kgc"
os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(output_dir, "Phi_latest_final.txt")
with open(file_path, 'w') as file:
    for item in all_pred_keyphrases:
        file.write(f"{item}\n")
    
if os.path.exists(file_path):
    print("File saved successfully.")
else:
    print("Error: File not saved.")





import re

pred_present_lists = []
pred_absent_lists = []

# Function to check if a keyphrase is present in the source text
def is_keyphrase_present(src, keyphrase):
    src_str = ' '.join(src)  # Join source list into a string
    keyphrase_str = ' '.join(keyphrase)  # Join keyphrase list into a string
    return re.search(r'\b{}\b'.format(re.escape(keyphrase_str)), src_str, re.IGNORECASE) is not None

# Iterate over the indices of total_data
for idx in range(len(total_data)):
    abstract = abstracts_all[idx]
    processed_abstract = preprocess_abstract((abstract))
    
    if idx % 1000 == 0:
        print(f"Generation of Pred Present absent for {idx} Documents Processed",flush=True)
        
    processed_preds= [preprocess_abstract((f"{item.strip()}")) for item in all_pred_keyphrases[idx]]
    
    pred_present = []
    pred_absent = []

    for keyphrase_list in processed_preds:
        if keyphrase_list: 
            keyphrase = keyphrase_list 
            if is_keyphrase_present(processed_abstract, keyphrase):
                pred_present.append(keyphrase)
            else:
                pred_absent.append(keyphrase)

    pred_present_lists.append(pred_present)
    pred_absent_lists.append(pred_absent)



print("len(all_pred_keyphrases), len(pred_present_lists), len(pred_absent_lists)")
print(len(all_pred_keyphrases), len(pred_present_lists), len(pred_absent_lists))
print('\n')
print("all_pred_keyphrases[2]")
print(all_pred_keyphrases[2])
print('\n')
print("abstracts_all[2]")
print(abstracts_all[2])
print('\n')
print("pred_present_lists[2]")
print(pred_present_lists[2])
print('\n')
print("pred_absent_lists[2]")
print(pred_absent_lists[2])
print()
print()
print()

for i in range(2,4,1):
    print('\n')
    print(f"pred_present_lists ({i})")
    print(pred_present_lists[i])
    print('\n')
    print(f"gt_present_lists({i})")
    print(gt_present_lists[i])
    print('\n')
    print(f"pred_absent_lists({i})")
    print(pred_absent_lists[i])
    print('\n')
    print(f"gt_absent_lists({i})")
    print(gt_absent_lists[i])


import copy

def calculate_f1_score(pred, gt):
    if len(pred) == 0 or len(gt) == 0:
        return 0
    precision = len(set(pred) & set(gt)) / len(pred)
    recall = len(set(pred) & set(gt)) / len(gt)
    if precision + recall == 0:
        return 0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def evaluate(pred_present_lists, gt_present_lists, pred_absent_lists, gt_absent_lists):
    total_present_f1 = {"5R": [], "M": [], "5": [], "10": [], "50": []}
    total_absent_f1 = {"5R": [], "M": [], "5": [], "10": [], "50": []}

    for pred_present, gt_present, pred_absent, gt_absent in zip(pred_present_lists, gt_present_lists, pred_absent_lists, gt_absent_lists):
        original_pred_present = copy.deepcopy(pred_present)
        original_pred_absent = copy.deepcopy(pred_absent)

        for topk in ["5R", "M", "5", "10", "50"]:
            pred_present = copy.deepcopy(original_pred_present)
            pred_absent = copy.deepcopy(original_pred_absent)

            if topk == "M":
                pass
            else:
                if "R" in topk:
                    R = True
                    topk_val = int(topk[:-1])
                else:
                    R = False
                    topk_val = int(topk)

                if len(pred_present) > topk_val:
                    pred_present = pred_present[:topk_val]
                elif R:
                    while len(pred_present) < topk_val:
                        pred_present.append("<fake keyphrase>")

                if len(pred_absent) > topk_val:
                    pred_absent = pred_absent[:topk_val]
                elif R:
                    while len(pred_absent) < topk_val:
                        pred_absent.append("<fake keyphrase>")

                topk = str(topk_val)
                if R:
                    topk = topk + "R"

            f1_present = calculate_f1_score(pred_present, gt_present)
            f1_absent = calculate_f1_score(pred_absent, gt_absent)
            total_present_f1[topk].append(f1_present)
            total_absent_f1[topk].append(f1_absent)

    avg_f1_scores_present = {topk: sum(total_present_f1[topk]) / len(total_present_f1[topk]) for topk in total_present_f1}
    avg_f1_scores_absent = {topk: sum(total_absent_f1[topk]) / len(total_absent_f1[topk]) for topk in total_absent_f1}

    return {"avg_f1_scores_present": avg_f1_scores_present, "avg_f1_scores_absent": avg_f1_scores_absent}



system_prompt = "You are an expert in finding the best keyphrases to represent the topical information of a scientific document just from reading its title and abstract."
user_prompt = "Generate present and absent keyphrases from the following title and abstract of a scientific document."
INS1 = "1. List the most appropriate and essential keyphrases. Each keyphrase should consist of only one to three words. Do not create too many keyphrases.\n2. Only respond with a list of comma-separated keyphrases in the following format: [\"keyphrase 1\", \"keyphrase 2\", ..., \"keyphrase n\"]"

print("system_prompt:")
print(system_prompt)
print("User Prompt:")
print(user_prompt)
print("\nInstructions:")
print(INS1)


print()
results = evaluate(pred_present_lists, gt_present_lists, pred_absent_lists, gt_absent_lists)
print("Present Scores",results['avg_f1_scores_present'])
print("Absent Scores",results['avg_f1_scores_absent'])
for topk, f1_score in results['avg_f1_scores_present'].items():
    print(f"F1@{topk} for present keyphrases:", f1_score)
print('\n')
for topk, f1_score in results['avg_f1_scores_absent'].items():
    print(f"F1@{topk} for absent keyphrases:", f1_score)

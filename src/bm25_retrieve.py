import os
import json
import argparse
from tqdm import tqdm
import jieba  # for text segmentation in Chinese
import pdfplumber 
from rank_bm25 import BM25Okapi  
import pandas as pd


def load_data(source_path):
    print("Loading data from {}...".format(source_path))
    masked_file_ls = os.listdir(source_path)  
    corpus_dict = {str(file.replace('.pdf', '')): read_pdf(os.path.join(source_path, file)) for file in tqdm(masked_file_ls)}  
    return corpus_dict



def read_pdf(pdf_loc, page_infos: list = None):
    pdf = pdfplumber.open(pdf_loc) 
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    for _, page in enumerate(pages): 
        text = page.extract_text()  
        if text:
            pdf_text += text
    pdf.close() 

    return pdf_text  



def BM25_retrieve_full(qs,tokenized_corpus,keys):
    bm25 = BM25Okapi(tokenized_corpus)  
    tokenized_query = list(jieba.cut_for_search(qs))  
    ans = bm25.get_top_n(tokenized_query, keys , n=1)  
    return ans[0]


# search only in the filtered_corpus
def BM25_retrieve_partial(qs,tokenized_corpus,keys):
    bm25 = BM25Okapi(tokenized_corpus) 
    tokenized_query = list(jieba.cut_for_search(qs))
    ans = bm25.get_top_n(tokenized_query, keys , n=1)
    return ans[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='Path to read the question file')  # Path for the question file
    parser.add_argument('--source_path', type=str, required=True, help='Path to read the reference materials')  # Path for reference materials
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output in the competition format')  # Path for saving output answers    
    parser.add_argument('--answer_path', type=str, required=True, help='Path to answer file') # Path the ground truth for calculating the score
    parser.add_argument('--search_full', action='store_true', help='Enable FuLL search') # New argument for enabling full_search

    args = parser.parse_args() 

    answer_dict = {"answers": []}  

    # Load the question file
    with open(args.question_path, 'rb') as f:
        qs_ref = json.load(f)  
    

    # key: file name, value: text content
    data_finance = os.path.join(os.path.dirname(__file__), 'finance.json') 
    if not os.path.exists(data_finance):
        source_path_finance = os.path.join(args.source_path, 'finance') 
        corpus_dict_finance = load_data(source_path_finance)
        with open(data_finance, 'w+') as f_s:
            json.dump(corpus_dict_finance, f_s,ensure_ascii=False)
    else:
        with open(data_finance, 'r') as f_s:
            corpus_dict_finance = json.load(f_s)

    

    if not os.path.exists('cut_terms.json'):
        cut_terms = {}
        cut_terms['finance'] = {}
        for key, value in corpus_dict_finance.items():
            cut_terms['finance'][key] = list(jieba.cut_for_search(value))

        with open('cut_terms.json', 'w') as f:
            json.dump(cut_terms, f,ensure_ascii=False)
    else:
        with open('cut_terms.json', 'r') as f:
            cut_terms = json.load(f)


    if args.search_full:    
        # Full search        
        for q_dict in tqdm(qs_ref['questions']):
            t = q_dict['category'] 
            if t in ['finance']:   
                retrieved = BM25_retrieve_full(q_dict['query'], list(cut_terms['finance'].values()),list(cut_terms['finance'].keys()))
                answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
    else:
        # Partial search
        for q_dict in tqdm(qs_ref['questions']):         
            t = q_dict['category'] 
            if t in ['finance']:   
                source_str = list(map(str,q_dict['source']))
                filtered_corpus = []
                for k in source_str:
                    filtered_corpus.append(cut_terms['finance'][k])                                
                retrieved = BM25_retrieve_partial(q_dict['query'], filtered_corpus,source_str)
                answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})    
        
    
    # save the answer dictionary as a json file
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)  
        
    if args.answer_path:
        with open(args.answer_path, 'r') as f:
            ground_truth = json.load(f)
        ground_truth_df = pd.DataFrame(ground_truth['ground_truths'])
        # show the ground truth
        print(ground_truth_df.head())
        
        
        pred_retrieve_df = pd.DataFrame(answer_dict['answers'])
        pred_retrieve_df['ori'] = pred_retrieve_df['retrieve']
        # extract doc_id and compare with the ground truth
        pred_retrieve_df['retrieve'] = pred_retrieve_df.apply(lambda x: x['ori'].split('_')[0], axis=1)
        # show the prediction
        print(pred_retrieve_df.head())
        
        
        ground_truth_df = pd.DataFrame(ground_truth['ground_truths'])
        compare_df = pd.merge(ground_truth_df, pred_retrieve_df, on='qid', suffixes=('_ground_truth', '_retrieve'))
        compare_df.head()
        compare_df['correct'] = compare_df.apply(lambda x: int(x['retrieve_retrieve']) in x['retrieve_ground_truth'], axis=1)
        percision = compare_df['correct'].sum() / len(compare_df)
        print(f"Precision: {percision}")
    
    
    

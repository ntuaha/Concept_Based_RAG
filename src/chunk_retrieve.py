import json
from rank_bm25 import BM25Okapi  
import jieba
from tqdm import tqdm
import argparse
import os
import math
import pandas as pd
import math

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


def BM25_retrieve(qs,tokenized_corpus,keys):
    bm25 = BM25Okapi(tokenized_corpus)  
    tokenized_query = list(jieba.cut_for_search(qs))  
    ans = bm25.get_top_n(tokenized_query, keys , n=1)  
    return ans[0]



def get_chunks(data):
    # key: [doc_id]_[paragraph_counter]_[concept_counter]_[type]
    # type: s for sentence, q for question
    new_docs = {}
    for key in data.keys():
        counter = 0
        for k2 in data[key].keys():
            for p in data[key][k2]['partitions']:
                if ('sentence' in p) and (p['sentence'] != ''):
                    new_docs[f'{key}_{k2}_{counter}_s'] = p['sentence']
                    counter += 1
                    for q in p['questions']:
                        new_docs[f'{key}_{k2}_{counter}_q'] = q
                        counter += 1
    return new_docs

def segment(chunk_with_segmentation_path, chunked_data):
    """
    Segments the provided chunked data using jieba's cut_for_search method.

    Args:
        chunked_data (dict): A dictionary where keys are identifiers and values are text data to be segmented.

    Returns:
        dict: A dictionary with segmented terms categorized under 'finance', 'medical', and 'travel'.
    """
    if not os.path.exists(chunk_with_segmentation_path):        
        cut_terms = {'finance': {}, 'medical': {}, 'travel': {}}    
        for key, value in tqdm(chunked_data.items()):
                cut_terms['finance'][key] = list(jieba.cut_for_search(value))
        with open(chunk_with_segmentation_path, 'w') as f:
            json.dump(cut_terms, f, ensure_ascii=False, indent=4)
    else:
        with open(chunk_with_segmentation_path, 'r') as f:
            cut_terms = json.load(f)    
    return cut_terms



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='Path to read the question file')  # Path for the question file
    parser.add_argument('--chunk_path', type=str, required=True, help='Path to read the reference materials chunked')  # Path for chunks
    parser.add_argument('--chunk_with_segmentation_path', type=str, default='cut_terms_2.json', required=True, help='Path to the reference materials that have been processed with word segmentation.')  # Path for chunks
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output in the competition format')  # Path for saving output answers
    parser.add_argument('--answer_path', type=str, required=True, help='Path to answer file') # Path the ground truth for calculating the score
    parser.add_argument('--search_full', action='store_true', help='Enable FuLL search') # New argument for enabling full_search

    args = parser.parse_args() 
    
    
    # Load the chunked data
    with open(args.chunk_path, 'r') as f:
        data = json.load(f)
                
    chunked_data = get_chunks(data)
    print(f"Chunked data: {len(chunked_data)}")
    print("Sample chunked data:")
    print(list(chunked_data.values())[:5])
    print(list(chunked_data.keys())[:5])
    
    # Segment the chunked data
    segmented_data = segment(args.chunk_with_segmentation_path, chunked_data)
    print(f"Segmented data: {len(segmented_data['finance'].keys())}")
    
    # load the question file
    with open(args.question_path, 'rb') as f:
        qs_ref = json.load(f)
    
    # only process finance category
    answer_dict = {}
    
    
    if not os.path.exists(args.output_path):
        answer_dict['answers'] = []

        for q_dict in tqdm(qs_ref['questions']):
                # search in the full documents or only in partial documents
                if not args.search_full:
                    t = q_dict['category']                         
                    source_str = list(map(str,q_dict['source']))
                    filtered_docs = []
                    filtered_keys = []   
                    if t in ['finance']:
                        for ck in segmented_data['finance'].keys():
                            if ck.split('_')[0] in source_str:
                                filtered_docs.append(segmented_data['finance'][ck])
                                filtered_keys.append(ck)                
                        retrieved = BM25_retrieve(q_dict['query'], filtered_docs,filtered_keys)
                        answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
                else:
                    filtered_docs = segmented_data['finance']
                    filtered_keys = list(segmented_data['finance'].keys())
                    retrieved = BM25_retrieve(q_dict['query'], filtered_docs,filtered_keys)
                    answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
                
        
        
        with open(args.output_path, 'w') as f:
            json.dump(answer_dict, f, ensure_ascii=False, indent=4)
    else:
        with open(args.output_path, 'r') as f:
            answer_dict = json.load(f)
        
        
    # calculate the score
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
        
        answer_dict['precision'] = percision
        with open(args.output_path, 'r') as f:
                answer_dict = json.load(f)
        
        # entropy under question and answer
        k = pred_retrieve_df[['retrieve']].groupby('retrieve').value_counts().to_dict()
        g = 0
        for key,value in k.items():
            g += math.log(value,2)
        answer_dict['entropy'] = {
            'entropy_total': g,
            'entropy_mean': g/len(k),
            'entropy_max': max(k.values()),
            'entropy_min': min(k.values()),
            'count': len(k),
            'entropy': k
        }
        #print(answer_dict['entropy'])
        with open(args.output_path, 'w') as f:
            json.dump(answer_dict, f, ensure_ascii=False, indent=4)

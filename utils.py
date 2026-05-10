# feature_utils.py
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import re
import math
from collections import Counter
from tqdm.auto import tqdm
from functools import partial 


def extract_features(code, feature_mask=None):
    lines = code.split('\n')

    # 1. Line-level features
    line_lengths = [len(l) for l in lines]
    avg_line_len = np.mean(line_lengths)
    std_line_len = np.std(line_lengths) if len(line_lengths) > 1 else 0

    # 2. Indentation features
    indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    avg_indent = np.mean(indents) if indents else 0
    std_indent = np.std(indents) if len(indents) > 1 else 0
    max_indent = max(indents) if indents else 0

    # 3. Empty line gap features
    gaps = []
    gap = 0
    for l in lines:
        if not l.strip():
            gap += 1
        else:
            if gap > 0:
                gaps.append(gap)
            gap = 0
    empty_gap_std = float(np.std(gaps)) if len(gaps) > 1 else 0.0
    empty_gap_mean = float(np.mean(gaps)) if gaps else 0.0

    # 4. Comments Extraction
    # Single-line comments (# or //)
    single_comments = re.findall(r'(?://|#)(.*)', code)
    # Block comments (/* ... */, """...""", '''...''') - Kept for text_comment_ratio accuracy
    block_comments_raw = re.findall(r'/\*.*?\*/|""".*?"""|\'\'\'.*?\'\'\'', code, re.DOTALL)
    
    n_single_comments = len(single_comments)

    # Heuristics for Text vs Code comments
    code_comment_count = 0
    text_comment_count = 0
    # Code comment heuristics: contains operators, brackets, or code keywords at the start of words
    code_pattern = re.compile(r'(=|==|!=|\+=|-=|\*=|/=|def\s+|class\s+|for\s+|while\s+|if\s+|elif\s+|else:|return\s+|import\s+|from\s+|\{|\}|\(|\)|\[|\]|;)')
    
    for c in single_comments:
        c_clean = c.strip()
        if not c_clean:
            continue
        if code_pattern.search(c_clean):
            code_comment_count += 1
        else:
            text_comment_count += 1
            
    for bc in block_comments_raw:
        bc_clean = re.sub(r'^/\*|\*/$|^"""|"""$|^\'\'\'|\'\'\'$', '', bc).strip()
        if not bc_clean:
            continue
        if code_pattern.search(bc_clean):
            code_comment_count += 1
        else:
            text_comment_count += 1
            
    total_non_empty_comments = text_comment_count + code_comment_count
    text_comment_ratio = text_comment_count / total_non_empty_comments if total_non_empty_comments > 0 else 0.0

    all_features = {
        'avg_line_len': float(avg_line_len),
        'std_line_len': float(std_line_len),
        'avg_indent': float(avg_indent),
        'std_indent': float(std_indent),
        'max_indent': float(max_indent),
        'empty_gap_std': float(empty_gap_std),
        'empty_gap_mean': float(empty_gap_mean),
        'n_single_comments': float(n_single_comments),
        'text_comment_ratio': float(text_comment_ratio),
    }

    if feature_mask is None:
        feature_mask = list(all_features.keys())

    return [all_features[feature] for feature in feature_mask if feature in all_features]

def process_batch(code_batch, features_mask=None):
    """
    Hàm xử lý cho một batch.
    """
    return [extract_features(code, features_mask) for code in code_batch]

def parallel_extract(code_list, features_mask=None, num_workers=10, batch_size=200):
    """
    Hàm xử lý song song có chia batch và hiển thị progress.
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        
    # 1. Chia danh sách code thành các batches
    batches = [code_list[i:i + batch_size] for i in range(0, len(code_list), batch_size)]
    
    results = []
    
    # 2. Tạo một partial function. 
    # Hàm này tương đương với: process_batch(batch, features_mask=features_mask)
    worker_func = partial(process_batch, features_mask=features_mask)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 3. Truyền worker_func vào executor.map thay vì lambda
        for batch_result in tqdm(executor.map(worker_func, batches), total=len(batches), desc="Batch Progress"):
            results.extend(batch_result)
            
    return np.array(results)
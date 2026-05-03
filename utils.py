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
    n_lines = max(len(lines), 1)

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
    # Block comments (/* ... */, """...""", '''...''')
    block_comments_raw = re.findall(r'/\*.*?\*/|""".*?"""|\'\'\'.*?\'\'\'', code, re.DOTALL)
    
    n_single_comments = len(single_comments)
    n_block_comments = len(block_comments_raw)
    total_comments = n_single_comments + n_block_comments
    
    # Calculate comment lengths
    single_comment_lengths = [len(c.strip()) for c in single_comments]
    
    block_comment_lengths = []
    for bc in block_comments_raw:
        bc_clean = re.sub(r'^/\*|\*/$|^"""|"""$|^\'\'\'|\'\'\'$', '', bc).strip()
        block_comment_lengths.append(len(bc_clean))
        
    all_comment_lengths = single_comment_lengths + block_comment_lengths
    avg_comment_len = np.mean(all_comment_lengths) if all_comment_lengths else 0.0
    
    comment_ratio = total_comments / n_lines
    comment_density = sum(all_comment_lengths) / max(len(code), 1)

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
    code_comment_ratio = code_comment_count / total_non_empty_comments if total_non_empty_comments > 0 else 0.0

    # 5. Variable Convention Extraction
    tokens = re.findall(r'\b[a-zA-Z_]\w*\b', code)
    common_keywords = {'if', 'else', 'elif', 'for', 'while', 'return', 'def', 'class',
                       'function', 'var', 'let', 'const', 'int', 'string', 'float', 'boolean',
                       'import', 'from', 'include', 'using', 'new', 'public',
                       'private', 'protected', 'static', 'void', 'true', 'false', 'null', 'None',
                       'try', 'catch', 'except', 'finally', 'throw', 'raise', 'switch', 'case', 'break', 'continue',
                       'self', 'this', 'and', 'or', 'not', 'in', 'is', 'as', 'with', 'yield', 'async', 'await'}
    identifiers = [t for t in tokens if t not in common_keywords]
    n_ids = max(len(identifiers), 1)
    
    camel_case_count = 0
    snake_case_count = 0
    pascal_case_count = 0
    screaming_snake_count = 0
    
    for t in identifiers:
        if re.match(r'^[a-z][a-z0-9]*([A-Z][a-z0-9]*)+$', t):
            camel_case_count += 1
        elif re.match(r'^[a-z][a-z0-9_]*$', t) and '_' in t:
            snake_case_count += 1
        elif re.match(r'^[a-z][a-z0-9]*$', t):
            snake_case_count += 1
        elif re.match(r'^[A-Z][a-z0-9]*([A-Z][a-z0-9]*)+$', t) and not t.isupper():
            pascal_case_count += 1
        elif re.match(r'^[A-Z][a-z0-9]+$', t):
            pascal_case_count += 1
        elif re.match(r'^[A-Z][A-Z0-9_]*$', t):
            screaming_snake_count += 1

    camel_case_ratio = camel_case_count / n_ids
    snake_case_ratio = snake_case_count / n_ids
    pascal_case_ratio = pascal_case_count / n_ids
    screaming_snake_ratio = screaming_snake_count / n_ids

    # Identifier length stats
    short_id_count = sum(1 for t in identifiers if len(t) <= 2)
    long_id_count = sum(1 for t in identifiers if len(t) >= 5)
    short_id_ratio = short_id_count / n_ids
    long_id_ratio = long_id_count / n_ids
    avg_id_len = np.mean([len(t) for t in identifiers]) if identifiers else 0.0

    all_features = {
        'avg_line_len': float(avg_line_len),
        'std_line_len': float(std_line_len),
        'avg_indent': float(avg_indent),
        'std_indent': float(std_indent),
        'max_indent': float(max_indent),
        'empty_gap_std': float(empty_gap_std),
        'empty_gap_mean': float(empty_gap_mean),
        
        # New comment features
        'n_single_comments': float(n_single_comments),
        'n_block_comments': float(n_block_comments),
        'avg_comment_len': float(avg_comment_len),
        'comment_ratio': float(comment_ratio),
        'comment_density': float(comment_density),
        'text_comment_ratio': float(text_comment_ratio),
        'code_comment_ratio': float(code_comment_ratio),
        
        # New variable convention features
        'camel_case_ratio': float(camel_case_ratio),
        'snake_case_ratio': float(snake_case_ratio),
        'pascal_case_ratio': float(pascal_case_ratio),
        'screaming_snake_ratio': float(screaming_snake_ratio),
        'short_id_ratio': float(short_id_ratio),
        'long_id_ratio': float(long_id_ratio),
        'avg_id_len': float(avg_id_len),
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
from asyncio import subprocess
import gc
from itertools import repeat
from tqdm.notebook import tqdm
import os
import numpy as np
import pandas as pd
import polars as pl
from concurrent.futures import ProcessPoolExecutor
import kagglehub

import re
import math
from collections import Counter

K_PATH = kagglehub.competition_download("sem-eval-2026-task-13-subtask-a")

def extract_features(code, feature_mask=None):
    """Extract language-agnostic handcrafted features from code."""
    lines = code.split('\n')
    n_lines = max(len(lines), 1)
    n_chars = max(len(code), 1)

    # Line-level features
    line_lengths = [len(l) for l in lines]
    avg_line_len = np.mean(line_lengths)
    std_line_len = np.std(line_lengths) if len(line_lengths) > 1 else 0
    max_line_len = max(line_lengths)

    # Indentation features
    indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    avg_indent = np.mean(indents) if indents else 0
    std_indent = np.std(indents) if len(indents) > 1 else 0
    max_indent = max(indents) if indents else 0
    max_nesting_depth = max_indent // 4 if max_indent > 0 else 0

    # Whitespace ratios
    n_spaces = code.count(' ')
    n_tabs = code.count('\t')
    space_ratio = n_spaces / n_chars
    tab_ratio = n_tabs / n_chars

    # Bracket/paren counts (removed n_open_bracket, n_close_bracket)
    n_open_paren = code.count('(')
    n_close_paren = code.count(')')
    n_open_brace = code.count('{')
    n_close_brace = code.count('}')

    # Keyword densities (language-agnostic)
    tokens = re.findall(r'\b\w+\b', code.lower())
    n_tokens = max(len(tokens), 1)

    common_keywords = {'if', 'else', 'for', 'while', 'return', 'def', 'class',
                       'function', 'var', 'let', 'const', 'int', 'string',
                       'import', 'from', 'include', 'using', 'new', 'public',
                       'private', 'static', 'void', 'true', 'false', 'null',
                       'try', 'catch', 'throw', 'switch', 'case', 'break', 'continue'}
    keyword_count = sum(1 for t in tokens if t in common_keywords)
    keyword_ratio = keyword_count / n_tokens

    # Comment indicators
    comment_lines = sum(1 for l in lines if l.strip().startswith(('//', '#', '/*', '*')))
    comment_ratio = comment_lines / n_lines

    # Empty line ratio
    empty_lines = sum(1 for l in lines if not l.strip())
    empty_ratio = empty_lines / n_lines

    # Character entropy
    char_counts = Counter(code)
    total = sum(char_counts.values())
    entropy = -sum((c/total) * math.log2(c/total) for c in char_counts.values() if c > 0)

    # Token uniqueness
    unique_tokens = len(set(tokens))
    token_unique_ratio = unique_tokens / n_tokens

    # Identifier length stats (non-keyword tokens)
    identifiers = [t for t in tokens if t not in common_keywords and not t.isdigit()]
    avg_id_len = np.mean([len(t) for t in identifiers]) if identifiers else 0

    # Semicolons, colons
    n_semicolons = code.count(';') / n_lines
    n_colons = code.count(':') / n_lines

    # Code length features (removed log_code_len)
    code_len = len(code)

    # Single-character identifier ratio
    single_char_ids = sum(1 for t in identifiers if len(t) == 1)
    single_char_ratio = single_char_ids / len(identifiers) if identifiers else 0

    # Spaces around operators (removed operator_density)
    operators = re.findall(r'[+\-*/%=<>!&|^~]', code)
    spaced_operators = len(re.findall(r'\s[+\-*/%=<>!&|^~]\s', code))
    operator_spacing_ratio = spaced_operators / (len(operators) + 1) if operators else 0

    # Average line variability
    line_len_variance = np.var(line_lengths) if len(line_lengths) > 1 else 0

    # Comment verbosity
    comment_line_texts = [l.strip() for l in lines if l.strip().startswith(('//', '#', '/*', '*'))]
    if comment_line_texts:
        comment_word_counts = [len(re.findall(r'\b\w+\b', cl)) for cl in comment_line_texts]
        avg_comment_words = np.mean(comment_word_counts)
        max_comment_words = max(comment_word_counts)
    else:
        avg_comment_words = 0.0
        max_comment_words = 0.0

    # Function definition density
    func_keywords = {'def', 'function', 'void', 'func', 'fn', 'sub', 'proc'}
    func_def_count = sum(1 for t in tokens if t in func_keywords)
    func_def_density = func_def_count / n_lines

    # Long identifier ratio
    long_id_count = sum(1 for t in identifiers if len(t) > 15)
    long_id_ratio = long_id_count / len(identifiers) if identifiers else 0

    # Duplicate line ratio
    non_empty_lines = [l.rstrip() for l in lines if l.strip()]
    n_non_empty = max(len(non_empty_lines), 1)
    duplicate_lines = n_non_empty - len(set(non_empty_lines))
    duplicate_line_ratio = duplicate_lines / n_non_empty

    # Max consecutive empty lines
    max_consec_empty = 0
    cur_consec = 0
    for l in lines:
        if not l.strip():
            cur_consec += 1
            max_consec_empty = max(max_consec_empty, cur_consec)
        else:
            cur_consec = 0

    # Token bigram repetition ratio
    if len(tokens) > 1:
        bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]
        bigram_counts = Counter(bigrams)
        repeated_bigrams = sum(1 for c in bigram_counts.values() if c > 1)
        bigram_repetition_ratio = repeated_bigrams / len(bigram_counts)
    else:
        bigram_repetition_ratio = 0.0

    # Import/include density (removed numeric_density)
    import_keywords = {'import', 'from', 'include', 'require', 'using', 'use', 'extern'}
    import_count = sum(1 for t in tokens if t in import_keywords)
    import_density = import_count / n_lines

    # Bracket imbalance
    paren_imbalance = abs(code.count('(') - code.count(')'))
    brace_imbalance = abs(code.count('{') - code.count('}'))
    bracket_imbalance_val = abs(code.count('[') - code.count(']'))
    total_bracket_imbalance = paren_imbalance + brace_imbalance + bracket_imbalance_val

    # Line length percentiles (removed line_len_p75)
    line_len_p25 = float(np.percentile(line_lengths, 25))
    line_len_iqr = float(np.percentile(line_lengths, 75)) - line_len_p25

    # Indentation delta entropy
    all_line_indents = [len(l) - len(l.lstrip()) for l in lines]
    indent_deltas = [abs(all_line_indents[i+1] - all_line_indents[i])
                     for i in range(len(all_line_indents) - 1)]
    if indent_deltas:
        delta_counts = Counter(indent_deltas)
        delta_total = sum(delta_counts.values())
        indent_delta_entropy = -sum((c/delta_total) * math.log2(c/delta_total)
                                     for c in delta_counts.values() if c > 0)
        most_common_delta_ratio = delta_counts.most_common(1)[0][1] / delta_total
    else:
        indent_delta_entropy = 0.0
        most_common_delta_ratio = 0.0

    # Empty line gap consistency
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

    # Identifier length std dev
    id_len_std = float(np.std([len(t) for t in identifiers])) if len(identifiers) > 1 else 0.0

    # Line length autocorrelation (lag-1)
    if len(line_lengths) > 2:
        ll = np.array(line_lengths, dtype=float)
        x = ll[:-1]
        y = ll[1:]

        if x.std() > 0 and y.std() > 0:
            lag1_autocorr = float(np.corrcoef(x, y)[0, 1])
        else:
            lag1_autocorr = 0.0
    else:
        lag1_autocorr = 0.0

    # Exact-multiple indentation ratio
    exact_indent_lines = sum(1 for i in all_line_indents if i % 4 == 0)
    exact_indent_ratio = exact_indent_lines / n_lines

    # Type annotation / hint density
    type_hint_count = len(re.findall(r':\s*\w+|<\w+>|->\s*\w+', code))
    type_hint_density = type_hint_count / n_lines

    all_features = {
        'n_lines': n_lines,
        'avg_line_len': avg_line_len,
        'std_line_len': std_line_len,
        'max_line_len': max_line_len,
        'avg_indent': avg_indent,
        'std_indent': std_indent,
        'max_indent': max_indent,
        'space_ratio': space_ratio,
        'tab_ratio': tab_ratio,
        'n_open_paren': n_open_paren,
        'n_close_paren': n_close_paren,
        'n_open_brace': n_open_brace,
        'n_close_brace': n_close_brace,
        'keyword_ratio': keyword_ratio,
        'comment_ratio': comment_ratio,
        'empty_ratio': empty_ratio,
        'entropy': entropy,
        'token_unique_ratio': token_unique_ratio,
        'avg_id_len': avg_id_len,
        'n_semicolons': n_semicolons,
        'n_colons': n_colons,
        'code_len': code_len,
        'n_tokens': n_tokens,
        'unique_tokens': unique_tokens,
        'max_nesting_depth': max_nesting_depth,
        'single_char_ratio': single_char_ratio,
        'operator_spacing_ratio': operator_spacing_ratio,
        'line_len_variance': line_len_variance,
        'avg_comment_words': avg_comment_words,
        'max_comment_words': max_comment_words,
        'func_def_density': func_def_density,
        'long_id_ratio': long_id_ratio,
        'duplicate_line_ratio': duplicate_line_ratio,
        'max_consec_empty': max_consec_empty,
        'bigram_repetition_ratio': bigram_repetition_ratio,
        'import_density': import_density,
        'total_bracket_imbalance': total_bracket_imbalance,
        'line_len_p25': line_len_p25,
        'line_len_iqr': line_len_iqr,
        'indent_delta_entropy': indent_delta_entropy,
        'most_common_delta_ratio': most_common_delta_ratio,
        'empty_gap_std': empty_gap_std,
        'empty_gap_mean': empty_gap_mean,
        'id_len_std': id_len_std,
        'lag1_autocorr': lag1_autocorr,
        'exact_indent_ratio': exact_indent_ratio,
        'type_hint_density': type_hint_density,
    }

    # Nếu không truyền mask, tự động sử dụng danh sách mặc định của bạn
    if feature_mask is None:
        feature_mask = [
            'n_lines', 'avg_line_len', 'std_line_len', 'max_line_len',
            'avg_indent', 'std_indent', 'max_indent', 'space_ratio', 'tab_ratio',
            'keyword_ratio', 'comment_ratio', 'empty_ratio', 'entropy',
            'token_unique_ratio', 'avg_id_len', 'code_len', 'n_tokens',
            'unique_tokens', 'max_nesting_depth', 'single_char_ratio',
            'operator_spacing_ratio', 'line_len_variance', 'avg_comment_words',
            'max_comment_words', 'func_def_density', 'long_id_ratio',
            'duplicate_line_ratio', 'max_consec_empty', 'bigram_repetition_ratio',
            'total_bracket_imbalance', 'line_len_p25', 'line_len_iqr',
            'indent_delta_entropy', 'most_common_delta_ratio', 'empty_gap_std',
            'empty_gap_mean', 'id_len_std', 'lag1_autocorr'
        ]

    # Lọc và trả về danh sách giá trị theo đúng thứ tự của mask
    return [all_features[feature] for feature in feature_mask if feature in all_features]

FEATURE_NAMES = [
    'n_lines', 
    'avg_line_len', 
    'std_line_len', 
    # 'max_line_len',
    'avg_indent', 
    'std_indent', 
    'max_indent',
    'space_ratio', 
    'tab_ratio',
    # 'n_open_paren',
    # 'n_close_paren',
    # 'n_open_brace',
    # 'n_close_brace',
    'keyword_ratio',
    'comment_ratio',
    'empty_ratio',
    'entropy',
    'token_unique_ratio',
    'avg_id_len',
    # 'n_semicolons',
    # 'n_colons',
    'code_len',
    'n_tokens',
    'unique_tokens',
    'max_nesting_depth',
    'single_char_ratio',
    'operator_spacing_ratio',
    'line_len_variance',
    'avg_comment_words',
    'max_comment_words',
    'func_def_density',
    'long_id_ratio',
    'duplicate_line_ratio',
    'max_consec_empty',
    'bigram_repetition_ratio',
    # 'import_density',
    # 'total_bracket_imbalance',
    'line_len_p25',
    'line_len_iqr',
    'indent_delta_entropy',
    'most_common_delta_ratio',
    'empty_gap_std',
    'empty_gap_mean',
    'id_len_std',
    'lag1_autocorr',
    # 'exact_indent_ratio',
    # 'type_hint_density',
]

# print('~'*70)
# print('Handcraft Features defined!')
# print('~'*70)

# !ls {K_PATH}/Task_A
file_dict = {
    'test_sample' : f'{K_PATH}/Task_A/test_sample.parquet',
    'validation'  : f'{K_PATH}/Task_A/validation.parquet',
    'test'        : f'{K_PATH}/Task_A/test.parquet',
    'train'       : f'{K_PATH}/Task_A/train.parquet',
}

# print('~'*70)
# for k, v in file_dict.items():
#     filename = ': ' + v.replace(K_PATH + '/', '')
#     print(f"{k} {filename:>30}")
# print('~'*70)


TRAIN_FEATURES_PATH = 'train_features.parquet'
TEST_FEATURES_PATH = 'test_features.parquet'

# print('~'*70)
# print('Reading train_df and test_df from parquet...')
# Assuming file_dict is defined previously in your notebook
train_df = pl.read_parquet(file_dict['train'])
test_df = pl.read_parquet(file_dict['test'])
# print('~'*70)

args = {
    'seed'        : 123456,
    'n_positives' : 109921,#'n_positives' : 1099,

    'if_n_estimator'    : 400,
    'if_max_samples'    : 'auto',
    'if_contamination'  : 'auto',

    'cnb_alpha'         : 1 / 47,
    'cnb_force_alpha'   : False,
    'cnb_fit_prior'     : True,
    'cnb_class_prior'   : None,
    'cnb_norm'          : True,
}

def submit(filename):
    import subprocess

    # Định nghĩa lệnh cần chạy dưới dạng một list các chuỗi
    command = [
        "kaggle", 
        "competitions", 
        "submit", 
        "-c", "sem-eval-2026-task-13-subtask-a", 
        "-f", f"{filename}", 
        "-m", filename.replace('.csv', '')
    ]

    try:
        # Chạy lệnh
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # In ra kết quả thành công từ Kaggle
        print("Nộp bài thành công!")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        # Bắt lỗi nếu lệnh CLI thất bại (ví dụ: sai tên file, chưa authenticate)
        print("Có lỗi xảy ra khi nộp bài:")
        print(e.stderr)

if __name__ == "__main__":
    for i in range(1):
    # for feature_to_drop in FEATURE_NAMES:
    # for feature_to_drop in ["comment_ratio", "keyword_ratio", "std_line_len"]:
    # Tạo mask mới: Lấy tất cả trừ đặc trưng đang cần bỏ
        # current_mask = [f for f in FEATURE_NAMES if f != feature_to_drop]
        current_mask = FEATURE_NAMES
        # In ra để kiểm tra hoặc đưa vào model để test độ chính xác
        # if os.path.exists(f"submission_drop_{feature_to_drop}.csv"):
        #     print(f"File 'submission_drop_{feature_to_drop}.csv' đã tồn tại, bỏ qua tính toán và nộp bài...")
        #     continue
        
        # print(f"Đã bỏ: '{feature_to_drop}'")
        
        os.remove("anomoly.parquet") if os.path.exists("anomoly.parquet") else None
        os.remove(TRAIN_FEATURES_PATH) if os.path.exists(TRAIN_FEATURES_PATH) else None
        os.remove(TEST_FEATURES_PATH) if os.path.exists(TEST_FEATURES_PATH) else None

        
        if os.path.exists(TRAIN_FEATURES_PATH) and os.path.exists(TEST_FEATURES_PATH):
            print(f'Loading pre-extracted features from {TRAIN_FEATURES_PATH} and {TEST_FEATURES_PATH}...')
            X_hand_train_pd = pd.read_parquet(TRAIN_FEATURES_PATH)
            X_hand_train = X_hand_train_pd.values
            y_train = train_df['label'].to_numpy()

            X_hand_test_pd = pd.read_parquet(TEST_FEATURES_PATH)
            X_hand_test = X_hand_test_pd.values
            print('~'*70)
            print(f'Handcrafted features: train={X_hand_train.shape}')
            print(f'Handcrafted features: test={X_hand_test.shape}')
            print('~'*70)
        else:
            print(f'Extracting {len(current_mask)} handcrafted features using multiprocessing...')
            print('~'*70)
            
            # Determine the number of workers. Leaves 1 core free to prevent UI freeze.
            max_workers = 8
            print(f'Starting ProcessPoolExecutor with {max_workers} workers...')

            train_codes = train_df['code'].to_list()
            test_codes = test_df['code'].to_list()

            # The chunksize parameter helps chunk the iterable into batches, 
            # which is much faster than sending elements to workers one by one.
            optimal_chunksize = 100

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Wrap executor.map in list() and tqdm() to evaluate the generator and show progress
                X_hand_train = np.array(list(
                    tqdm(executor.map(extract_features, train_codes, repeat(current_mask), chunksize=optimal_chunksize), 
                        total=len(train_codes), 
                        desc="Extracting Train")
                ))
            
            y_train = train_df['label'].to_numpy()
            print(f'Handcrafted features: train={X_hand_train.shape}')

            # Recalculate chunksize for test set in case sizes differ vastly
            optimal_chunksize_test = max(1, len(test_codes) // (max_workers * 4))

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                X_hand_test = np.array(list(
                    tqdm(executor.map(extract_features, test_codes, repeat(current_mask), chunksize=optimal_chunksize_test), 
                        total=len(test_codes), 
                        desc="Extracting Test")
                ))

            print(f'Handcrafted features: test={X_hand_test.shape}')
            print('~'*70)

            # Save the extracted features for future runs
            print(f'Saving extracted features to {TRAIN_FEATURES_PATH} and {TEST_FEATURES_PATH}...')
            pd.DataFrame(X_hand_train, columns=current_mask).to_parquet(TRAIN_FEATURES_PATH, index=False)
            pd.DataFrame(X_hand_test, columns=current_mask).to_parquet(TEST_FEATURES_PATH, index=False)
            print('~'*70)
        
        
        from sklearn.preprocessing import QuantileTransformer

        # print('~'*70)
        quant = QuantileTransformer(output_distribution="uniform", n_quantiles=1000,
                                    random_state=args['seed'])
        # print('Preprocessing train...')
        quant.fit(X_hand_train)
        X_train = quant.transform(X_hand_train)
        # print('X_train shape:', X_train.shape, ' ' * 8 + 'y_train_shape:', y_train.shape)
        # print('~'*70)
        X_human = X_train[y_train == 0]
        X_machine = X_train[y_train == 1]
        
        from sklearn.ensemble import IsolationForest
        
        if_human   = IsolationForest(n_estimators=args['if_n_estimator'], max_samples=args['if_max_samples'], contamination=args['if_contamination'],
                                    max_features=1.0, bootstrap=False, n_jobs=-1,
                                    random_state=args['seed'], warm_start=False)

        if_machine = IsolationForest(n_estimators=args['if_n_estimator'], max_samples=args['if_max_samples'], contamination=args['if_contamination'],
                                    max_features=1.0, bootstrap=False, n_jobs=-1,
                                    random_state=args['seed'], warm_start=False)

        gc.collect()
        
        print('Fitting if_human...')
        if_human.fit(X_human)
        print('Fitting if_machine...')
        if_machine.fit(X_machine)
        
        import os

        ANOMALY_PATH = 'anomoly.parquet'

        print('~'*70)
        print('Calculating anomaly_score...')

        def scale_anomaly(raw: np.ndarray) -> np.ndarray:
            lo, hi = raw.min(), raw.max()
            return np.zeros_like(raw) if hi == lo else 1.0 - (raw - lo) / (hi - lo)

        def bi_anomoly(X_hand, quantiler, forest_1, forest_2):
            X_hand_q = quantiler.transform(X_hand)
            s_1 = scale_anomaly(forest_1.decision_function(X_hand_q))
            s_2 = scale_anomaly(forest_2.decision_function(X_hand_q))
            return np.column_stack((s_1, s_2))

        if os.path.exists(ANOMALY_PATH):
            print(f'Loading pre-calculated anomaly scores from {ANOMALY_PATH}...')
            X_2d_train = pd.read_parquet(ANOMALY_PATH).values
        else:
            print('Calculating scores from scratch...')
            X_2d_train = bi_anomoly(X_hand_train, quant, if_human, if_machine)
            print(f'Saving scores to {ANOMALY_PATH}...')
            pd.DataFrame(X_2d_train, columns=['human_anomaly', 'machine_anomaly']).to_parquet(ANOMALY_PATH, index=False)

        print('~'*70)
        print('X_2d_train shape:', X_2d_train.shape)
        print('~'*70)
        print(X_2d_train[:5])
        print('~'*70)

        gc.collect()
        
        print('~'*70)
        print('Training Classifier : QuadraticDiscriminantAnalysis...')
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

        def get_proba(X_hand, classifier):
            score_2d = bi_anomoly(X_hand, quant, if_human, if_machine)
            X_2d = np.array(list(score_2d))
            return classifier.predict_proba(X_2d)[:, 1]

        # Instantiate QDA
        clf = QuadraticDiscriminantAnalysis()

        print('~'*70)
        print('Model ready! now training...')

        clf.fit(X_2d_train, y_train)

        print('~'*70)
        
        # test_df, X_test and y_test is loaded above, so no need to load them again
        print('~'*70)
        print('Calculating threshold...')
        predictions = get_proba(X_hand_test, clf)

        # Calculate the threshold based on args['n_positives']
        if args['n_positives'] <= 0:
            threshold = np.max(predictions) + 1.0 # If 0 positives needed, threshold is higher than any prediction
        elif args['n_positives'] >= len(predictions):
            threshold = np.min(predictions) - 1.0 # If all positives needed, threshold is lower than any prediction
        else:
            # Sort predictions in descending order to find the (args['n_positives'])-th highest score
            sorted_predictions = np.sort(predictions)[::-1]
            threshold = sorted_predictions[args['n_positives'] - 1]

        print(f"Calculated threshold: {threshold:.6f}")
        print('~'*70)

        gc.collect()
        
        print('~'*70)
        print('Pending predictions...')

        def pr_2_label(scores):
            return [(1 if score > threshold else 0) for score in scores]
            # return [(1 if score > 0.9 else 0) for score in scores]
            
        # Apply the threshold and perform random tie-breaking for final labels
        final_labels = np.zeros_like(predictions, dtype=int)

        # Identify items strictly above the threshold
        strictly_above_indices = np.where(predictions > threshold)[0]
        final_labels[strictly_above_indices] = 1

        num_strictly_above = len(strictly_above_indices)
        num_needed_from_ties = args['n_positives'] - num_strictly_above

        # if num_needed_from_ties > 0:
        #     # Identify items equal to the threshold
        #     equal_indices = np.where(predictions == threshold)[0]
        #     if len(equal_indices) > 0:
        #         rng = np.random.default_rng(args['seed']) # Use seed for reproducibility
        #         selected_from_ties_indices = rng.choice(equal_indices, size=num_needed_from_ties, replace=False)
        #         final_labels[selected_from_ties_indices] = 1

        print(f"Number of predicted positives after thresholding and random tie-breaking: {np.sum(final_labels)}")
        print('~'*70)
        
        print('~'*70)
        print('Exporting predictions...')
        test_df = test_df.with_columns(pl.Series(name='label', values=final_labels))

        test_df.select('ID', 'label').write_csv(f'submission_drop_{feature_to_drop}.csv')
        print('Done!')
        print('~'*70)

        gc.collect()
        
        submit(f'submission_drop_{feature_to_drop}.csv')

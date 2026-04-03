# Dataset Description

## Files
train.parquet - the training set
test.parquet - the test set (currently only the public part is released, rest will be uploaded in December 2025 / January 2026)
sample_submission.csv - a sample submission file in the correct format (majority class prediction)
validation.parquet - validation set (same domain as train)
test_sample.parquet - 1000 samples from real test set with corresponding labels. It is given for evaluation purposes ONLY and is matching the original Public leaderboard. This data will not be used for final evaluation and is shared just as a sample

# Columns
code - program code, you are expected to use it for prediction
generator - the model which generated the code (or 'human' if it is human-written)
language - programming language of the code
label - either 0 (human) or 1 (AI)
# Overview
In this subtask, participants are asked to identify whether a given code snippet is (i) fully human-written or (ii) fully machine-generated. While the task is a binary classification problem, it is designed with a strong focus on out-of-distribution (OOD) generalization.

# Motivation

Our previous studies (study 1, study 2) highlight a major limitations of existing AI-content detectors: they often break down in OOD conditions. This task is designed to directly address that gap.

# Description
Challenge Design To rigorously test generalization, our evaluation spans two axes: programming languages and domains.

Seen languages (training): C++, Python, Java
Unseen languages (test): Go, PHP, C#, C, JavaScript
Seen domain: Algorithmic code
Unseen domains: Research code and generic deployed code
Evaluation Settings The test set covers four complementary scenarios:

Seen languages, seen domains – test code matches training distribution.
Unseen languages, seen domains – new languages, but familiar domains.
Seen languages, unseen domains – known languages, but different domains.
Unseen languages and unseen domains – both language and domain differ, requiring robust generalization.
Evaluation
Submission Format

Submit a .csv file with two columns:

ID: unique identifier of the code snippet
label: the label ID (not the string label)
Evaluation measure The primary evaluation measure is macro F1-score

# Citation
Daniil Orel, Dilshod Azizov, Indraneil Paul, and Yuxia Wang. SemEval-2026-Task13-Subtask-A. https://kaggle.com/competitions/sem-eval-2026-task-13-subtask-a, 2025. Kaggle.
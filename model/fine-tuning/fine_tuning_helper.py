from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import torch 
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import Dataset
from sklearn.model_selection import train_test_split
import evaluate
import json


model_checkpoint = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def epfl_course_split_dataset(dataset):
    """
    Function to split the epfl course dataset into questions, answers, and courses.

    Args:
        dataset: list of dictionaries with keys "question", "answer", and "course"

    Returns:
        questions: list of questions
        answers: list of answers
        courses: list of courses 
    """
    questions = []
    answers = []
    courses = []

    for data in dataset:
        questions.append(data["question"])
        answers.append(data["answer"])
        courses.append(data["course"])

    return questions, answers, courses


def split_test_dataset(dataset):
    """
    Function to split the test dataset into questions and answers.

    Args:
        dataset: list of dictionaries with keys "question" and "answers"

    Returns:
        questions: list of questions
        answers: list of answers (list of lists)
    """
    questions = []
    answers = []

    for data in dataset:
        questions.append(data["question"])
        answers.append(data["answers"])

    return questions, answers


def create_prompting_inputs(questions, courses, answers, tokenizer):
    """
    Function to create the prompting inputs for the model.

    Args:
        questions: list containing the questions of the dataset
        courses: list containing the courses of the dataset
        answers: list containing the answers of the dataset
        tokenizer: tokenizer 

    Returns:
        inputs: list containing the prompting inputs
        labels: list containing the labels
    """
    inputs = []
    labels = []
    for i in range(len(questions)):
        # promting taken from the template of flan T5
        prompt = "Question: " + questions[i] + "Answer: "
        inputs.append(prompt)
        labels.append(answers[i])

    return inputs, labels


def create_prompting_inputs_test_dataset(questions, answers, tokenizer):
    """
    Function to create the prompting inputs for the model for the test dataset.

    Args:
        questions: list containing the questions of the test dataset
        answers: list containing the answers of the test dataset
        tokenizer: tokenizer 

    Returns:
        inputs: list containing the prompting inputs
        labels: list containing the labels
    """
    inputs = []
    labels = []
    for i in range(len(questions)):
        # promting taken from the template of flan T5
        prompt = "Question:\n" + questions[i] + "\nAnswer: "
        inputs.append(prompt)
        labels.append(answers[i])

    return inputs, labels


def mask_padtokens_labels(labels, tokenizer):
    """
    Function to mask the padding tokens in the labels.

    Args:
        labels: tensor of labels
        tokenizer: tokenizer 

    Returns:
        labels_masked: tensor of labels with padding tokens masked
    """
    labels_masked = labels
    # maks token -> -100
    labels_masked[labels == tokenizer.pad_token_id] = -100

    return labels_masked


def stem_split_dataset(dataset, test_size=0.2, random_state=7):
    """
    Function to split the stem dataset into questions, answers, and courses.

    Args:
        dataset: list of dictionaries with keys "question", "answer", and "subject"

    Returns:
        questions: list of questions
        answers: list of answers
        courses: list of courses 
    """
    questions = []
    answers = []
    courses = []

    for data in dataset:
        questions.append(data["question"])
        answers.append(data["answer"])
        courses.append(data["subject"])

    return questions, answers, courses



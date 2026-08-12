import torch # type:ignore

def preprocess_arc(samples,tokenizer,max_length):
    batch_size = len(samples['question'])
    inputs, targets = [], []

    for q, choices, answer_key in zip(samples['question'], samples['choices'], samples['answerKey']):
        choice_texts = [f"({label}) {text}" for label, text in zip(choices['label'], choices['text'])]
        formatted_choices = "\n".join(choice_texts)
        input_text = f"Question: {q}\n{formatted_choices}\n\nAnswer:"
        inputs.append(input_text)

        if answer_key in choices["label"]:
            correct_choice_index = choices["label"].index(answer_key)
            correct_choice_text = choices["text"][correct_choice_index]
        else:
            correct_choice_text = ""

        targets.append(correct_choice_text)

    tokenizer.padding_side = "left"  # safe for causal LM

    model_inputs = tokenizer(inputs, add_special_tokens=True, truncation=False)

    labels = tokenizer(targets, add_special_tokens=False, truncation=False)


    batch_input_ids, batch_labels, batch_attention_mask = [], [], []

    for input_ids, label_ids in zip(model_inputs["input_ids"], labels["input_ids"]):
        total_length = len(input_ids) + len(label_ids) + 1
        if total_length > max_length:
            continue

        if len(label_ids) == 0:
            label_ids = [tokenizer.eos_token_id]
        else:
            label_ids.append(tokenizer.eos_token_id)

        input_with_label = input_ids + label_ids
        attention_mask = [1] * len(input_with_label)
        labels_ids_padded = [-100] * len(input_ids) + label_ids

        padding_length = max_length - len(input_with_label)

        input_with_label = [tokenizer.pad_token_id] * padding_length + input_with_label
        attention_mask = [0] * padding_length + attention_mask
        labels_ids_padded = [-100] * padding_length + labels_ids_padded

        batch_input_ids.append(input_with_label[-max_length:])
        batch_attention_mask.append(attention_mask[-max_length:])
        batch_labels.append(labels_ids_padded[-max_length:])

    return {
        "input_ids": torch.tensor(batch_input_ids),
        "attention_mask": torch.tensor(batch_attention_mask),
        "labels": torch.tensor(batch_labels),
    }

TARGET_MAPPING = {
    0: "no",
    1: "yes",
}

def preprocess_boolq(samples, tokenizer, max_length):
    inputs = [f"{p}\nQuestion: {q}?\nAnswer:" for p, q in zip(samples['passage'], samples['question'])]
    targets = [TARGET_MAPPING[label] for label in samples['answer']]

    model_inputs = tokenizer(inputs, add_special_tokens=True, truncation=False) # leave room for label and eos
    labels = tokenizer(targets, add_special_tokens=False)

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for input_ids, label_ids in zip(model_inputs["input_ids"], labels["input_ids"]):
        total_length = len(input_ids) + len(label_ids) + 1  # +1 for eos_token_id
        if total_length > max_length:
            continue

        label_ids = label_ids + [tokenizer.eos_token_id]

        combined_input = input_ids + label_ids
        combined_attention_mask = [1] * len(combined_input)

        combined_labels = [-100] * len(input_ids) + label_ids

        padding_length = max_length - len(combined_input)

        padded_input_ids = [tokenizer.pad_token_id] * padding_length + combined_input
        padded_attention_mask = [0] * padding_length + combined_attention_mask
        padded_labels = [-100] * padding_length + combined_labels

        batch_input_ids.append(padded_input_ids[:max_length])
        batch_attention_mask.append(padded_attention_mask[:max_length])
        batch_labels.append(padded_labels[:max_length])

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
    }

def preprocess_winogrande(samples, tokenizer, max_length):
    all_input_texts = []
    all_labels = []

    for s, a, b, ans in zip(samples['sentence'], samples['option1'], samples['option2'], samples['answer']):
        idx = s.index("_")
        option_sentences = [
            s[:idx] + a + s[idx + 1:],
            s[:idx] + b + s[idx + 1:]
        ]

        correct_option = int(ans) - 1

        for i, sentence in enumerate(option_sentences):
            prompt = "Complete the sentence:\n"
            # prompt = "Answer true or false: \n"
            # prompt = ""
            input_text = prompt + sentence
            all_input_texts.append(input_text)

            all_labels.append(i == correct_option)

    tokenized = tokenizer(all_input_texts, add_special_tokens=True, truncation=False)

    batch_input_ids, batch_labels, batch_attention_mask = [], [], []

    for input_ids, is_correct in zip(tokenized["input_ids"], all_labels):
        if len(input_ids) > max_length:
            continue
        if is_correct:
            labels = input_ids.copy()
        else:
            continue # loss is zero in this case which biases sample mean loss
            labels = [-100] * len(input_ids)

        padding_length = max_length - len(input_ids)

        # LEFT padding
        input_ids = [tokenizer.pad_token_id] * padding_length + input_ids
        labels = [-100] * padding_length + labels
        attention_mask = [0] * padding_length + [1] * (len(input_ids) - padding_length)

        batch_input_ids.append(input_ids[-max_length:])
        batch_attention_mask.append(attention_mask[-max_length:])
        batch_labels.append(labels[-max_length:])

    return {
        "input_ids": torch.tensor(batch_input_ids),
        "attention_mask": torch.tensor(batch_attention_mask),
        "labels": torch.tensor(batch_labels),
    }

def preprocess_gsm8k(samples, tokenizer, max_length):
    # intended for openai/gsm8k as testing data (and metamath as training data)
    inputs=[f"Question: \n{q}\nAnswer:" for q in samples['question']]
    targets=[f"{a}" for a in samples['answer']]


    model_inputs = tokenizer(inputs, add_special_tokens=True, truncation=False)
    labels = tokenizer(targets, add_special_tokens=False, truncation=False)

    batch_input_ids, batch_labels, batch_attention_mask = [], [], []

    for input_ids, label_ids in zip(model_inputs["input_ids"], labels["input_ids"]):

        total_length = len(input_ids) + len(label_ids) + 1  # +1 for eos_token_id

        if total_length > max_length:
            continue

        label_ids.append(tokenizer.eos_token_id)

        input_with_label = input_ids + label_ids
        attention_mask = [1] * len(input_with_label)
        labels_ids_padded = [-100] * len(input_ids) + label_ids

        padding_length = max_length - len(input_with_label)

        input_with_label = [tokenizer.pad_token_id] * padding_length + input_with_label
        attention_mask = [0] * padding_length + attention_mask
        labels_ids_padded = [-100] * padding_length + labels_ids_padded

        batch_input_ids.append(input_with_label)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels_ids_padded)

    return {
        "input_ids": torch.tensor(batch_input_ids),
        "attention_mask": torch.tensor(batch_attention_mask),
        "labels": torch.tensor(batch_labels),
    }

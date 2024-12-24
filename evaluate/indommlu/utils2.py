import torch
import torch.nn.functional as F
import numpy as np


def softmax(x):
    """Numpy softmax function to ensure consistent behavior."""
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    return numerator / denominator


@torch.no_grad()
def get_logprobs_causal(model, tokenizer, prompt, device):
    """Get log probabilities for causal language models."""
    # Prepare inputs and move to the specified device
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Remove token_type_ids for certain model types
    if model.config.model_type == 'falcon':
        inputs.pop("token_type_ids", None)
    
    # Ensure all tensors are on the same device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Prepare input and output ids
    input_ids = inputs["input_ids"]
    output_ids = input_ids[:, 1:]
    
    # Compute model outputs
    outputs = model(**inputs, labels=input_ids)

    # Ensure logits are on the correct device and converted to double
    logits = outputs.logits.to(torch.double).to(device)
    output_ids = output_ids.to(device)
    
    # Compute log probabilities
    logprobs = torch.gather(F.log_softmax(logits, dim=2), 2, output_ids.unsqueeze(2))
    
    return logprobs.mean()


def predict_classification_causal(model, tokenizer, input_text, labels, device):
    """Predict classification probabilities for causal models."""
    probs = [get_logprobs_causal(model, tokenizer, input_text + label, device) for label in labels]
    return probs


def predict_classification_causal_by_letter(model, tokenizer, input_text, labels, device):
    """Predict classification by letter for causal models."""
    choices = ['A', 'B', 'C', 'D', 'E'][:len(labels)]
    choice_ids = [tokenizer.encode(choice)[-1] for choice in choices]
    
    with torch.no_grad():
        # Prepare inputs and move to device
        inputs = tokenizer(input_text, return_tensors="pt")
        if model.config.model_type == 'falcon':
            inputs.pop("token_type_ids", None)
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Compute model outputs
        outputs = model(**inputs, labels=inputs["input_ids"])
        
        # Process logits
        last_token_logits = outputs.logits[:, -1, :]
        choice_logits = last_token_logits[:, choice_ids].detach().cpu().numpy()
        
        # Compute probabilities
        conf = softmax(choice_logits[0])
        pred = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}[np.argmax(choice_logits[0])]
    
    return conf, pred


@torch.no_grad()
def get_logprobs_mt0(model, tokenizer, prompt, device, label_ids=None, label_attn=None):
    """Get log probabilities for MT0 models."""
    # Prepare inputs and move to device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    # Ensure label_ids and label_attn are on the correct device
    label_ids = label_ids.to(device)
    label_attn = label_attn.to(device)
    
    # Compute model outputs
    outputs = model(**inputs, decoder_input_ids=model._shift_right(label_ids))
    logits = outputs.logits

    # Compute log probabilities
    logprobs = torch.gather(F.log_softmax(logits, dim=2), 2, label_ids.unsqueeze(2)) * label_attn.unsqueeze(2)
    return logprobs.sum() / label_attn.sum()


def predict_classification_mt0(model, tokenizer, input_text, labels, device):
    """Predict classification probabilities for MT0 models."""
    # Encode labels and move to device
    labels_encoded = tokenizer(labels, add_special_tokens=False, padding=True, return_tensors='pt')
    list_label_ids = labels_encoded['input_ids'].to(device)
    list_label_attn = labels_encoded['attention_mask'].to(device)
    
    # Compute probabilities
    probs = [
        get_logprobs_mt0(model, tokenizer, input_text, device, label_ids.view(1,-1), label_attn.view(1,-1)) 
        for (label_ids, label_attn) in zip(list_label_ids, list_label_attn)
    ]
    return probs


def predict_classification_mt0_by_letter(model, tokenizer, input_text, labels, device):
    """Predict classification by letter for MT0 models."""
    choices = ['A', 'B', 'C', 'D', 'E'][:len(labels)]
    choice_ids = [tokenizer.encode(choice)[0] for choice in choices]
    
    with torch.no_grad():
        # Prepare start token and move to device
        start_token = tokenizer('<pad>', return_tensors="pt").to(device)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        # Compute model outputs
        outputs = model(**inputs, decoder_input_ids=start_token['input_ids'])
        last_token_logits = outputs.logits[:, -1, :]
        
        # Process logits
        choice_logits = last_token_logits[:, choice_ids].detach().cpu().numpy()
        conf = softmax(choice_logits[0])
        pred = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}[np.argmax(choice_logits[0])]

    return conf, pred
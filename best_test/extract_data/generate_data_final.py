import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer


embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")




def save_checkpoint(batch_idx, results, checkpoint_path="processing_checkpoint.json"):
    """Save current processing state to checkpoint file"""
    checkpoint_data = {
        'last_completed_batch': batch_idx,
        'total_processed_rows': len(results),
        'processed_seq_ids': [r['seq_id'] for r in results]
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    print(f"Checkpoint saved: processed {len(results)} rows up to batch {batch_idx + 1}")

def load_checkpoint(checkpoint_path="processing_checkpoint.json"):
    """Load checkpoint if exists"""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return None

def load_existing_results(save_path):
    """Load existing results from CSV if exists"""
    if os.path.exists(save_path):
        try:
            df = pd.read_csv(save_path)
            results = df.to_dict('records')
            print(f"Loaded {len(results)} existing results from {save_path}")
            return results
        except Exception as e:
            print(f"Error loading existing results: {e}")
            return []
    return []




def extract_comprehensive_uncertainty_features(output, tokenizer) -> Dict[str, float]:
    features = {}
    if not hasattr(output, 'scores') or not output.scores:
        return {f: 0.0 for f in get_feature_names()}
    
    logits_list = [score[0] for score in output.scores]  
    probs_list = [F.softmax(logits, dim=-1) for logits in logits_list]

    
    entropies = [-(probs * torch.log(probs + 1e-8)).sum().item() for probs in probs_list]
    features['mean_entropy'] = float(np.mean(entropies))
    features['max_entropy'] = float(np.max(entropies))
    features['min_entropy'] = float(np.min(entropies))
    features['std_entropy'] = float(np.std(entropies))

    
    max_probs = [torch.max(probs).item() for probs in probs_list]
    features['mean_max_prob'] = float(np.mean(max_probs))
    features['min_max_prob'] = float(np.min(max_probs))
    features['std_max_prob'] = float(np.std(max_probs))

    max_possible_entropy = torch.log(torch.tensor(probs_list[0].shape[-1])).item()
    confidences = [1 - (ent / max_possible_entropy) for ent in entropies]
    features['mean_confidence'] = float(np.mean(confidences))
    features['min_confidence'] = float(np.min(confidences))

    
    top2_ratios, margins, residual5, residual10 = [], [], [], []
    for probs in probs_list:
        sorted_probs, _ = torch.sort(probs, descending=True)
        if len(sorted_probs) >= 2:
            top2_ratios.append((sorted_probs[0] / (sorted_probs[1] + 1e-8)).item())
            margins.append((sorted_probs[0] - sorted_probs[1]).item())
        residual5.append((1.0 - sorted_probs[:5].sum()).item())
        residual10.append((1.0 - sorted_probs[:10].sum()).item())

    features['mean_top2_ratio'] = float(np.mean(top2_ratios)) if top2_ratios else 0.0
    features['mean_margin'] = float(np.mean(margins)) if margins else 0.0
    features['residual_top5'] = float(np.mean(residual5))
    features['residual_top10'] = float(np.mean(residual10))

    
    features['sequence_length'] = len(output.scores)
    features['prob_variance'] = float(np.var(max_probs))
    features['entropy_variance'] = float(np.var(entropies))
    features['entropy_range'] = float(np.max(entropies) - np.min(entropies))
    features['perplexity'] = float(np.exp(features['mean_entropy']))

    
    sharpness_scores = []
    for probs in probs_list:
        sorted_probs = torch.sort(probs, descending=True)[0]
        n = len(sorted_probs)
        index = torch.arange(1, n + 1, dtype=torch.float32, device=probs.device)
        gini = (2 * torch.sum(index * sorted_probs)) / (n * torch.sum(sorted_probs)) - (n + 1) / n
        sharpness_scores.append(gini.item())
    features['mean_sharpness'] = float(np.mean(sharpness_scores))

    
    if hasattr(output, 'sequences'):
        generated_tokens = output.sequences[0]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        features['has_syntax_keywords'] = float(any(k in generated_text.lower() for k in ['def','class','if','for','while','return']))
        features['has_parentheses'] = float('(' in generated_text and ')' in generated_text)
        features['has_brackets'] = float('[' in generated_text and ']' in generated_text)
        features['has_braces'] = float('{' in generated_text and '}' in generated_text)
        features['indentation_consistency'] = calculate_indentation_consistency(generated_text)
        words = generated_text.split()
        features['word_repetition_ratio'] = 1.0 - (len(set(words)) / len(words)) if words else 0.0
    else:
        features.update({
            'has_syntax_keywords': 0.0, 'has_parentheses': 0.0,
            'has_brackets': 0.0, 'has_braces': 0.0,
            'indentation_consistency': 0.0, 'word_repetition_ratio': 0.0
        })

    
    features['attention_entropy'] = 0.0
    features['attention_variance'] = 0.0

    return features

def calculate_indentation_consistency(text: str) -> float:
    lines = text.split('\n')
    indentations = []
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            indentations.append(indent)
    if len(indentations) < 2:
        return 1.0
    if all(indent % 4 == 0 for indent in indentations):
        return 1.0
    elif all(indent % 2 == 0 for indent in indentations):
        return 0.8
    else:
        return 0.5

def get_feature_names() -> List[str]:
    return [
        'mean_entropy','max_entropy','min_entropy','std_entropy',
        'mean_max_prob','min_max_prob','std_max_prob',
        'mean_confidence','min_confidence',
        'mean_top2_ratio','mean_margin','residual_top5','residual_top10',
        'sequence_length','perplexity','prob_variance','entropy_variance','entropy_range','mean_sharpness',
        'has_syntax_keywords','has_parentheses','has_brackets','has_braces',
        'indentation_consistency','word_repetition_ratio',
        'attention_entropy','attention_variance',
        
        'emb_mean','emb_std','emb_min','emb_max'
    ]




def extract_embedding_features(instruction: str) -> Dict[str, float]:
    emb = embedder.encode(instruction, convert_to_numpy=True, normalize_embeddings=True)
    return {
        "emb_mean": float(np.mean(emb)),
        "emb_std": float(np.std(emb)),
        "emb_min": float(np.min(emb)),
        "emb_max": float(np.max(emb)),
    }




def process_dataset_with_enhanced_features(csv_path, model, tokenizer, save_path="enhanced_router_dataset.csv", 
                                         batch_size=3, checkpoint_path="processing_checkpoint.json", 
                                         resume=True):
    """
    Process dataset with resume functionality
    
    Args:
        csv_path: Input CSV file path
        model: The model to use for generation
        tokenizer: The tokenizer
        save_path: Output CSV file path
        batch_size: Number of rows to process per batch
        checkpoint_path: Path to store checkpoint information
        resume: Whether to resume from checkpoint if available
    """
    df = pd.read_csv(csv_path)
    
    
    results = []
    start_batch = 0
    processed_seq_ids = set()
    
    if resume:
        
        results = load_existing_results(save_path)
        processed_seq_ids = {r['seq_id'] for r in results}
        
        
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            start_batch = checkpoint['last_completed_batch'] + 1
            print(f"Resuming from batch {start_batch + 1}")
            print(f"Already processed {len(processed_seq_ids)} rows")
        elif results:
            
            print(f"Found existing results but no checkpoint. Determining restart point...")
            
            for batch_idx in range((len(df) + batch_size - 1) // batch_size):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, len(df))
                batch_seq_ids = set(df.iloc[batch_start:batch_end]['seq_id'])
                
                
                if not batch_seq_ids.issubset(processed_seq_ids):
                    start_batch = batch_idx
                    break
            else:
                
                print("All data already processed!")
                return pd.DataFrame(results)
    
    total_batches = (len(df) + batch_size - 1) // batch_size
    print(f"Processing batches {start_batch + 1} to {total_batches} (total: {total_batches - start_batch} batches)")

    try:
        for batch_idx in range(start_batch, total_batches):
            print(f"\n--- Processing batch {batch_idx + 1}/{total_batches} ---")
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]

            batch_results = []
            for idx, row in batch_df.iterrows():
                
                if row['seq_id'] in processed_seq_ids:
                    print(f"Skipping seq_id {row['seq_id']} (already processed)")
                    continue
                
                print(f"Processing seq_id: {row['seq_id']} ({idx + 1}/{len(df)})")
                
                try:
                    
                    messages = [
                        {"role": "system", "content": "You are an expert Python programmer. Write only the function code that solves the problem exactly as specified. Ensure your solution passes all test cases. Use efficient algorithms and handle all edge cases correctly."},
                        {"role": "user", "content": f"Write a Python function named '{row['entry_point']}' that solves this problem:\n\n{row['instruction']}\n\nIMPORTANT: The function must be named exactly '{row['entry_point']}' - do not use any other name."}
                    ]
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer(prompt, return_tensors="pt")
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) for k,v in inputs.items()}

                    with torch.no_grad():
                        output = model.generate(
                            **inputs, max_new_tokens=500, return_dict_in_generate=True,
                            output_scores=True, do_sample=True, temperature=0.7, top_p=0.9,
                            pad_token_id=tokenizer.eos_token_id
                        )

                    generated_text = tokenizer.decode(output.sequences[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    generated_code = extract_code_from_response(generated_text)

                    
                    passes_tests = execute_code_with_tests(generated_code, row['entry_point'], row['testcase'])
                    label = 0 if passes_tests else 1

                    
                    uncertainty_features = extract_comprehensive_uncertainty_features(output, tokenizer)
                    embedding_features = extract_embedding_features(row['instruction'])

                    result = {
                        'seq_id': row['seq_id'],
                        'instruction': row['instruction'],
                        'output': row['output'],
                        'code': row['code'],
                        'entry_point': row['entry_point'],
                        'testcase': row['testcase'],
                        'formatted_prompt': prompt,
                        'generated_code': generated_code,
                        'generated_text': generated_text,
                        'passes_tests': passes_tests,
                        'label': label,
                    }
                    result.update(uncertainty_features)
                    result.update(embedding_features)

                    batch_results.append(result)
                    processed_seq_ids.add(row['seq_id'])

                except Exception as e:
                    print(f"Error processing seq_id {row['seq_id']}: {e}")
                    result = {col: 0.0 for col in get_feature_names()}
                    result.update({
                        'seq_id': row['seq_id'], 'instruction': row['instruction'],
                        'output': row['output'], 'code': row['code'],
                        'entry_point': row['entry_point'], 'testcase': row['testcase'],
                        'formatted_prompt': '', 'generated_code': '', 'generated_text': '',
                        'passes_tests': False, 'label': 1,
                    })
                    batch_results.append(result)
                    processed_seq_ids.add(row['seq_id'])

            
            results.extend(batch_results)

            
            if batch_results:  
                
                save_checkpoint(batch_idx, results, checkpoint_path)
                
                
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(save_path, index=False)
                print(f"Progress saved: {len(results)} total rows processed")

    except KeyboardInterrupt:
        print("\n--- Processing interrupted by user ---")
        print(f"Saving progress... Processed {len(results)} rows")
        if results:
            final_df = pd.DataFrame(results)
            final_df.to_csv(save_path, index=False)
            print(f"Results saved to {save_path}")
        return pd.DataFrame(results) if results else None
    
    except Exception as e:
        print(f"Error during processing: {e}")
        print(f"Saving progress... Processed {len(results)} rows")
        if results:
            final_df = pd.DataFrame(results)
            final_df.to_csv(save_path, index=False)
            print(f"Results saved to {save_path}")
        raise e

    
    final_df = pd.DataFrame(results)
    final_df.to_csv(save_path, index=False)
    
    
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Processing completed successfully. Checkpoint file removed.")
    
    print(f"\n--- Processing Complete ---")
    print(f"Total rows processed: {len(results)}")
    print(f"Results saved to: {save_path}")
    
    return final_df




def extract_code_from_response(response: str) -> str:
    if "```python" in response:
        start = response.find("```python") + 9
        end = response.find("```", start)
        if end != -1: return response[start:end].strip()
    if "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        if end != -1: return response[start:end].strip()
    lines, code_lines, in_function = response.split('\n'), [], False
    for line in lines:
        if line.strip().startswith('def '): in_function = True
        if in_function: code_lines.append(line)
    return '\n'.join(code_lines).strip() if code_lines else response.strip()

def execute_code_with_tests(code: str, entry_point: str, testcase: Any) -> bool:
    try:
        namespace = {}
        exec(code, namespace)
        if isinstance(testcase, str):
            testcase = testcase.strip('[]')
            test_cases, current_test, paren_count = [], "", 0
            i = 0
            while i < len(testcase):
                char = testcase[i]; current_test += char
                if char == '(': paren_count += 1
                elif char == ')': paren_count -= 1
                elif char == ',' and paren_count == 0:
                    if current_test.strip():
                        test_cases.append(current_test.rstrip(',').strip())
                        current_test = ""
                i += 1
            if current_test.strip(): test_cases.append(current_test.strip())
        for test in test_cases:
            if test.startswith('assert'):
                exec(test, namespace)
        return True
    except Exception: return False


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    model_name = "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit"
    

    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",         
        quantization_config=bnb_config
    )

    
    enhanced_dataset = process_dataset_with_enhanced_features(
        csv_path="C:\\Users\\s\\Desktop\\Dev\\SamsungProject\\extract\\code_dataset_10k.csv",
        model=model,  
        tokenizer=tokenizer,
        save_path="enhanced_router_dataset_emb4.csv",
        batch_size=3,  
        checkpoint_path="processing_checkpoint.json",  
        resume=True  
    )
import pandas as pd
import json
import requests
import time
from typing import List, Dict, Tuple
import ast

class DataProcessor:
    def __init__(self, ollama_base_url="http://localhost:11434"):
        self.ollama_base_url = ollama_base_url
        
    def parse_data_line(self, line: str) -> Dict:
        """Parse a single line of your data format"""
        parts = line.strip().split(',')
        if len(parts) < 6:
            return None
            
        return {
            'id': parts[0],
            'instruction': parts[1],
            'description': parts[2], 
            'code': parts[3],
            'function_name': parts[4],
            'tests': parts[5]  # This contains the assert statements
        }
    
    def load_data(self, file_path: str) -> List[Dict]:
        """Load and parse your data file"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parsed = self.parse_data_line(line)
                if parsed:
                    data.append(parsed)
        return data
    
    def call_ollama(self, model_name: str, prompt: str) -> str:
        """Call Ollama API for inference"""
        url = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                print(f"Error calling Ollama: {response.status_code}")
                return ""
        except Exception as e:
            print(f"Exception calling Ollama: {e}")
            return ""
    
    def extract_code_from_response(self, response: str) -> str:
        """Extract Python code from model response"""
        # Look for code blocks
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        # If no code blocks, try to find function definition
        lines = response.split('\n')
        code_lines = []
        in_function = False
        
        for line in lines:
            if line.strip().startswith('def '):
                in_function = True
                code_lines.append(line)
            elif in_function:
                if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    break
                code_lines.append(line)
        
        return '\n'.join(code_lines) if code_lines else response.strip()
    
    def test_code_solution(self, code: str, tests: str, function_name: str) -> bool:
        """Test if the generated code passes the test cases"""
        try:
            # Create a safe execution environment
            exec_globals = {}
            exec_locals = {}
            
            # Execute the code
            exec(code, exec_globals, exec_locals)
            
            # Check if function exists
            if function_name not in exec_locals:
                return False
            
            # Parse and execute test cases
            test_lines = tests.strip('[]').split(', assert ')
            test_lines = [t.strip() for t in test_lines]
            test_lines = [t[7:] if t.startswith('assert ') else t for t in test_lines]
            
            for test in test_lines:
                if test.strip():
                    try:
                        # Execute the test
                        result = eval(test, exec_globals, exec_locals)
                        if not result:
                            return False
                    except:
                        return False
            
            return True
        except Exception as e:
            print(f"Error testing code: {e}")
            return False
    
    def generate_model_comparison_data(self, data: List[Dict], 
                                     model_7b: str = "qwen2.5-coder:7b", 
                                     model_32b: str = "qwen2.5-coder:32b") -> List[Dict]:
        """Generate comparison data between 7B and 32B models"""
        comparison_data = []
        
        for i, item in enumerate(data):
            print(f"Processing item {i+1}/{len(data)}: {item['id']}")
            
            instruction = item['instruction']
            function_name = item['function_name']
            tests = item['tests']
            
            # Create prompt for both models
            prompt = f"""Write a Python function to solve this problem:

{instruction}

Please provide only the function implementation."""
            
            # Get responses from both models
            response_7b = self.call_ollama(model_7b, prompt)
            response_32b = self.call_ollama(model_32b, prompt)
            
            # Extract code
            code_7b = self.extract_code_from_response(response_7b)
            code_32b = self.extract_code_from_response(response_32b)
            
            # Test both solutions
            pass_7b = self.test_code_solution(code_7b, tests, function_name)
            pass_32b = self.test_code_solution(code_32b, tests, function_name)
            
            # Determine preference (which model should be used)
            if pass_32b and not pass_7b:
                preference = "32b"  # 32B is better
            elif pass_7b and not pass_32b:
                preference = "7b"   # 7B is sufficient
            elif pass_7b and pass_32b:
                preference = "7b"   # Both work, prefer cheaper 7B
            else:
                preference = "32b"  # Neither works well, try 32B
            
            comparison_data.append({
                'id': item['id'],
                'instruction': instruction,
                'function_name': function_name,
                'tests': tests,
                'code_7b': code_7b,
                'code_32b': code_32b,
                'pass_7b': pass_7b,
                'pass_32b': pass_32b,
                'preference': preference,
                'features': self.extract_features(instruction)
            })
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.1)
        
        return comparison_data
    
    def extract_features(self, instruction: str) -> Dict:
        """Extract features from instruction for router training"""
        instruction_lower = instruction.lower()
        
        features = {
            'length': len(instruction.split()),
            'has_algorithm': any(word in instruction_lower for word in 
                               ['algorithm', 'sort', 'search', 'optimize', 'complex']),
            'has_data_structure': any(word in instruction_lower for word in 
                                    ['array', 'list', 'dict', 'tree', 'graph', 'hash']),
            'has_math': any(word in instruction_lower for word in 
                           ['calculate', 'math', 'number', 'sum', 'count']),
            'has_string': any(word in instruction_lower for word in 
                            ['string', 'text', 'character', 'word']),
            'complexity_indicators': len([word for word in 
                                        ['nested', 'recursive', 'dynamic', 'optimization', 'efficient'] 
                                        if word in instruction_lower])
        }
        
        return features
    
    def save_comparison_data(self, data: List[Dict], output_path: str):
        """Save the comparison data to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} comparison examples to {output_path}")

if __name__ == "__main__":
    processor = DataProcessor()
    
    # Load your data
    print("Loading data...")
    data = processor.load_data("C:\\Users\\s\\Desktop\\Dev\\SamsungProject\\extract\\top_30_by_code_token_length.csv")  # Replace with your actual file path
    print(f"Loaded {len(data)} examples")
    
    # Process first 100 examples for testing (adjust as needed)
    sample_data = data[:30]
    
    # Generate comparison data
    print("Generating model comparison data...")
    comparison_data = processor.generate_model_comparison_data(sample_data)
    
    # Save results
    processor.save_comparison_data(comparison_data, "router_training_data.json")
    
    # Print statistics
    preferences = [item['preference'] for item in comparison_data]
    print(f"\nPreference distribution:")
    print(f"7B preferred: {preferences.count('7b')}")
    print(f"32B preferred: {preferences.count('32b')}")
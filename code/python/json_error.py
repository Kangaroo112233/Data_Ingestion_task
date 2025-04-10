import re
import json
from typing import Any, Dict, Union

class LLMJsonParser:
    """
    A robust parser for handling malformed JSON from LLM responses,
    particularly focused on issues from Llama 3.3 70B.
    """
    
    @staticmethod
    def fix_missing_colons(json_str: str) -> str:
        """Fix instances where field names are missing colons before their values."""
        # Pattern to find field names without colons (e.g., "name" something instead of "name": something)
        pattern = r'"([^"]+)"\s+(?=["{\[\w])'
        replacement = r'"\1": '
        return re.sub(pattern, replacement, json_str)
    
    @staticmethod
    def fix_unquoted_field_values(json_str: str) -> str:
        """Fix instances where string values are not properly quoted."""
        # This is a simplified approach - a more comprehensive solution would need
        # to handle nested structures carefully
        pattern = r':\s*([^",\{\}\[\]\s][^",\{\}\[\]\n]*[^",\{\}\[\]\s,])\s*([,\}\]]|$)'
        replacement = r': "\1"\2'
        return re.sub(pattern, replacement, json_str)
    
    @staticmethod
    def fix_trailing_commas(json_str: str) -> str:
        """Remove trailing commas in arrays and objects."""
        # Fix trailing commas in objects
        json_str = re.sub(r',\s*}', '}', json_str)
        # Fix trailing commas in arrays
        json_str = re.sub(r',\s*\]', ']', json_str)
        return json_str
    
    @staticmethod
    def fix_unclosed_structures(json_str: str) -> str:
        """Attempt to fix unclosed braces, brackets, and quotes."""
        # Count opening and closing braces/brackets
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        # Add missing closing braces/brackets
        json_str = json_str.rstrip()
        while close_braces < open_braces:
            json_str += '}'
            close_braces += 1
        while close_brackets < open_brackets:
            json_str += ']'
            close_brackets += 1
            
        return json_str
    
    @staticmethod
    def fix_duplicate_field_names(json_str: str) -> str:
        """
        Fix duplicate field names by adding a suffix.
        This is a simplistic approach; in practice, you might want to merge values or handle differently.
        """
        # This is a complex problem to solve with regex alone
        # For now, we'll use a placeholder solution that processes the JSON as a string
        # A more robust solution would require parsing the JSON structure
        
        seen_fields = {}
        lines = json_str.split('\n')
        result = []
        
        for line in lines:
            # Simple pattern to find field names in the current line
            matches = re.findall(r'"([^"]+)"\s*:', line)
            for field in matches:
                if field in seen_fields:
                    seen_fields[field] += 1
                    # Replace the duplicate field name with a suffixed version
                    new_field = f'{field}_{seen_fields[field]}'
                    line = line.replace(f'"{field}":', f'"{new_field}":', 1)
                else:
                    seen_fields[field] = 0
            
            result.append(line)
        
        return '\n'.join(result)
    
    @staticmethod
    def parse(json_str: str, max_attempts: int = 3) -> Union[Dict[str, Any], Any]:
        """
        Try to parse the JSON string with progressive fixing attempts.
        
        Args:
            json_str: The potentially malformed JSON string to parse
            max_attempts: Maximum number of repair attempts before giving up
            
        Returns:
            Parsed JSON object or raises the final exception
        """
        original_json = json_str
        attempts = 0
        
        # Define repair functions to try in sequence
        repair_functions = [
            LLMJsonParser.fix_missing_colons,
            LLMJsonParser.fix_unquoted_field_values,
            LLMJsonParser.fix_trailing_commas,
            LLMJsonParser.fix_unclosed_structures,
            LLMJsonParser.fix_duplicate_field_names
        ]
        
        # First, try parsing as-is
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Initial parsing failed: {str(e)}")
        
        # Try each repair function individually
        for repair_func in repair_functions:
            if attempts >= max_attempts:
                break
                
            try:
                fixed_json = repair_func(json_str)
                if fixed_json != json_str:  # Only count as attempt if something changed
                    attempts += 1
                    json_str = fixed_json
                    result = json.loads(json_str)
                    print(f"Successfully fixed JSON with {repair_func.__name__}")
                    return result
            except json.JSONDecodeError:
                continue
        
        # If individual repairs didn't work, try applying all of them in sequence
        if attempts < max_attempts:
            try:
                for repair_func in repair_functions:
                    json_str = repair_func(json_str)
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                pass
        
        # If we still can't parse it, try the hjson library if available
        try:
            import hjson
            return hjson.loads(original_json)
        except (ImportError, Exception) as e:
            print(f"hjson fallback failed or not available: {str(e)}")
        
        # Last resort: Try using a regex-based approach to extract a partial JSON object
        try:
            # Find the first complete JSON object in the text
            match = re.search(r'\{.*\}', original_json, re.DOTALL)
            if match:
                possible_json = match.group(0)
                return json.loads(possible_json)
        except json.JSONDecodeError:
            pass
            
        # If all else fails, raise the original exception
        raise json.JSONDecodeError(f"Failed to parse JSON after {attempts} repair attempts", original_json, 0)


# Example usage:
def parse_llm_response(response_text: str) -> Any:
    """Parse an LLM response that should contain JSON."""
    try:
        return LLMJsonParser.parse(response_text)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        return None


# Example of how to use it with your Llama 3.3 70B responses:
if __name__ == "__main__":
    # Example of a malformed JSON from an LLM
    llm_response = '''
    {
      "query": "What is the capital of France?",
      "answer": "The capital of France is Paris",
      "confidence" 0.95,
      "sources": [
        "geography database",
        "world facts"
      ],
      "generated_at" "2023-10-15T12:34:56Z",
      "model_version": "llama-3.3-70b"
    }
    '''
    
    parsed_data = parse_llm_response(llm_response)
    if parsed_data:
        print("Successfully parsed JSON:")
        print(json.dumps(parsed_data, indent=2))

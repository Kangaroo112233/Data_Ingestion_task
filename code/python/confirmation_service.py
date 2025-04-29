DOCUMENT_CONFIRMATION_PROMPT = """You are a Document Confirmation model. Your task is to determine whether a document belongs to a specific customer based on the provided System of Record (SoR).

Use semantic similarity (not exact matching). Consider shortened names and partial addresses as valid. If any conflict is found, reject.

Return ONLY a JSON with a single key having the value 'yes' or 'no':
{{
    "decision": "yes" or "no"
}}
"""



elif doc_label == "document_confirmation":
    system_prompt = DOCUMENT_CONFIRMATION_PROMPT
    calibration_model = None  # If no calibration model used


prompt = get_prompt(tokenizer, system_prompt=system_prompt, message=full_text, user_role="document")
tokenized_input = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    outputs = model.generate(
        input_ids=tokenized_input.input_ids,
        attention_mask=tokenized_input.attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        max_length=model.config.max_position_embeddings,
        output_logits=True,
        return_dict_in_generate=True,
        top_p=0.95,
        temperature=1e-3,
    )

response_text = get_response_from_output(tokenizer, outputs).replace("\n", "").strip()


# Fix JSON formatting issues
result = response_text.split("{", 1)[-1]
if len(result.rsplit("}", 1)) == 1:
    result += '}'
else:
    result = result.rsplit("}", 1)[0] + '}'

result = json.loads("{" + result)

output = [{
    "type": "classification",
    "prediction": {
        "decision": result["decision"]
    }
}]


extraction_output(model, tokenizer, pages, doc_label="document_confirmation", traceId=..., batchGuid=...)



pip install flask



from flask import Flask, request, jsonify
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM  # replace with your actual model class
from model import get_prompt, get_response_from_output

# Flask app init
app = Flask(__name__)

# Load model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained("your-model-name")  # replace with your model
model = AutoModelForCausalLM.from_pretrained("your-model-name").to("cuda" if torch.cuda.is_available() else "cpu")

DOCUMENT_CONFIRMATION_PROMPT = """You are a Document Confirmation model. Your task is to determine whether a document belongs to a specific customer based on the provided System of Record (SoR).

Use semantic similarity (not exact matching). Consider shortened names and partial addresses as valid. If any conflict is found, reject.

Return ONLY a JSON with a single key having the value 'yes' or 'no':
{
    "decision": "yes" or "no"
}
"""

@app.route('/confirm-document', methods=['POST'])
def confirm_document():
    try:
        data = request.json
        full_text = data["full_text"]
        sor_info = f"Name: {data['sor_name']}, Address: {data['sor_address']}"

        full_prompt = f"Document:\n{full_text}\n\nSystem of Record:\n{sor_info}\n\n{DOCUMENT_CONFIRMATION_PROMPT}"

        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                max_length=model.config.max_position_embeddings,
                return_dict_in_generate=True,
                output_logits=True,
                top_p=0.95,
                temperature=1e-3,
            )

        response = get_response_from_output(tokenizer, outputs).strip()

        # JSON fix
        result = response.split("{", 1)[-1]
        if len(result.rsplit("}", 1)) == 1:
            result += '}'
        else:
            result = result.rsplit("}", 1)[0] + '}'

        result = json.loads("{" + result)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)



{
  "full_text": "Document content goes here...",
  "sor_name": "John Doe",
  "sor_address": "1234 Main St, Springfield, IL"
}


# run the server


python app.py


curl -X POST http://localhost:5000/confirm-document \
     -H "Content-Type: application/json" \
     -d '{"full_text": "some text...", "sor_name": "John Doe", "sor_address": "1234 Main St"}'

from transformers import AutoTokenizer

# 1) load your tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 2) character count
df1_with_text["num_chars"] = df1_with_text["text"].str.len()

# 3) word count (naïvely split on whitespace)
df1_with_text["num_words"] = df1_with_text["text"].str.split().str.len()

# 4) token count
#    – here we tokenize each string and count the length of input_ids
df1_with_text["num_tokens"] = (
    df1_with_text["text"]
    .fillna("")                        # avoid None
    .map(lambda txt: len(
        tokenizer.encode(txt, 
                           add_special_tokens=False, 
                           truncation=False)
    ))
)

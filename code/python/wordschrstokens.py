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



import pandas as pd

# 1) rename to df3’s schema
df1_with_text = df1_with_text.rename(columns={
    "billingCycleDate":         "statement_end_date",
    "paymentDueDate":           "due_date",
    "billingRecipientAddress":  "billing_recipient_address",
    "vendorAddress":            "vendor_address",
    "vendorName":               "vendor_name",
    "paymentAmount":            "payment_amount"
})

# 2) reorder to exactly df3’s columns
df1_with_text = df1_with_text[df3.columns]

# 3) stitch them together
df_full = pd.concat([df3, df1_with_text], ignore_index=True)

# check
print(df_full.shape)    # (148, 18)

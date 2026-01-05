import re
import string

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans(' ',' ',string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

sample = "I am santhoshi, I am earning 3 to 4 L per month in Remote job happly, thank you god   !!!!"
print(clean_text(sample))
import pandas as pd
import csv
# Load the two CSV files
file1 = pd.read_csv("./trainDataset.csv")
file2 = pd.read_csv("./transDataset_inducedSentencePair.csv")

original_sen_pair_codeText=list(file1['Text'])
original_sen_pair_translatedCodeText=list(file1['Translation'])

induced_sen_pair_codeText=list(file2['inducedCodeText'])
induced_sen_pair_translatedCodeText=list(file2['inducedTranslatedCodeText'])

combined_code_text = original_sen_pair_codeText+induced_sen_pair_codeText
combined_translated_code_text = original_sen_pair_translatedCodeText + induced_sen_pair_translatedCodeText
assert len(combined_code_text) == len(combined_translated_code_text)
with open('./combined_trainDataset.csv', 'w', encoding="utf_8_sig",
          newline="",
          errors="ignore") as csvfile:  # 
    fieldnames = ['Text', 'Translation']
    write = csv.DictWriter(csvfile, fieldnames=fieldnames)
    write.writeheader()  # 
    for i in range(len(combined_code_text)):
        if not pd.isna(combined_code_text[i]):
            write.writerow({'Text': combined_code_text[i],
                            'Translation': combined_translated_code_text[i]})

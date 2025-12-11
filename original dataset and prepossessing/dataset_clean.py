import csv
import json
import os

def process_goemotions():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    label_file = os.path.join(base_dir, 'emotions.txt')
    files_to_process = [
        ('train.tsv', 'train_cleaned.jsonl'),
        ('dev.tsv', 'dev_cleaned.jsonl'),
        ('test.tsv', 'test_cleaned.jsonl')
    ]

    print(f"📖 Reading label file: {label_file}")
    try:
        with open(label_file, 'r', encoding='utf-8') as f:
            emotion_map = [line.strip() for line in f.readlines()]
        
        print(f"✅ Successfully loaded {len(emotion_map)} labels.")
        print(f"   Mapping example: ID 0 -> {emotion_map[0]}, ID 27 -> {emotion_map[-1]}")
    except FileNotFoundError:
        print(f"❌ Critical Error: emotions.txt not found in path {base_dir}")
        print("Please ensure emotions.txt and the script are in the same folder!")
        return

    for input_name, output_name in files_to_process:
        input_path = os.path.join(base_dir, input_name)
        output_path = os.path.join(base_dir, output_name)
        
        if not os.path.exists(input_path):
            print(f"⚠️ Skipping: File {input_name} not found")
            continue
            
        print(f"🔄 Processing {input_name} -> {output_name} ...")
        
        count = 0
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            reader = csv.reader(f_in, delimiter='\t')
            
            for row in reader:
                if len(row) < 2:
                    continue
                
                text = row[0]
                ids_string = row[1]  
                
                if not ids_string:
                    continue
                
                try:
                    id_list = [int(i) for i in ids_string.split(',')]
                    
                    label_list = [emotion_map[i] for i in id_list]
                    
                    output_labels = ", ".join(label_list)
                    
                    json_line = {
                        "instruction": "Analyze the text and identify the emotions.",
                        "input": text,
                        "output": output_labels
                    }
                    
                    f_out.write(json.dumps(json_line, ensure_ascii=False) + "\n")
                    count += 1
                    
                except (ValueError, IndexError) as e:
                    print(f"   ⚠️ Data parsing error: {row} - {e}")
                    continue
                    
        print(f"   💾 Done. Wrote {count} records.")

if __name__ == "__main__":
    process_goemotions()
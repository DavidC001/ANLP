def get_subtree_span(token_id, deprel_dict):
    span = {token_id}
    stack = [token_id]
    while stack:
        current = stack.pop()
        if current in deprel_dict:
            for child in deprel_dict[current]:
                if child not in span:
                    span.add(child)
                    stack.append(child)
    return span

sent_number = 0
def extract_info_from_conllu(conllu_data):
    global sent_number
    sentences = conllu_data.strip().split("\n\n")
    result = []

    for sentence in sentences:
        lines = sentence.strip().split("\n")
        
        # Ensure we have the necessary lines to process
        sent_id_line = next((line for line in lines if line.startswith("# sent_id =")), None)
        text_line = next((line for line in lines if line.startswith("# text =")), None)
        
        if not sent_id_line:
            continue
        
        try:
            sent_id = sent_id_line.split(" = ")[1]
            true_text = text_line.split(" = ")[1]
        except IndexError:
            continue

        text = ""
        tokens = []
        deprel_dict = {}
        predicates = []
        roles = {}

        for line in lines:
            if not line.startswith("#"):
                columns = line.split("\t")
                if len(columns) >= 12:
                    try:
                        token_id = int(columns[0]) - 1  # Adjust to 0-based index
                    except ValueError:
                        continue  # Skip lines with decimal token IDs
                    word = columns[1]
                    text += word + " "
                    head = int(columns[6]) - 1  # Adjust to 0-based index
                    sense = columns[10]
                    wordRoles = columns[11:]
                    # breakpoint()
                    # remove \n
                    wordRoles[-1] = wordRoles[-1].replace("\n", "")

                    tokens.append((token_id, word))
                    
                    if head not in deprel_dict:
                        deprel_dict[head] = []
                    deprel_dict[head].append(token_id)

                    roles[token_id] = wordRoles

                    if "V" in wordRoles:
                        predicates.append((token_id, wordRoles.index("V"), sense))

        text = text.strip()

        for predicate in predicates:
            token_id, rel_position,sense = predicate
            rel = f"{token_id}:{token_id}-rel"
            args = []

            associated_roles = 0 # filter relations with no arguments
            for idx, wordRoles in sorted(roles.items()):
                if (idx != token_id) and (wordRoles[rel_position] != "_"):
                    span = get_subtree_span(idx, deprel_dict)
                    arg_span = f"{min(span)}:{max(span)}-{wordRoles[rel_position]}"
                    args.append(arg_span)
                    associated_roles += 1
            
            try:
                if associated_roles > 0:
                    result.append(f"{sent_id}\t{sent_number}\t{true_text}\t{text}\t{sense}\t{rel}\t{'\t'.join(args)}")
            except IndexError:
                continue
        
        sent_number += 1

    return result

def main():
    # Read the input files
    file_paths = [
        'datasets/UP-1.0/UP_English-EWT/en_ewt-up-dev.conllu',
        'datasets/UP-1.0/UP_English-EWT/en_ewt-up-test.conllu',
        'datasets/UP-1.0/UP_English-EWT/en_ewt-up-train.conllu'
    ]

    output_files = [
        'datasets/preprocessed/en_ewt-up-dev.tsv',
        'datasets/preprocessed/en_ewt-up-test.tsv',
        'datasets/preprocessed/en_ewt-up-train.tsv'
    ]

    for input_file, output_file in zip(file_paths, output_files):
        with open(input_file, 'r', encoding='utf-8') as f:
            conllu_data = f.read()
        
        transformed_data = extract_info_from_conllu(conllu_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in transformed_data:
                f.write(line + '\n')

    output_files

if __name__ == "__main__":
    main()
    print("Done!")
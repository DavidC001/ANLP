import nltk
from nltk.corpus import treebank
from nltk import Nonterminal, induce_pcfg
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from openai import OpenAI
# deepcopy
from copy import deepcopy
from numpy.random import choice

client = OpenAI(api_key="sk-proj-7qNdKPCibTLWB1E9UrIPT3BlbkFJZYwIFcPDF7WGrznU2bUM")

# Step 1: Extract productions from the Treebank corpus
productions = []
for item in treebank.fileids():
    for tree in treebank.parsed_sents(item):
        productions += tree.productions()

# Step 2: Induce PCFG from the productions
S = Nonterminal('S')
grammar = induce_pcfg(S, productions)

tags = """Tag Description
CC	Coordinating conjunction
CD	Cardinal number
DT	Determiner
EX	Existential there
FW	Foreign word
IN	Preposition or subordinating conjunction
JJ	Adjective
JJR	Adjective, comparative
JJS	Adjective, superlative
LS	List item marker
MD	Modal
NN	Noun, singular or mass
NNS	Noun, plural
NNP	Proper noun, singular
NNPS	Proper noun, plural
PDT	Predeterminer
POS	Possessive ending
PRP	Personal pronoun
PRP$	Possessive pronoun
RB	Adverb
RBR	Adverb, comparative
RBS	Adverb, superlative
RP	Particle
SYM	Symbol
TO	to
UH	Interjection
VB	Verb, base form
VBD	Verb, past tense
VBG	Verb, gerund or present participle
VBN	Verb, past participle
VBP	Verb, non-3rd person singular present
VBZ	Verb, 3rd person singular present
WDT	Wh-determiner
WP	Wh-pronoun
WP$	Possessive wh-pronoun
WRB	Wh-adverb
"""

# Step 4: Load the pre-trained LM model
API_KEY = "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"

login(token=API_KEY)

device = "cpu" # the device to load the model onto
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# Function to generate a terminal with constraints
def generate_terminal_with_LM(model, tokenizer, input_text='', top_k=10):
    messages = [
    {"role": "user", "content": "Complete the following sentences with a only one valid word terminal to form coherent statements, the requested word is tagged as \"ReqWord\". Ensure that the completed sentences make grammatical sense and follows the tree used to generated the phrase."},
    {"role": "assistant", "content": "I'll help you complete the sentences. Please provide the derivation tree structure."},
    {"role": "user", "content": "Tree: (S (NP-SBJ (NN Mary)) (VP (VB put) (NP (NNS (ReqWord ))) (SBAR-TMP )) (. ))\nCurrent Sentence: Mary put"},
    {"role": "assistant", "content": "valuables"},
    {"role": "user", "content": "Tree: (S (S (NP-SBJ (DT (ReqWord )) (NN ) (NN )) (VP )) (: ) (S ) (. ))\nCurrent Sentence: "},
    {"role": "assistant", "content": "The"},
    {"role": "user", "content": "Tree: (S (NP-SBJ (NNP John)) (VP (VBD gave) (NP (DT a) (NN (ReqWord ))) (PP (TO ) (NP (DT ) (NN )))) (. ))\nCurrent Sentence: John gave a"},
    {"role": "assistant", "content": "gift"},
    {"role": "user", "content": input_text}
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)
    model.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id, num_return_sequences=1, top_k=top_k, temperature=0.5)
    decoded = tokenizer.batch_decode(generated_ids)

    out = decoded[0].split("[/INST]")[-1].replace("</s>", "").strip()
    print("\t[LM]\n\t\tgenerated", out)
    
    return out

def get_chatGpt3_response(input_text=''):
    print("\t[ChatGPT3]\n\t\tRequesting GPT-3.5 Turbo for completion of sentence with terminal word\n\t\tInput:", input_text)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You generate ONE word at the time to fill in the requested word tagged as ReqWord. Ensure that the completed sentences make grammatical sense and the generated words follow the provided tag.\nHere is the list of possible tags:\n"+tags},
            {"role": "user", "content": "Tree: (S (NP-SBJ (NN Mary)) (VP (VB put) (NP (NNS (ReqWord ))) (SBAR-TMP )) (. ))\nCurrent Sentence: Mary put <NNS>"},
            {"role": "assistant", "content": "valuables"},
            {"role": "user", "content": "Tree: (S (S (NP-SBJ (DT (ReqWord )) (NN ) (NN )) (VP )) (: ) (S ) (. ))\nCurrent Sentence: <DT>"},
            {"role": "assistant", "content": "The"},
            {"role": "user", "content": "Tree: (S (NP-SBJ (NNP John)) (VP (VBD gave) (NP (DT a) (NN (ReqWord ))) (PP (TO ) (NP (DT ) (NN )))) (. ))\nCurrent Sentence: John gave a <NN>"},
            {"role": "assistant", "content": "gift"},
            {"role": "user", "content": input_text}
        ],
        stop=" ",
        temperature=0.3
    )
    print("\t\tResponse:", response.choices[0].message.content)
    return response.choices[0].message.content

def get_terminal_from_command(tree, current_sentence, symbol):
    print("[Require Terminal]\n\tTree:"+str(tree).replace("\n",""))
    print(tree.pretty_print())
    print("\tCurrent Sentence:", current_sentence)
    print("\tGrammar suggests:", symbol)
    out = input("Enter word: ")
    
    return out



# Function to recursively generate sentence using the grammar and LM for terminals
def generate_sentence(grammar, symbol, currTree, prev_symb="",current_sentence='', max_depth=5):
    print("[Gen Sent]\n\tdepth="+str(max_depth)+"\n\tSymbol: "+str(symbol)+"\n\tCurrent Tree:", str(currTree).replace("\n",""))
    #pretty print the tree
    print(currTree.pretty_print())
    #if symbol is a terminal, generate a terminal using a LM
    if not isinstance(symbol, Nonterminal):
        print("\t[Gen Term]\n\t\tGenerating Terminal for: ", str(currTree).replace("\n",""), "\n\t\tCurrent Sentence:", current_sentence, "\n\t\tsuggested symbol:", symbol)
        if str(symbol) in ['.', '?', '!', ',', "to", "``", "''", "'s", "there"]:
            terminal = str(symbol).lower()
        else:
            terminal = get_chatGpt3_response(f"Tree: {str(currTree)}\nCurrent Sentence: {current_sentence} <{str(prev_symb)}>")
            # terminal = generate_terminal_with_constraints(model, tokenizer, f"Tree: {str(currTree).replace("\n","")}\nCurrent Sentence: {current_sentence}")
            # terminal = get_terminal_from_command(currTree, current_sentence, symbol)
        #set terminal as a terminal node in the tree
        return current_sentence + ' ' + terminal, terminal
    # print("Current Sentence:", current_sentence, "Symbol:", symbol)
    production = grammar.productions(lhs=symbol)
    #sample using the production probabilities
    if max_depth == 0:
        #get only the productions to non-terminals
        production = [p for p in production if not isinstance(p.rhs()[0], Nonterminal)]
    if len(production) < 1: return current_sentence, "INVALID"

    valid_production = False
    retires = 0
    while not valid_production:
        new_current_sentence = current_sentence
        retires += 1
        if retires > 10: return current_sentence, "INVALID"
        p = choice(production, p=[p.prob() for p in production])
        children = []
        rhs_symbols_no_terminal = [nltk.Tree(str(rhs_symbol), []) if isinstance(rhs_symbol, Nonterminal) else "Terminal" for rhs_symbol in p.rhs()]
        for i, rhs_symbol in enumerate(p.rhs()):
            if str(rhs_symbol) == '-NONE-': 
                children = []
                break

            prevTree = deepcopy(currTree)
            if isinstance(rhs_symbol, Nonterminal):
                subTree = children+rhs_symbols_no_terminal[i:]
            else:
                subTree = children+[nltk.Tree("ReqWord",[])]+rhs_symbols_no_terminal[i+1:]
            print("\t[SubTree]\n\t\t", subTree)
            #filter only the nodes with no children
            for node in prevTree.subtrees(lambda t: len(t) == 0):
                print("\t\t[Node]\n\t\t\t", node, len(node))
                if str(symbol) == node.label():
                    print("\t\t[Append]\n\t\t\t", subTree)
                    #append to the node list the new production
                    node.extend(subTree)
                    break
                    
            new_current_sentence, tree = generate_sentence(grammar, rhs_symbol, prevTree, symbol, new_current_sentence, max_depth-1)
            if tree == "INVALID":
                children = []
                break
            children.append(tree)
            
        if len(children) > 0: valid_production = True
    
    parse_tree = nltk.Tree(str(symbol), children)
    current_sentence = new_current_sentence
    
    return current_sentence, parse_tree

def generate_sentence_no_Terminals(grammar, symbol, max_depth=5):
    if not isinstance(symbol, Nonterminal):
        term = "TERM" if str(symbol) not in ['.', '?', '!', ',', "to", "``", "''", "'s", "there"] else str(symbol).lower()
        return "TERM"
    if max_depth == 0:
        return "INVALID"
    production = grammar.productions(lhs=symbol)
    valid_production = False
    retries = 0
    while not valid_production:
        children = []
        if retries > 20: return "INVALID"
        retries += 1
        random_production = choice(production, p=[p.prob() for p in production])
        for rhs_symbol in random_production.rhs():
            if str(rhs_symbol) == '-NONE-':
                children = []
                break
            if isinstance(rhs_symbol, Nonterminal):
                tree =generate_sentence_no_Terminals(grammar, rhs_symbol, max_depth-1)
                if tree == "INVALID": 
                    children = []
                    break
                children.append(tree)
            else:
                term = "TERM" if str(symbol) not in ['.', '?', '!', ',', "to", "``", "''", "'s", "there"] else str(symbol).lower()
                children.append(term)
        if len(children) > 0: valid_production = True
    parse_tree = nltk.Tree(str(symbol), children)
    return parse_tree

def fill_in_terminals(parse_tree):
    message = [
        {"role": "system", "content": "Fill in the terminals \"TERM\" to create a valid phrase. Ensure that the sentences make grammatical sense, if needed change the structure of the tree to make the sentence grammatically and semantically coherent, when doing so also change the non-terminals tag to be consistent.\nHere is the list of possible tags (without the variations SUBJ, LOC, ...):\n"+tags},
        {"role": "user", "content": "(S (NP-SBJ-19 (DT TERM) (JJ TERM) (NN TERM)) (VP (VB TERM) (RB TERM)))"},
        {"role": "assistant", "content": "(S (NP-SBJ-19 (DT The) (JJ quick) (NN rabbit)) (VP (VBZ jumps) (RB quickly)))"},
        # {"role": "user", "content": "(S (NP-SBJ (NP (JJ TERM) (NN TERM) (NNS TERM)) (PP (IN TERM) (NP (DT TERM) (NN TERM) (POS 's) (NN TERM))))) (VP (VBZ TERM) (NP (DT TERM) (NN TERM))))"},
        # {"role": "assistant", "content": "(S (NP-SBJ (NP (JJ Beautiful) (NN garden) (NNS flowers)) (PP (IN of) (NP (DT TERM) (NN TERM) (POS 's) (NN TERM))))) (VP (VBZ TERM) (NP (DT TERM) (NN TERM))))(JJ Beautiful) (NN garden) (NNS flowers) (IN of) (DT the) (NN lady) (POS 's) (NN garden) (VBZ blooms) (DT the) (NN house)"},
        # {"role": "user", "content": "(S (NP-SBJ (CD TERM) (NN TERM)) (VP (VB TERM) (PP-PRD (IN TERM) (NP (DT TERM) (NN TERM)))) (. .))"},
        # {"role": "assistant", "content": "(CD Three) (NN cats) (VB chase) (IN after) (DT the) (NN mouse) (. .)"},
        {"role": "user", "content": str(parse_tree)}
    ]

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=message
    )
    return response.choices[0].message.content


def generate_gpt(n=10):
    message = [
        {"role": "system", "content": "Generate a phrase with a wide range of lenghts with its parsing tree structure using the provided tags, write ONLY the parse tree. \nHere is the list of possible tags:\n"+tags+"You also should provide information about what is the subject, complements and their type, etc..."},
        {"role": "user", "content": f"Generate {n} trees in this style: (S (NP-SBJ (DT The) (NN dog)) (VP (VBZ barks)) (. .)) separated by a new line."},
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=message
    )
    print(response.choices[0].message.content)

    return response.choices[0].message.content.split("\n")

import os
# Start generation with the initial non-terminal symbol 'S'
with open("generated_sentences.txt", "a") as f:
    generated = 0
    while generated < 5:
        #empty tree
        nltk_tree = nltk.Tree('S', [])

        # generated_sentence, parse_tree = generate_sentence(grammar, S, nltk_tree, max_depth=3)
        parse_tree = generate_sentence_no_Terminals(grammar, S, max_depth=8)
        print(parse_tree.pretty_print())
        accept_tree = input("Accept Tree? (y/n): ")
        if accept_tree.lower() != 'y': continue
        parse_tree = fill_in_terminals(parse_tree)
        try:
            parse_tree = nltk.Tree.fromstring(parse_tree)
            print(parse_tree.pretty_print())
            generated_sentence = " ".join(parse_tree.leaves())
            print("Generated Sentence:", generated_sentence)

            accept = input("Accept phrase? (y/n): ")
            if accept.lower() != 'y': continue
            f.write("Generated Sentence: "+generated_sentence+"\n")
            f.write("Parse Tree: "+str(parse_tree)+"\n\n")
            generated += 1
        except:
            print("INVALID TREE")

    # parse_trees = generate_gpt(5)
    # for tree in parse_trees:
    #     try:
    #         parse_tree = nltk.Tree.fromstring(tree)
    #         print(parse_tree.pretty_print())
    #         generated_sentence = " ".join(parse_tree.leaves())
    #         print("Generated Sentence:", generated_sentence)

    #         accept = input("Accept phrase? (y/n): ")
    #         if accept.lower() != 'y': continue
    #         f.write("Generated Sentence: "+generated_sentence+"\n")
    #         f.write("Parse Tree: "+str(parse_tree)+"\n\n")
    #     except:
    #         print("INVALID TREE")

        
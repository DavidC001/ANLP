import requests
from openai import OpenAI
from nltk.tokenize import TreebankWordTokenizer, sent_tokenize
import wikipedia

# Define the ConceptNet API endpoint
CONCEPTNET_API = "http://api.conceptnet.io"

# Set up the OpenAI client pointing to the local LM Studio server
local = OpenAI(base_url="http://localhost:5000/v1", api_key="lm-studio")
# client to use the OpenAI API (get key from .env file)
OPENAI_KEY = ""
client = OpenAI(api_key=OPENAI_KEY)


def query_conceptnet(concept):
    url = f"{CONCEPTNET_API}/c/en/{concept}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def generate_and_annotate_sentence(prompt, model="gpt-4o"):
    history = [
        {"role": "system", "content": """Your purpose is to accurately tag the phrases the user sends you. 
This is an example on how to annotate the sentence "You wonder if he was manipulating the market with his bombing targets ." with semantic roles:
    <ARG0> You </ARG0> <rel> wonder </rel> <ARG1> if he was manipulating the market with his bombing targets </ARG1> .
    You wonder if <ARG0> he </ARG0> was <rel> manipulating </rel> <ARG1> the market </ARG1> <ARGM-MNR> with his bombing targets </ARGM-MNR> .
Here is another example:
    
Respond ONLY with the annotation in the EXACT format as shown above, you should repeat the whole phrase with the annotations for one relation at the time as shown in the example, the tags should always be opened and closed. Consider only the propositions as relations.
The available roles are:
    "Agent <ARG0> - The doer of the action",
    "Patient or theme <ARG1> - The entity affected by the action",
    "Instrument <ARG2> - The instrument or means by which the action is performed",
    "Starting point <ARG3> - The starting point or origin of the action",
    "Cause <ARG4> - The cause or reason for the action",
    "Beneficiary <ARG5> - The beneficiary of the action",
    "Extent <ARGM-EXT> - The extent or degree of the action or property",
    "Location <ARGM-LOC> - Where the action occurs",
    "Light verb <ARGM-LVB> - Light verb construction",
    "Goal <ARGM-GOL> - The endpoint or recipient of the action",
    "Recipient <ARGM-REC> - The recipient or beneficiary of the action",
    "Negation <ARGM-NEG> - Negation of the action",
    "Purpose <ARGM-PRP> - The purpose or reason for the action",
    "Direction <ARGM-DIR> - The direction of the action",
    "Temporal <ARGM-TMP> - When the action occurs",
    "Manner <ARGM-MNR> - How the action is performed",
    "Adjectival <ARGM-ADJ> - Adjectival modification",
    "General argument A <ARGA> - An unspecified argument A",
    "Discourse <ARGM-DIS> - Discourse markers",
    "Purpose clause <ARGM-PRR> - Purpose or result clause",
    "Modal <ARGM-MOD> - Modal verbs or constructions",
    "Adverbial <ARGM-ADV> - Adverbial modification",
    "Cause <ARGM-CAU> - The cause of the action",
    "Comitative <ARGM-COM> - Accompaniment or comitative",
    "Predicate <ARGM-PRD> - Secondary predication",
    "Complex connective <ARGM-CXN> - Complex connectives or constructions" """
    "Displaced ARG1 <ARG1-DSP> - Displaced or alternative ARG1",
         },
        {"role": "user", "content": prompt}
    ]
    
    if model.startswith("gpt"):
        completion = client.chat.completions.create(
            model=model,
            messages=history,
            temperature=0.1,
            stream=True,
        )
    else:
        completion = local.chat.completions.create(
            model=model,
            messages=history,
            temperature=0.1,
            stream=True,
        )

    new_message = {"role": "assistant", "content": ""}
    
    for chunk in completion:
        if chunk.choices[0].delta.content:
            new_message["content"] += chunk.choices[0].delta.content

    return new_message["content"]

def fetch_wikipedia_article(title):
    try:
        page = wikipedia.page(title)
        return page.content
    except Exception as e:
        print(f"Error fetching Wikipedia article: {e}")
        return None

def main():
    seed_concept = 'dad'
    concept_data = query_conceptnet(seed_concept)
    
    # for edge in concept_data['edges']:
    #     rel = edge['rel']['label']
    #     related = edge['end']['label']
    #     print (f"{seed_concept} {rel} {related}")
        
    #     # Generate sentence
    #     prompt_generate = f"Generate a sentence involving '{seed_concept}' and '{rel} {related}'. Respond only with the sentence. If you cannot generate a sentence, respond with 'FAIL'."
    #     sentence = generate_and_annotate_sentence(prompt_generate)
    #     if sentence == "FAIL":
    #         continue

    # Read the wikipedia article
    article_title = "Walt Disney"
    article_content = fetch_wikipedia_article(article_title)
    # Split the article into sentences
    sentences = sent_tokenize(article_content)

    tokenizer = TreebankWordTokenizer()
    for sentence in sentences:
        sentence = tokenizer.tokenize(sentence)
        sentence = " ".join(sentence)
        
        # Annotate sentence
        prompt_annotate = f"{sentence}"
        
        annotation = generate_and_annotate_sentence(prompt_annotate, model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF")
        # annotation = generate_and_annotate_sentence(prompt_annotate, model="gpt-4o")
        
        print(sentence)
        print(annotation, end="\n\n")
        accept = input("Accept? (y/n) ")
        if accept == "y":
            with open("annotations.txt", "a") as f:
                f.write(f"{sentence}\t{annotation}\n\n")

if __name__ == "__main__":
    main()

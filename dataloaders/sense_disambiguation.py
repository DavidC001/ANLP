import requests
from bs4 import BeautifulSoup
import json
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
# Point to the local server
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)
# client = OpenAI(base_url="http://localhost:5000/v1", api_key="lm-studio")

def get_subpages(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    subpages = [a['href'] for a in soup.find_all('a', href=True)]
    return subpages

def fetch_page_content(subpage_url):
    response = requests.get(subpage_url)
    return response.text

def process_with_llm(content):
    prompt = f"""
    Extract the mapping between semantic roles and logical roles from the following content. Use the following logical roles for the mapping:

    - Subj: Subject
    - Obj: Object
    - IO: Indirect Object
    - Pred: Predicate
    - Comp: Complement
    - Adv: Adverbial
    - Attr: Attribute
    - Poss: Possessor
    - Top: Topic
    
    {content}
    
    Provide the mapping in JSON format with the proposition sense (the role set id like be.01) as the key. return ONLY the JSON object.
    """
    model = "gpt-4o-mini"
    # model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF"
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return only the mapping between semantic roles and logical roles."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )
    result = completion.choices[0].message.content.strip()

    #take from the first { to the last }
    start = result.find('{')
    end = result.rfind('}') + 1
    return result[start:end]

def main():
    base_url = 'https://verbs.colorado.edu/propbank/framesets-english-aliases/'
    subpages = get_subpages(base_url)
    
    mappings = {}    
    for subpage in subpages[1:]:
        subpage_url = base_url + subpage
        # breakpoint()
        content = fetch_page_content(subpage_url)
        mapping = process_with_llm(content)
        print(f"Processed {subpage}:\n\t{mapping.replace("\n"," ")}")
        mappings[subpage] = json.loads(mapping)
    
        with open('semantic_to_logical_roles.json', 'w') as outfile:
            json.dump(mappings, outfile, indent=4)

if __name__ == '__main__':
    main()

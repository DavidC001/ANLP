import requests

def get_conceptnet_relationships(concept):
    url = f"http://api.conceptnet.io/c/en/{concept}"
    response = requests.get(url).json()
    edges = response['edges']
    relationships = []
    for edge in edges:
        start = edge['start']['label']
        end = edge['end']['label']
        rel = edge['rel']['label']
        relationships.append((start, rel, end))
    return relationships

def generate_srl_phrases(concept):
    relationships = get_conceptnet_relationships(concept)
    print(f"Relationships for {concept}: {relationships}")
    phrases = []
    for rel in relationships:
        if rel[1] == "involves":
            phrase = f"{rel[0]} involves {rel[2]}."
            roles = {"Predicate": rel[0], "Theme": rel[2]}
        elif rel[1] == "requires":
            phrase = f"{rel[0]} requires {rel[2]}."
            roles = {"Predicate": rel[0], "Theme": rel[2]}
        elif rel[1] == "IsA":
            phrase = f"{rel[0]} is a type of {rel[2]}."
            roles = {"Theme": rel[0], "Predicate": "is", "Attribute": f"a type of {rel[2]}"}
        elif rel[1] == "hasPrerequisite":
            phrase = f"{rel[0]} has a prerequisite of {rel[2]}."
            roles = {"Predicate": rel[0], "Theme": "has", "Attribute": f"a prerequisite of {rel[2]}"}
        elif rel[1] == "UsedFor":
            phrase = f"{rel[0]} is used for {rel[2]}."
            roles = {"Predicate": rel[0], "Theme": "is used for", "Attribute": rel[2]}
        else:
            continue
        phrases.append((phrase, roles))
    return phrases

concept = "clover"
srl_phrases = generate_srl_phrases(concept)
print(f"Concept: {concept}\n SRL Phrases: {srl_phrases}\n")
for phrase, roles in srl_phrases:
    print(f"Phrase: {phrase}")
    print(f"SRL Roles: {roles}\n")

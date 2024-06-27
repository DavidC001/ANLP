import torch
from nltk.tokenize import TreebankWordTokenizer, sent_tokenize
import json
import wikipedia
import re
import matplotlib.pyplot as plt
import networkx as nx
from langchain_community.graphs import Neo4jGraph
from tqdm import tqdm
import sys
sys.path.append('.')

from dataloaders.UP_dataloader import roles 
from model import SRL_MODEL, print_results

def escape_text(text):
    return re.sub(r"([\'\"\\])", r"\\\1", text)

def get_spans(text, word_ids, role_logits, threshold=0.75):
    spans = []
    current_span = []
    current_role = None

    for idx, logits in enumerate(role_logits):
        for role_idx, logit in enumerate(logits):
            if torch.sigmoid(logit) > threshold:
                role = roles[role_idx + 2]
                if role != current_role:
                    if current_span:
                        spans.append((current_span, current_role))
                    current_span = [text[idx]]
                    current_role = role
                else:
                    current_span.append(text[idx])

    if current_span:
        spans.append((current_span, current_role))

    return spans

def populate_knowledge_graph(relational_logits, results, text, word_ids, graph, sent_id=0):
    tokenizer = TreebankWordTokenizer()
    text_tokens = tokenizer.tokenize(text)

    relation_positions = [idx for idx, logit in enumerate(relational_logits) if torch.sigmoid(logit) > 0.75]

    for relation_position, result in zip(relation_positions, results):
        # Get the spans
        spans = get_spans(text_tokens, word_ids, result)

        relation_node = {
            "id": f"relation_{relation_position}_{sent_id}",
            "text": escape_text(text_tokens[relation_position]),
            "role": "relation"
        }
        # Create relation node in Neo4j
        graph.query(f"CREATE (n:Relation {{id: '{relation_node['id']}', text: '{relation_node['text']}', role: '{relation_node['role']}'}})")

        for i, (span, role) in enumerate(spans):
            span_text = " ".join(span)
            span_node = {
                "id": f"span_{relation_position}_{i}_{sent_id}",
                "text": escape_text(span_text),
                "role": role
            }
            # Create span node in Neo4j
            graph.query(f"CREATE (n:Span {{id: '{span_node['id']}', text: '{span_node['text']}', role: '{span_node['role']}'}})")

            # Connect span node to the relation node with the role as the type of connection
            role = role.replace("-", "_")  # Replace dashes with underscores for Neo4j compatibility
            graph.query(f"""
            MATCH (a:Relation {{id: '{relation_node['id']}'}}), (b:Span {{id: '{span_node['id']}'}})
            CREATE (a)-[:{role}]->(b)
            """)

def show_KG(graph):
    # Fetch nodes and relationships from Neo4j
    query_nodes = "MATCH (n) RETURN n"
    query_rels = "MATCH (n)-[r]->(m) RETURN n, type(r) as rel_type, m"

    # Use the query method provided by Neo4jGraph
    nodes = graph.query(query_nodes)
    relationships = graph.query(query_rels)

    # Create a NetworkX graph
    G = nx.DiGraph()

    # Add nodes to the NetworkX graph
    node_colors = []
    for node_record in nodes:
        node = node_record['n']
        node_id = node['text']
        node_label = node.get('text', 'Unknown')
        if not G.has_node(node_id):
            if node.get('role') == 'relation':
                node_colors.append('red')
            else:
                node_colors.append('skyblue')
            G.add_node(node_id, label=node_label)

    # Debugging: Print the number of nodes and colors
    print(f"Number of nodes: {len(G.nodes())}")
    print(f"Number of node colors: {len(node_colors)}")

    # Add edges to the NetworkX graph
    for rel_record in relationships:
        start_node = rel_record['n']
        end_node = rel_record['m']
        relationship_type = rel_record['rel_type']  # Relationship type
        start_id = start_node['text']  # Start node ID
        end_id = end_node['text']  # End node ID
        G.add_edge(start_id, end_id, label=relationship_type)

    # Adjust the layout for better spacing
    pos = nx.kamada_kawai_layout(G)
    plt.figure(figsize=(10, 7))

    # Draw the nodes and edges
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color=node_colors, font_size=10, font_color="black", font_weight="bold", edge_color="gray")
    
    # Draw edge labels
    labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.title("Neo4j Knowledge Graph Visualization")
    plt.show()

def clean_database(graph):
    graph.query("MATCH (n) DETACH DELETE n")

def fetch_wikipedia_article(title):
    try:
        page = wikipedia.page(title)
        return page.content
    except Exception as e:
        print(f"Error fetching Wikipedia article: {e}")
        return None

if __name__ == "__main__":
    # Initialize the Neo4j connection
    url = "neo4j+s://70b9b6b6.databases.neo4j.io"
    username = "neo4j"
    password = "4euztjZ-dqQH2HwstTtkxsmznyjfvoHugK2puR3he78"
    graph = Neo4jGraph(
        url=url,
        username=username,
        password=password
    )

    # Clean the database before populating
    clean_database(graph)

    # Fetch a Wikipedia article
    # article_title = "Natural language processing"
    # article_content = fetch_wikipedia_article(article_title)
    # if not article_content:
    #     sys.exit("Failed to fetch Wikipedia article.")
    
    # Read text from a file text.txt
    with open("text.txt", "r") as f:
        article_content = f.read()

    # Initialize the model
    model_name = "SRL_BERT_gated_red100_100_norm_L2"
    with open(f"models/{model_name}.json", "r") as f:
        config = json.load(f)

    model = SRL_MODEL(**config)
    model.load_state_dict(torch.load(f"models/{model_name}.pt"))

    # Split the article into sentences
    sentences = sent_tokenize(article_content)

    # Process each sentence
    i=0
    for sentence in tqdm(sentences):
        print(f"\nProcessing sentence: {sentence}")
        relational_logits, senses_logits, results = model.inference(sentence)
        print_results(relational_logits, senses_logits, results, sentence)
        word_ids = list(range(len(sentence.split())))  # Adjust this to match your word_ids processing
        populate_knowledge_graph(relational_logits, results[0], sentence, word_ids, graph, i)
        i+=1

    # Show the knowledge graph
    show_KG(graph)

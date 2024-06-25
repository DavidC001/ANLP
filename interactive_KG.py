import torch
from transformers import AutoTokenizer
from nltk.tokenize import TreebankWordTokenizer, sent_tokenize
import json
import sys
import wikipedia
import re
from dash import Dash, dcc, html, Input, Output
import dash_cytoscape as cyto
import networkx as nx
from langchain_community.graphs import Neo4jGraph
from tqdm import tqdm

sys.path.append('.')
from dataloaders.UP_dataloader import roles 
from model import SRL_BERT, print_results

def escape_text(text):
    return re.sub(r"([\'\"\\])", r"\\\1", text)

def get_spans(text, role_logits, threshold=0.75):
    role_spans = {}

    for idx, logits in enumerate(role_logits):
        for role_idx, logit in enumerate(logits):
            if torch.sigmoid(logit) > threshold:
                if role_idx not in role_spans:
                    role_spans[role_idx] = ([text[idx]], [idx])
                else:
                    role_spans[role_idx][0].append(text[idx])
                    role_spans[role_idx][1].append(idx)

    spans = []
    for role_idx, (span, pos) in role_spans.items():
        spans.append((span, pos, roles[role_idx + 2]))

    return spans

def populate_knowledge_graph(relational_logits, results, text, graph, sent_id=0):
    tokenizer = TreebankWordTokenizer()
    text_tokens = tokenizer.tokenize(text)

    relation_positions = [idx for idx, logit in enumerate(relational_logits) if torch.sigmoid(logit) > 0.75]

    for relation_position, result in zip(relation_positions, results):
        # Get the spans
        spans = get_spans(text_tokens, result)

        relation_node = {
            "id": f"relation_{relation_position}_{sent_id}",
            "text": escape_text(text_tokens[relation_position]),
            "role": "relation"
        }
        # Create relation node in Neo4j
        graph.query(f"CREATE (n:Relation {{id: '{relation_node['id']}', text: '{relation_node['text']}', role: '{relation_node['role']}'}})")

        for span, pos, role in spans:
            span_text = " ".join(span)
            position = ".".join([str(p) for p in pos])
            span_node = {
                "id": f"span_{position}_{sent_id}",
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

def fetch_graph_data(graph):
    # Fetch nodes and relationships from Neo4j
    query_nodes = "MATCH (n) RETURN n"
    query_rels = "MATCH (n)-[r]->(m) RETURN n, type(r) as rel_type, m"

    # Use the query method provided by Neo4jGraph
    nodes = graph.query(query_nodes)
    relationships = graph.query(query_rels)

    # Create a NetworkX graph
    G = nx.DiGraph()

    # Add nodes to the NetworkX graph
    for node_record in nodes:
        node = node_record['n']
        node_id = node['id']
        node_label = node.get('text', 'Unknown')
        G.add_node(node_id, label=node_label, role=node.get('role'))

    # Add edges to the NetworkX graph
    for rel_record in relationships:
        start_node = rel_record['n']
        end_node = rel_record['m']
        relationship_type = rel_record['rel_type']  # Relationship type
        start_id = start_node['id']  # Start node ID
        end_id = end_node['id']  # End node ID
        G.add_edge(start_id, end_id, label=relationship_type)

    return G

def create_cytoscape_elements(G):
    elements = []

    for node in G.nodes(data=True):
        elements.append({
            'data': {'id': node[0], 'label': node[1]['label'], 'role': node[1]['role']}
        })

    for edge in G.edges(data=True):
        elements.append({
            'data': {'source': edge[0], 'target': edge[1], 'label': edge[2]['label']}
        })

    return elements

def clean_database(graph):
    graph.query("MATCH (n) DETACH DELETE n")

def fetch_wikipedia_article(title):
    try:
        page = wikipedia.page(title)
        return page.content
    except Exception as e:
        print(f"Error fetching Wikipedia article: {e}")
        return None

def compute_graph(graph, mode):
    # Clean the database before populating
    clean_database(graph)

    if mode == "w":
        # Fetch a Wikipedia article
        article_title = "Kratos"
        article_content = fetch_wikipedia_article(article_title)
        if not article_content:
            sys.exit("Failed to fetch Wikipedia article.")
    else:
        # Read text from a file text.txt
        with open("text.txt", "r") as f:
            article_content = f.read()

    # Initialize the model
    model_name = "SRL_BERT_gated_transform_red100_100_norm_L2"
    with open(f"models/{model_name}.json", "r") as f:
        config = json.load(f)

    model = SRL_BERT(**config)
    model.load_state_dict(torch.load(f"models/{model_name}.pt"))

    # Split the article into sentences
    sentences = sent_tokenize(article_content)

    # Process each sentence
    i = 0
    for sentence in tqdm(sentences):
        #remove \n
        sentence = sentence.replace('\n', ' ')
        print(f"\nProcessing sentence: {sentence}")

        relational_logits, senses_logits, results = model.inference(sentence)
        # print_results(relational_logits, senses_logits, results, sentence)

        if (len(results) > 0):
            populate_knowledge_graph(relational_logits, results[0], sentence, graph, i)
        i += 1


def serve_KG(graph):
    G = fetch_graph_data(graph)
    elements = create_cytoscape_elements(G)

    app = Dash(__name__)

    app.layout = html.Div([
        dcc.Dropdown(
            id='dropdown',
            options=[{'label': node[1]['label'], 'value': node[0]} for node in G.nodes(data=True) if node[1]['role'] == 'relation'],
            placeholder="Select a relation"
        ),
        cyto.Cytoscape(
            id='cytoscape',
            layout={'name': 'cose'},  # Using 'cose' layout for better node spreading
            style={'width': '100%', 'height': '600px'},
            elements=elements,
            stylesheet=[
                {'selector': 'node', 'style': {'label': 'data(label)', 'background-color': 'skyblue'}},
                {'selector': '[role = "relation"]', 'style': {'background-color': 'red'}},
                {'selector': 'edge', 'style': {'label': 'data(label)', 'text-rotation': 'autorotate', 'text-margin-y': '-10px'}}
            ]
        )
    ])

    @app.callback(
        Output('cytoscape', 'elements'),
        Input('dropdown', 'value')
    )
    def update_graph(selected_relation):
        if not selected_relation:
            return create_cytoscape_elements(G)

        subgraph = nx.ego_graph(G, selected_relation)
        return create_cytoscape_elements(subgraph)

    app.run_server(debug=True)


if __name__ == '__main__':
    # Initialize the Neo4j connection
    url = "neo4j+s://70b9b6b6.databases.neo4j.io"
    username = "neo4j"
    password = "4euztjZ-dqQH2HwstTtkxsmznyjfvoHugK2puR3he78"
    graph = Neo4jGraph(
        url=url,
        username=username,
        password=password
    )

    mode = input("Do you want to compute the graph from a Wikipedia article or from a text file? (w/f) or only show the graph? (s): ")

    if mode == "w" or mode == "f":
        compute_graph(graph, mode)

    serve_KG(graph)
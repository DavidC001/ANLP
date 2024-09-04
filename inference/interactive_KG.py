import torch
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import TreebankWordTokenizer, sent_tokenize
import json
import sys
import wikipedia
import re
from dash import Dash, dcc, html, Input, Output, State
import dash_cytoscape as cyto
import networkx as nx
from langchain_community.graphs import Neo4jGraph
from tqdm import tqdm
import time
from dotenv import load_dotenv
import os

sys.path.append('.')
from dataloaders.UP_dataloader import roles as UP_roles
from dataloaders.NomBank_dataloader import roles as NOM_roles
from model import SRL_MODEL

roles = []

def escape_text(text):
    return re.sub(r"([\'\"\\])", r"\\\1", text)


# Initialize the model
model_name = input("Enter the name of the model to use: ")
with open(f"models/{model_name}.json", "r") as f:
    config = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
config["device"] = device
model = SRL_MODEL(**config)
model.load_state_dict(torch.load(f"models/{model_name}.pt"))

# function to check if a span corresponds to an already existing node in the graph, for ARGM use exact match, for the others use a subset match
entity_nodes = [] # saves the span id used for the entity nodes
bert_representation = [] # saves the representation of the span
argm_nodes = [] # saves the span id used for the argm nodes
argm_texts = [] # saves the text of the argm nodes
entity_tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
entity_model = AutoModel.from_pretrained("bert-large-cased")
def check_existing_node(text, span_text, span_idx, role, id):
    if role.startswith("ARGM"):
        if span_text in argm_texts:
            return argm_nodes[argm_texts.index(span_text)]
        else:
            argm_nodes.append(id)
            argm_texts.append(span_text)
    else:
        # load a BERT model and use it to score similarity between the spans
        with torch.no_grad():
            tokenized_text = entity_tokenizer(text, return_tensors="pt")
            input_ids = tokenized_text['input_ids']
            attention_mask = tokenized_text['attention_mask']
            word_ids = tokenized_text.word_ids()
            word_ids = [(word_id if word_id is not None else -1) for word_id in word_ids]
            # brutta soluzione a J.
            delta = 0
            seen_ids = set()
            for i, word_id in enumerate(word_ids):
                if word_id <= 0: continue
                start, end = tokenized_text.word_to_chars(word_id)
                if text[start-1] != ' ' and word_id not in seen_ids:
                    delta += 1
                    seen_ids.add(word_id)
                word_ids[i] = word_id - delta
            
            outputs = entity_model(input_ids, attention_mask=attention_mask)

            # get the mean of the last hidden states for the span
            last_hidden_states = outputs.last_hidden_state
            span_tokens = [i for i, word_id in enumerate(word_ids) if word_id in span_idx]
            mean_representation = last_hidden_states[0,span_tokens].mean(dim=0)

            best = 0
            indx = None
            for i,rep in enumerate(bert_representation):
                # breakpoint()
                sim = torch.cosine_similarity(rep,mean_representation,dim=0)
                if (sim>0.9 and sim>best):
                    best = sim
                    indx = i

            if indx:
                return entity_nodes[indx]
            else:
                entity_nodes.append(id)
                bert_representation.append(mean_representation)
    
    return id

# function to get the spans of the roles from the logits
def get_spans(text, role_logits, threshold=0.5, mode="t"):
    """
        Get the spans of the roles from the logits

        Parameters:
            text: The text of the sentence
            role_logits: The logits of the roles
            threshold: The threshold for the probability
            mode: The mode to use for the spans (t for taking the most probable, a for aggregating all)

        Returns:
            The spans of the roles
    """
    role_spans = {}
    # breakpoint()
    for idx, logits in enumerate(role_logits):
        for role_idx, logit in enumerate(logits):
            prob = torch.sigmoid(logit)
            if prob > threshold:
                if role_idx not in role_spans:
                    role_spans[role_idx] = [([text[idx]], [idx], [prob])]
                else:
                    if (role_spans[role_idx][-1][1][-1] + 1) == idx: # Check if the current token is contiguous with the previous one
                        role_spans[role_idx][-1][0].append(text[idx])
                        role_spans[role_idx][-1][1].append(idx)
                        role_spans[role_idx][-1][2].append(prob)
                    else: # If not, start a new span
                        role_spans[role_idx].append(([text[idx]], [idx], [prob]))

    final_spans = []
    for role_idx, spans in role_spans.items():
        if mode == "t":
            # take the most probable (use mean)
            max_prob = 0
            max_span = None
            for span in spans:
                prob = torch.mean(torch.tensor(span[2]))
                if prob > max_prob:
                    max_prob = prob
                    max_span = span
            final_spans.append((max_span[0], max_span[1], roles[role_idx+2]))

        elif mode == "a":
            # concatenate all spans
            span_text = []
            span_pos = []
            for span in spans:
                span_text.extend(span[0])
                span_pos.extend(span[1])
            final_spans.append((span_text, span_pos, roles[role_idx+2]))


    return final_spans

# function to query the Neo4j database
def query_neo4j(graph, query):
    """
        Query the Neo4j database

        Parameters:
            graph: The Neo4jGraph object
            query: The query to execute

        Returns:
            The results of the query
    """
    succesful = False
    while not succesful:
        try:
            results = graph.query(query)
            succesful = True
        except Exception as e:
            print(f"Error executing query: {e}")
            succesful = False
            #wait for 5 seconds
            time.sleep(5)
    return results

# function to populate the knowledge graph with the results of the inference
def populate_knowledge_graph(relational_logits, results, text, graph, sent_id=0, aggregate_spans="t"):
    """
        Populate the knowledge graph with the results of the inference

        Parameters:
            relational_logits: The logits of the relations
            results: The logits of the roles
            text: The text of the sentence
            graph: The Neo4jGraph object
            sent_id: The ID of the sentence
            aggregate_spans: The mode to use for the spans (t for taking the most probable, a for aggregating all)
    """
    tokenizer = TreebankWordTokenizer()
    text_tokens = tokenizer.tokenize(text)

    relation_positions = [idx for idx, logit in enumerate(relational_logits) if torch.sigmoid(logit) > 0.75]

    for relation_position, result in zip(relation_positions, results):
        # Get the spans
        spans = get_spans(text_tokens, result, mode=aggregate_spans)

        relation_node = {
            "id": f"relation_{relation_position}_{sent_id}",
            "text": escape_text(text_tokens[relation_position]),
            "role": "relation"
        }

        # Create relation node in Neo4j
        query_neo4j(graph, f"CREATE (n:Relation {{id: '{relation_node['id']}', text: '{relation_node['text']}', role: '{relation_node['role']}'}})")

        for span, pos, role in spans:
            span_text = " ".join(span)
            position = ".".join([str(p) for p in pos])
            span_node = {
                "id": f"span_{position}_{sent_id}",
                "text": escape_text(span_text),
                "role": role
            }

            # Check if the span corresponds to an already existing node in the graph
            span_node["id"] = check_existing_node(text, span_text, pos, role, span_node["id"])

            # Create span node in Neo4j
            query_neo4j(graph, f"CREATE (n:Span {{id: '{span_node['id']}', text: '{span_node['text']}', role: '{span_node['role']}'}})")

            # Connect span node to the relation node with the role as the type of connection
            role = role.replace("-", "_")  # Replace dashes with underscores for Neo4j compatibility
            query_neo4j(graph, f"""
                MATCH (a:Relation {{id: '{relation_node['id']}'}}), (b:Span {{id: '{span_node['id']}'}})
                CREATE (a)-[:{role}]->(b)
            """)

def fetch_graph_data(graph):
    """
        Fetch the graph data from Neo4j

        Parameters:
            graph: The Neo4jGraph object

        Returns:
            The NetworkX graph
    """

    # Fetch nodes and relationships from Neo4j
    query_nodes = "MATCH (n) RETURN n"
    query_rels = "MATCH (n)-[r]->(m) RETURN n, type(r) as rel_type, m"

    # Use the query method provided by Neo4jGraph
    nodes = query_neo4j(graph, query_nodes)
    relationships = query_neo4j(graph, query_rels)

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
    """
        Create the elements for the Cytoscape component

        Parameters:
            G: The NetworkX graph

        Returns:
            The elements for the Cytoscape component
    """
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
    query_neo4j(graph, "MATCH (n) DETACH DELETE n")

def fetch_wikipedia_article(title):
    """
        Fetch a Wikipedia article

        Parameters:
            title: The title of the article
    """
    try:
        page = wikipedia.page(title)
        return page.content
    except Exception as e:
        print(f"Error fetching Wikipedia article: {e}")
        return None

def compute_graph(graph, mode):
    """
        Compute the knowledge graph from the selected input mode

        Parameters:
            graph: The Neo4jGraph object
            mode: The mode to use for the input (w for Wikipedia, f for file)
    """

    # Clean the database before populating
    reset_database = input("Do you want to reset the database? (y/n): ")
    if reset_database == "y":
        clean_database(graph)

    if mode == "w":
        # Fetch a Wikipedia article
        article_content = None
        while not article_content:
            article_title = input("Enter the title of the Wikipedia article: ")
            article_content = fetch_wikipedia_article(article_title)
    else:
        # Read text from a file text.txt
        with open("text.txt", "r", encoding="utf-8") as f:
            article_content = f.read()

    separate_sentences = input("Do you want to process each sentence separately? (y/n): ")
    # Split the article into sentences
    if separate_sentences == "y":
        sentences = sent_tokenize(article_content)
    else:
        sentences = [article_content]

    aggregate_spans = input("Do you want to aggregate all spans for each relation or only the most probable one? (a/t): ")

    # Process each sentence
    i = 0
    for sentence in tqdm(sentences):
        #remove \n
        sentence = sentence.replace('\n', ' ')
        print(f"\nProcessing sentence: {sentence}")

        relational_logits, senses_logits, results = model.inference(sentence)

        if (results and len(results) > 0):
            populate_knowledge_graph(relational_logits, results[0], sentence, graph, i, aggregate_spans)
        i += 1


def parse_natural_language_query(query, model):
    """
    Parse a natural language query using the SRL model.

    Parameters:
        query: The natural language query string
        model: The SRL model instance

    Returns:
        A structured format of the query (e.g., list of entities and relationships)
    """
    tokenizer = TreebankWordTokenizer()
    query_tokens = tokenizer.tokenize(query)

    # Perform SRL inference
    relational_logits, senses_logits, results = model.inference(query)
    
    if results and len(results) > 0:
        spans = []
        for result in results[0]:
            spans.extend(get_spans(query_tokens, result, threshold=0.5, mode="a"))
        return spans
    else:
        return None

def map_to_cypher_query(parsed_query):
    """
    Convert the parsed query from SRL model output to a Cypher query.

    Parameters:
        parsed_query: The parsed query output from the SRL model

    Returns:
        A Cypher query string
    """
    if not parsed_query:
        return None

    # Initialize lists to hold MATCH and WHERE clauses
    match_clauses = []
    where_clauses = []

    # Construct MATCH and WHERE clauses based on the parsed query
    for span_text, span_pos, role in parsed_query:
        # Convert the span text to a plain string
        span_text_str = " ".join(span_text)

        where_clauses.append(f"n.text CONTAINS '{span_text_str}' OR m.text CONTAINS '{span_text_str}'")

    # Construct the MATCH clause
    cypher_query = "MATCH (n)<-[r]-(P)-[r2]->(m)"

    # Construct the WHERE clause if conditions are available
    if where_clauses:
        cypher_query += " WHERE " + " OR ".join(where_clauses)

    # Add the RETURN statement to complete the Cypher query
    cypher_query += " RETURN n, type(r) as r, P, type(r2) as r2, m"
    return cypher_query


def query_neo4j_with_nl(graph, query, model):
    """
    Query the Neo4j database using a natural language query.

    Parameters:
        graph: The Neo4jGraph object
        query: The natural language query string
        model: The SRL model instance

    Returns:
        The results of the query
    """
    # Parse the natural language query
    parsed_query = parse_natural_language_query(query, model)
    
    if not parsed_query:
        print("Could not parse the query.")
        return None

    # Convert to Cypher query
    cypher_query = map_to_cypher_query(parsed_query)

    if not cypher_query:
        print("Could not map the query to Cypher.")
        return None

    # Execute the Cypher query
    results = query_neo4j(graph, cypher_query)

    # breakpoint()
    return results



def serve_KG(graph):
    """
        Serve the knowledge graph visualization using Dash

        Parameters:
            graph: The Neo4jGraph object
    """

    G = fetch_graph_data(graph)
    elements = create_cytoscape_elements(G)

    app = Dash(__name__)

    app.layout = html.Div([
        dcc.Input(id='nl-query', type='text', placeholder='Enter a natural language query'),
        html.Button('Query', id='query-button', n_clicks=0),
        dcc.Dropdown(
            id='dropdown',
            options=[{'label': node[1]['label'], 'value': node[0]} for node in G.nodes(data=True) if node[1]['role'] == 'relation'],
            placeholder="Select a relation"
        ),
        cyto.Cytoscape(
            id='cytoscape',
            layout={'name': 'cose', 'randomize': True, 'nodeRepulsion': 1000000, 'animate': False},
            style={'width': '100%', 'height': '800px'},
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
        [Input('query-button', 'n_clicks'),
        Input('dropdown', 'value')],
        State('nl-query', 'value')
    )
    def update_graph(n_clicks, selected_relation, query):
        # Use nonlocal to modify the global graph G
        nonlocal G

        if n_clicks > 0 and query:
            print(f"Natural language query received: {query}")
            # Query the Neo4j database using the natural language query
            results = query_neo4j_with_nl(graph, query, model)
            if results:
                # Create a new NetworkX graph with the results
                G = nx.DiGraph()
                for record in results:
                    node1 = record['n']
                    role1 = record['r']
                    proposition = record['P']
                    node2 = record['m']
                    role2 = record['r2']

                    G.add_node(node1['id'], label=node1['text'], role=node1['role'])
                    G.add_node(node2['id'], label=node2['text'], role=node2['role'])
                    G.add_node(proposition['id'], label=proposition['text'], role=proposition['role'])
                    G.add_edge(node1['id'], proposition['id'], label=role1)
                    G.add_edge(proposition['id'], node2['id'], label=role2)

                return create_cytoscape_elements(G)

        if selected_relation:
            # If a relation is selected in the dropdown, filter the graph accordingly
            subgraph = nx.ego_graph(G, selected_relation)
            return create_cytoscape_elements(subgraph)

        # Default case, return the entire graph
        return create_cytoscape_elements(G)


    app.run() 

if __name__ == '__main__':
    load_dotenv()
    # Initialize the Neo4j connection
    url = os.getenv("NEO4JS_URL")
    username = os.getenv("NEO4JS_USER")
    password = os.getenv("NEO4JS_PASSWORD")
    graph = Neo4jGraph(
        url=url,
        username=username,
        password=password
    )

    dataset = input("Enter the role format (UP/NOM): ")
    roles = UP_roles if dataset == "UP" else NOM_roles

    mode = input("Do you want to compute the graph from a Wikipedia article or from a text file? (w/f) or only show the graph? (s): ")


    if mode == "w" or mode == "f":
        compute_graph(graph, mode)

    serve_KG(graph)
import requests
import graphviz

# Function to get relationships from ConceptNet
def get_conceptnet_relationships(concept):
    url = f"http://api.conceptnet.io/c/en/{concept}"
    response = requests.get(url).json()
    edges = response['edges']
    relationships = []
    for edge in edges:
        start = edge['start']['label']
        end = edge['end']['label']
        rel = edge['rel']['label']
        #filter out non-English words & relation Synonym
        if start.isascii() and end.isascii():
            relationships.append((start, rel, end))
    return relationships

# Function to generate and visualize the graph
def generate_conceptnet_graph(concept):
    relationships = get_conceptnet_relationships(concept)
    
    dot = graphviz.Digraph(comment='ConceptNet')
    
    nodes = set()
    for start, rel, end in relationships:
        nodes.add(start)
        nodes.add(end)
        dot.edge(start, end, label=rel)
    
    # Add nodes with unique colors
    for node in nodes:
        if node == concept:
            dot.node(node, shape='box', style='filled', fillcolor='#a0d080')
        else:
            dot.node(node, shape='box', style='filled', fillcolor='#e0e0e0')
    
    # Render and display the graph
    dot.render(f'{concept}_conceptnet_graph', format='png', cleanup=True)
    dot.view()

# Example usage
concept = "lake"
generate_conceptnet_graph(concept)

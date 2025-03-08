"""
All these functions are redefined in the notebook but using a single star import at the very top of the notebook allows us to run cells out of order.
"""

import os
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from IPython.display import Markdown
from openai import NOT_GIVEN, OpenAI
import openparse


client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))
batch_size = 250
embedding_model = "text-embedding-3-small"

prompt_template = """
Using the document provided, answer the following question:

question: {question}

context: {context}
"""


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# wrapper function around openai to directly return embedding of text
def get_embedding(text: str | list[str], dimensions: int = NOT_GIVEN) -> list[float]:
    """Get the embedding of the input text."""
    if dimensions:
        assert dimensions <= 256, "The maximum number of dimensions is 256."

    response = client.embeddings.create(
        input=text, model=embedding_model, dimensions=dimensions
    )
    return response.data[0].embedding


def get_many_embeddings(texts: list[str]) -> list[list[float]]:
    """Get the embeddings of multiple texts."""
    batch_size = 250
    res = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        api_resp = client.embeddings.create(input=batch_texts, model=embedding_model)
        batch_res = [val.embedding for val in api_resp.data]
        res.extend(batch_res)

    return res


# simple utility function to add a vector to a 3D plot
def add_vector_to_graph(
    fig: go.Figure, vector: list[float], color: str = "red", name: Optional[str] = None
) -> go.Figure:
    # Ensure vector has exactly three components
    assert len(vector) == 3, "Vector must have exactly 3 components."

    # Origin point
    origin = [0, 0, 0]

    # Components of the vector
    x_component, y_component, z_component = vector

    # Adding the line part of the vector
    fig.add_trace(
        go.Scatter3d(
            x=[origin[0], x_component],
            y=[origin[1], y_component],
            z=[origin[2], z_component],
            mode="lines",
            line=dict(color=color, width=5),
            name=name,
        )
    )

    # Adding the cone at the tip of the vector
    fig.add_trace(
        go.Cone(
            x=[x_component],
            y=[y_component],
            z=[z_component],
            u=[x_component],
            v=[y_component],
            w=[z_component],
            sizemode="scaled",
            sizeref=0.1,
            showscale=False,
            colorscale=[[0, color], [1, color]],
            hoverinfo="none",
        )
    )
    return fig


def create_new_graph() -> go.Figure:
    """Create a 3D plotly figure with a simple layout."""
    fig = go.Figure()

    # make sure the plot isn't rotated
    fig.update_layout(
        scene=dict(
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.5),  # Adjust the camera position
                up=dict(x=0, y=0, z=1),  # Sets the z-axis as "up"
                center=dict(x=0, y=0, z=0),  # Focuses the camera on the origin
            ),
            aspectmode="cube",
        )
    )

    # Add a dot at the origin
    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode="markers",
            marker=dict(size=6, color="black", symbol="circle"),
            name="Origin",
        )
    )

    return fig


class VectorDatabase:
    """
    A simple in-memory database to store nodes along with their vectors and perform similarity search.
    """

    def __init__(self):
        self.nodes = []

    def add_node(self, node: openparse.Node) -> None:
        """Add a node along with its vector to the database."""
        assert node.embedding is not None, "Node must have an embedding."

        for existing_node in self.nodes:
            if existing_node.text == node.text:
                print(f"Node with id {node.node_id} already exists. Skipping")
                return

        self.nodes.append(node)

    def find_node(self, node_id: str):
        """Retrieve a node by its ID."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def remove_node(self, node_id: int) -> None:
        """Remove a node from the database."""
        for node in self.nodes:
            if node.node_id == node_id:
                self.nodes.remove(node)
                return
        raise ValueError(f"Node with id {node_id} not found.")

    def find_similar_node(
        self, input_vector: list[float], top_k: int = 3
    ) -> list[openparse.Node]:
        """Find the top_k nodes with the highest cosine similarity to the input_vector."""
        assert self.nodes, "Database is empty. Please add nodes first."
        assert top_k <= len(
            self.nodes
        ), "top_k should be less than or equal to the number of nodes."

        similarities = []
        for node in self.nodes:
            similarity = cosine_similarity(input_vector, node.embedding)
            similarities.append((node, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)

        return [node for node, _ in similarities[:top_k]]

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the database."""
        return len(self.nodes)

    def delete_all_nodes(self) -> None:
        """Delete all nodes from the database."""
        self.nodes = []


def get_completion(prompt: str) -> Markdown:
    """
    OpenAI returns a complex object, this is a simple wrapper function to directly return the completion text.
    """

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    cost_per_million_tokens = 4.00
    cost_dollars = completion.usage.total_tokens / 1_000_000 * cost_per_million_tokens

    print(
        f"Completion used {completion.usage.total_tokens} tokens costing ${cost_dollars:.2f}"
    )

    return Markdown(completion.choices[0].message.content)


db = VectorDatabase()

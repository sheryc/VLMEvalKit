from typing import Optional

import htmlmin
import torch
import zss
from bs4 import BeautifulSoup, Comment
from torchmetrics import Metric

from vlmeval.dataset.utils.bigdocs.metrics_utils import bootstrap_std


class HTMLSimilarityMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("dom_tree_edit_distance", default=list(), dist_reduce_fx="cat")
        self.add_state("screenshot_ssim", default=list(), dist_reduce_fx="cat")

    def update(self, reference: Optional[str], prediction: Optional[str]):
        if reference is not None and prediction is not None and len(reference) > 0 and len(prediction) > 0:
            similarity_dict = html_similarity(reference=reference, prediction=prediction)
            self.dom_tree_edit_distance.append(similarity_dict['dom_tree_edit_distance'])
        else:
            self.dom_tree_edit_distance.append(0.)
            self.screenshot_ssim.append(0.)

    def compute(self):
        if len(self.dom_tree_edit_distance) == 0:
            return {"dom_ted_mean": torch.tensor(0.0), "dom_ted_std": torch.tensor(0.0),
                    "screenshot_ssim_mean": torch.tensor(0.0), "screenshot_ssim_std": torch.tensor(0.0)}
        dom_ted = torch.tensor(self.dom_tree_edit_distance)
        dom_ted_std, dom_ted_lb, dom_ted_ub = bootstrap_std(dom_ted, n_bootstrap=1000, ci=0.95)
        return {"dom_ted_mean": dom_ted.mean(), "dom_ted_std": dom_ted_std}


# Helper class to represent HTML nodes for the ZSS algorithm
class Node:
    def __init__(self, tag_name, attributes):
        self.tag_name = tag_name
        self.attributes = attributes
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def get_children(self):
        return self.children

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        # Compare the tag names and sorted attributes to determine if nodes are equal
        return self.tag_name == other.tag_name and sorted(self.attributes.items()) == sorted(other.attributes.items())

    def __hash__(self):
        return hash((self.tag_name, tuple(sorted(self.attributes.items()))))

    def __repr__(self):
        return f"Node(tag_name={self.tag_name}, attributes={self.attributes})"


def count_nodes(node):
    if node is None:
        return 0
    count = 1
    for child in node.get_children():
        count += count_nodes(child)  # Recursively count all children
    return count


# Utility to normalize HTML
def normalize_html(html_code):
    # Remove comments
    soup = BeautifulSoup(html_code, 'lxml')
    for element in soup(text=lambda text: isinstance(text, Comment)):
        element.extract()

    # Sort attributes for each tag
    for tag in soup.find_all(True):
        if tag.attrs:
            sorted_attrs = dict(sorted(tag.attrs.items()))
            tag.attrs = sorted_attrs

    # Convert to string and minify
    normalized_html = str(soup)
    normalized_html = htmlmin.minify(normalized_html, remove_empty_space=True, remove_comments=True)

    return normalized_html


# Function to convert a BeautifulSoup DOM into a tree structure for ZSS
def build_tree(soup_element):
    if soup_element is None:
        return None

    # Create a Node for the current element
    node = Node(soup_element.name, soup_element.attrs)

    # Recursively add child nodes for each child element
    for child in soup_element.children:
        if isinstance(child, str):
            continue  # Skip text nodes
        child_node = build_tree(child)
        if child_node:
            node.add_child(child_node)
    return node


# Compute ZSS-based Tree Edit Distance (TED) for two BeautifulSoup trees
def compute_tree_edit_distance(soup1, soup2):
    # Build trees for ZSS
    try:
        root1 = build_tree(soup1)
        root2 = build_tree(soup2)
    except RecursionError as e:
        return 1.  # Return maximum distance if recursion limit is reached

    # Define cost functions for insert, remove, and substitute operations
    def insert_cost(node):
        return 1  # Cost for insertion

    def remove_cost(node):
        return 1  # Cost for deletion

    def update_cost(node1, node2):
        return 0 if node1 == node2 else 1  # Cost for substitution (0 if equal, 1 if not)

    # Compute the Tree Edit Distance using ZSS
    ted = zss.distance(root1, root2, insert_cost=insert_cost, remove_cost=remove_cost, update_cost=update_cost,
                       get_children=lambda node: node.get_children(), )

    # Normalize the distance
    max_nodes = max(count_nodes(root1), count_nodes(root2))
    return float(ted / max_nodes)


def validate_html_bs4(html_code):
    try:
        soup = BeautifulSoup(html_code, 'html.parser')
        return True
    except Exception as e:
        return False


# Main function to calculate HTML similarity
def html_similarity(reference, prediction):
    if not validate_html_bs4(prediction):
        return {'dom_tree_edit_distance': 0., 'screenshot_ssim': 0.}

    # Normalize the HTML codes
    normalized_html1 = normalize_html(reference)
    normalized_html2 = normalize_html(prediction)

    # Parse the normalized HTML
    soup1 = BeautifulSoup(normalized_html1, 'lxml')
    soup2 = BeautifulSoup(normalized_html2, 'lxml')

    # Calculate structural similarity (ZSS-based Tree Edit Distance)
    dom_tree_edit_distance = max(0., min(1., 1. - compute_tree_edit_distance(soup1, soup2)))

    return {'dom_tree_edit_distance': dom_tree_edit_distance}


if __name__ == '__main__':
    pass

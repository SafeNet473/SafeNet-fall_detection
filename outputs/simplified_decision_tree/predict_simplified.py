"""Binary-only export. Preserve locked Python comparison semantics."""
import json
from pathlib import Path
import numpy as np

def predict(features):
    """Features must use the original 21-column order; only indices 5/15/18 are read."""
    tree=json.loads(Path(__file__).with_name("simplified_tree.json").read_text())["tree"]
    features=np.asarray(features,dtype=np.float32)
    if features.ndim!=2 or features.shape[1]!=21 or not np.isfinite(features).all():
        raise ValueError("Expected finite (N,21) feature matrix")
    result=np.empty(len(features),dtype=np.uint8)
    for i,row in enumerate(features):
        node=tree
        while "prediction" not in node:
            node=node["left" if row[node["feature_index"]] <= node["threshold"] else "right"]
        result[i]=node["prediction"]
    return result

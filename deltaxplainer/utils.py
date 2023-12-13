from sklearn.tree import _tree
import numpy as np
from typing import List, Optional

def extract_rules(
    tree: _tree.Tree,
    feature_names: List[str],
    class_names: Optional[List[str]],
    class_diff: int
) -> List[str]:
    """
    Extract rules from a decision tree.

    Args:
        tree (_tree.Tree): Model to extract rules from
        feature_names (List[str]): Feature names
        class_names (List[str], optional): Class names
        class_diff (int): Class to consider as disagreeing

    Returns:
        List[str]: Extracted rules
    """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []
    
    def recurse(node, path, paths):
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]
            
    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]
    
    rules = []
    for path in paths:
        rule = "if "
        
        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: "+str(np.round(path[-1][0][0][0],3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (Probability: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]
    
    delta_rules = []
    for rule in rules:
        if "then class: " + str(class_diff) in rule:
            delta_rules.append(rule)
    return delta_rules

def compress_rule(rule: str) -> str:
    """
    Compresses a rule extracted from a decision tree.

    Args:
        rule (str): Rule extracted from the decision tree

    Returns:
        str: Compressed rule
    """
    def parse_condition(condition):
        condition = condition.replace('(', '').replace(')', '')
        if '<=' in condition:
            feature, value = condition.split(' <= ')
            operator = '<='
        elif '>=' in condition:
            feature, value = condition.split(' >= ')
            operator = '>='
        elif '<' in condition:
            feature, value = condition.split(' < ')
            operator = '<'
        elif '>' in condition:
            feature, value = condition.split(' > ')
            operator = '>'
        else:
            print("Condition :",condition)
            raise()
            return "", "", 0
        return feature.strip(), operator, float(value.strip())
    condition_list = rule.split('if ')[1].split(' then ')[0].split(' and ')
    second_part = rule.split('if ')[1].split(' then ')[1]
    all_conds = []
    for condition in condition_list:
        all_conds.append(parse_condition(condition))
    new_conds = {}
    for ext_cond in all_conds:
        feature, operator, value = ext_cond
        key  = (feature,operator)
        if key not in new_conds:
            new_conds[key] = value
        else:
            if (operator in ["<", "<="] and value <= new_conds[key]) or (operator in [">", ">="] and value >= new_conds[key]):
                new_conds[key] = value

    substring = "if "
    i = 1
    for cond in new_conds:

        element = "(" + cond[0] +" "+ cond[1] +" "+  str(new_conds[cond]) + ")"
        substring += element
        if i == len(new_conds):
            substring += " then " + second_part
        else:
            substring += " and "
        i += 1
    return(substring)
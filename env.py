"""
Retrosynthesis environment.
"""
from __future__ import annotations 

from enum import Enum
import typing as ty

from rdkit import Chem 


class Node:
    def __init__(self, idx: int) -> None:
        self.idx = idx 


class Edge:
    def __init__(self, s_idx: int, e_idx: int) -> None:
        self.s_idx = s_idx
        self.e_idx = e_idx


class State:
    def __init__(self) -> None:
        self.nodes: ty.List[Node] = list()
        self.edges: ty.List[Edge] = list()


class Action: ...


class Env:
    def __init__(self, smiles: str) -> None:
        self.smiles = smiles
        self.state: State = initialize_state(smiles)

        self.done = False
        self.reward = 0

    def reset(self) -> None:
        self.state = initialize_state(self.smiles)

    def run_frame(self) -> None:
        print("Drawing frame not yet implemented!")
        return

    def step(self, action: Action) -> None:
        match action:
            case x if isinstance(x, Action): pass  # TODO
        
        self.run_frame()

        self.mutations += 1

        return self.reward, self.state, self.done 


def initialize_state(smiles: str) -> State:
    state = State()

    mol = Chem.MolFromSmiles(smiles)
    
    for atom in mol.GetAtoms():
        state.nodes.append(Node(atom.GetIdx()))
    
    for bond in mol.GetBonds():
        state.edges.append(Edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

    return state

"""Batched pUCT MCTS over learned dynamics (negamax backup, MinMax-normalized Q).

Legality masking is applied at the root only; interior nodes expand the
top-k prior actions of the full policy (MuZero standard — the learned
dynamics cannot produce a legal-move list)."""

from __future__ import annotations

import math
import threading

import numpy as np
import torch

from muzero.config import MuZeroConfig


class NetRunner:
    """Thread-safe batched inference wrapper around MuZeroNet."""

    def __init__(self, net, device):
        self.net = net
        self.device = device
        self.lock = threading.Lock()

    def initial(self, obs_batch: np.ndarray) -> dict:
        with self.lock, torch.inference_mode():
            self.net.eval()  # BatchNorm: batch-stats + running-stat mutation otherwise
            obs = torch.from_numpy(np.ascontiguousarray(obs_batch)).to(self.device)
            out = self.net.initial_inference(obs)
        return self._detach(out)

    def recurrent(self, hidden: torch.Tensor, actions: np.ndarray) -> dict:
        with self.lock, torch.inference_mode():
            self.net.eval()  # see initial()
            acts = torch.from_numpy(np.ascontiguousarray(actions)).to(self.device)
            out = self.net.recurrent_inference(hidden, acts)
        return self._detach(out)

    @staticmethod
    def _detach(out: dict) -> dict:
        keep = {"hidden": out["hidden"]}
        for k in ("policy_logits", "value", "reward"):
            if k in out:
                keep[k] = out[k].float().cpu().numpy()
        return keep


class MinMaxStats:
    def __init__(self):
        self.minimum, self.maximum = float("inf"), float("-inf")

    def update(self, value: float):
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node:
    __slots__ = (
        "prior",
        "visit_count",
        "value_sum",
        "reward",
        "hidden",
        "cand_actions",
        "cand_priors",
        "children",
        "prior_action",
    )

    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.reward = 0.0
        self.hidden = None  # row-tensor in the net's hidden batch
        self.cand_actions = None  # np.ndarray of action indices
        self.cand_priors = None  # np.ndarray aligned with cand_actions
        self.children = {}  # candidate position -> Node

    def expanded(self) -> bool:
        return self.cand_actions is not None

    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


class MCTS:
    def __init__(self, config: MuZeroConfig, rng=None):
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng(config.seed)

    def run(self, runner: NetRunner, roots_data: list, add_noise: bool) -> list:
        """roots_data: list of (obs (115,10,9) float32, legal action indices).
        Returns per game: ({action: visit_count}, root_value, search_kl) where
        search_kl = KL(visit distribution || raw pre-noise prior) — how much
        the search improved on the raw policy ("search gain")."""
        cfg = self.config
        obs_batch = np.stack([obs for obs, _ in roots_data])
        out = runner.initial(obs_batch)
        roots, stats, raw_priors = [], [], []
        for g, (_, legal) in enumerate(roots_data):
            root = Node(0.0)
            priors = _masked_softmax(out["policy_logits"][g], legal)
            raw_priors.append(priors)  # noise mixing below builds a new array
            if add_noise:
                noise = self.rng.dirichlet([cfg.dirichlet_alpha] * len(legal))
                priors = (
                    1 - cfg.exploration_fraction
                ) * priors + cfg.exploration_fraction * noise
            _expand(root, legal, priors, out["hidden"][g], reward=0.0)
            roots.append(root)
            stats.append(MinMaxStats())

        for _ in range(cfg.num_simulations):
            paths, hiddens, actions = [], [], []
            for g, root in enumerate(roots):
                node, path = root, [root]
                while node.expanded():
                    node = self._select_child(node, stats[g])
                    path.append(node)
                paths.append(path)
                hiddens.append(path[-2].hidden)
                actions.append(path[-1].prior_action)
            out = runner.recurrent(
                torch.stack(list(hiddens)), np.array(actions, dtype=np.int64)
            )
            for g, path in enumerate(paths):
                leaf = path[-1]
                logits = out["policy_logits"][g]
                topk = np.argpartition(logits, -cfg.interior_topk)[-cfg.interior_topk :]
                _expand(
                    leaf,
                    topk.astype(np.int64),
                    _masked_softmax(logits, topk),
                    out["hidden"][g],
                    reward=float(out["reward"][g]),
                )
                self._backup(path, float(out["value"][g]), stats[g])
        return [
            (
                {
                    int(root.cand_actions[p]): ch.visit_count
                    for p, ch in root.children.items()
                },
                root.value(),
                _search_kl(root, raw_priors[g]),
            )
            for g, root in enumerate(roots)
        ]

    def _select_child(self, node: Node, stats: MinMaxStats) -> Node:
        cfg = self.config
        n = node.cand_priors.shape[0]
        q = np.zeros(n, dtype=np.float32)
        visits = np.zeros(n, dtype=np.float32)
        for pos, ch in node.children.items():
            visits[pos] = ch.visit_count
            if ch.visit_count > 0:
                q[pos] = stats.normalize(ch.reward + cfg.discount * -ch.value())
        pb_c = (
            (
                math.log((node.visit_count + cfg.pb_c_base + 1) / cfg.pb_c_base)
                + cfg.pb_c_init
            )
            * math.sqrt(max(node.visit_count, 1))
            / (1.0 + visits)
        )
        pos = int(np.argmax(q + pb_c * node.cand_priors))
        child = node.children.get(pos)
        if child is None:
            child = Node(float(node.cand_priors[pos]))
            child.prior_action = int(node.cand_actions[pos])
            node.children[pos] = child
        return child

    def _backup(self, path: list, leaf_value: float, stats: MinMaxStats):
        cfg = self.config
        root = path[0]
        v = leaf_value  # perspective of the player to move at the leaf
        for node in reversed(path):
            node.value_sum += v
            node.visit_count += 1
            if node is not root:
                # Track the exact quantity _select_child normalizes: the
                # child's running-mean Q from its parent's perspective.
                stats.update(node.reward + cfg.discount * -node.value())
            v = node.reward + cfg.discount * -v


def _expand(node: Node, actions: np.ndarray, priors: np.ndarray, hidden, reward: float):
    node.cand_actions = np.asarray(actions, dtype=np.int64)
    node.cand_priors = np.asarray(priors, dtype=np.float32)
    node.hidden = hidden
    node.reward = reward


def _search_kl(root: Node, raw_prior: np.ndarray) -> float:
    """KL(root visit distribution || raw pre-noise prior), >= 0."""
    total = sum(ch.visit_count for ch in root.children.values())
    if total <= 0:
        return 0.0
    kl = 0.0
    for pos, ch in root.children.items():
        if ch.visit_count == 0:
            continue
        pi = ch.visit_count / total
        kl += pi * math.log(pi / max(float(raw_prior[pos]), 1e-12))
    return float(kl)


def _masked_softmax(logits: np.ndarray, indices: np.ndarray) -> np.ndarray:
    x = logits[indices].astype(np.float64)
    x = np.exp(x - x.max())
    return (x / x.sum()).astype(np.float32)

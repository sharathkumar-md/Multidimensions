from __future__ import annotations

import torch

from src.conversational import route_query, rewrite_query, simple_reply

_MODEL_ID = "Qwen/Qwen3-8B"


# ── fake model / tokenizer ────────────────────────────────────────────────────
# Drives the real _chat() path on CPU. The tokenizer pops a canned string from a
# shared queue on each decode() call; apply_chat_template records the messages so
# tests can assert what was actually sent to the model.

class _FakeInputs(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    eos_token_id = 0

    def __init__(self, responses: list[str], records: list):
        self._responses = responses
        self.records = records

    def apply_chat_template(self, messages, **kwargs):
        self.records.append(messages)
        return "\n".join(m["content"] for m in messages)

    def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
        n = max(min(len(text), max_length or 8), 1)
        return _FakeInputs(input_ids=torch.arange(n).unsqueeze(0))

    def decode(self, ids, skip_special_tokens=True):
        return self._responses.pop(0)


class _FakeModel:
    device = "cpu"

    def generate(self, input_ids=None, **kwargs):
        # append a single dummy "new" token; decode() ignores it and returns canned text
        new = torch.zeros((1, 1), dtype=input_ids.dtype)
        return torch.cat([input_ids, new], dim=1)

    def eval(self):
        return self


def _pair(*responses: str):
    records: list = []
    return _FakeModel(), _FakeTokenizer(list(responses), records), records


# ── 1. router: route_query ──────────────────────────────────────────────────

def test_router_no_skips_retrieval():
    model, tok, _ = _pair("NONE")
    assert route_query("hi there", model, tok, _MODEL_ID) == "NONE"


def test_router_yes_triggers_retrieval():
    model, tok, _ = _pair("LOCAL")
    assert route_query("what linear motors do you have?", model, tok, _MODEL_ID) == "LOCAL"


def test_router_no_prefix_phrase_skips():
    # model elaborates but still leads with "none"
    model, tok, _ = _pair("NONE, that's just a greeting.")
    assert route_query("thanks!", model, tok, _MODEL_ID) == "NONE"


def test_router_defaults_to_retrieval_on_garbage():
    # ambiguous / empty output must NOT skip the catalog (fail-safe toward retrieval)
    model, tok, _ = _pair("")
    assert route_query("?", model, tok, _MODEL_ID) == "LOCAL"


# ── 2. contextual query rewriter ──────────────────────────────────────────────

def test_rewrite_no_history_returns_question_unchanged():
    # with no history the model must NOT be invoked at all
    model, tok, records = _pair("SHOULD NOT BE USED")
    q = "what is the stroke length?"
    assert rewrite_query(q, "", model, tok, _MODEL_ID) == q
    assert records == []  # model never called


def test_rewrite_resolves_followup_with_history():
    model, tok, records = _pair("LinMot P01 linear motor stroke length")
    out = rewrite_query("what's its stroke?", "• linear motors → LinMot P01 series", model, tok, _MODEL_ID)
    assert out == "LinMot P01 linear motor stroke length"
    # the history must have been passed into the rewrite prompt
    assert any("LinMot P01" in m["content"] for m in records[0])


def test_rewrite_strips_surrounding_quotes():
    model, tok, _ = _pair('"LinMot P01 stroke length"')
    out = rewrite_query("its stroke?", "discussing LinMot P01", model, tok, _MODEL_ID)
    assert out == "LinMot P01 stroke length"


def test_rewrite_falls_back_on_runaway_output():
    # model returns junk far longer than the 4*len+200 guard -> keep the original
    q = "its stroke?"
    junk = "x" * (4 * len(q) + 500)
    model, tok, _ = _pair(junk)
    assert rewrite_query(q, "some history", model, tok, _MODEL_ID) == q


def test_rewrite_falls_back_on_empty_output():
    model, tok, _ = _pair("")
    q = "its stroke?"
    assert rewrite_query(q, "some history", model, tok, _MODEL_ID) == q


def test_rewrite_passes_through_standalone_query_unchanged():
    # when the model (correctly) echoes an already-standalone query, no mangling
    q = "what linear motors do you sell?"
    model, tok, _ = _pair(q)
    assert rewrite_query(q, "some prior history", model, tok, _MODEL_ID) == q


def test_rewrite_prompt_carries_anti_vague_rules_and_examples():
    # guards the weak-spot fix: the rewrite system prompt must keep the
    # "standalone unchanged" + "never vaguer" rules and the few-shot examples
    model, tok, records = _pair("anything")
    rewrite_query("its stroke?", "discussing LinMot P01", model, tok, _MODEL_ID)
    system = records[0][0]["content"]
    assert "EXACTLY unchanged" in system          # don't rewrite standalone queries
    assert "vaguer" in system                       # don't broaden specific questions
    assert "the same" in system                     # resolve 'the same'-style anaphora
    assert system.count("Query:") >= 3              # few-shot examples present


# ── 3. simple reply (no-retrieval path) ───────────────────────────────────────

def test_simple_reply_returns_model_text():
    model, tok, records = _pair("Hi! I help you find products to pitch.")
    out = simple_reply("hello", "", model, tok, _MODEL_ID)
    assert out == "Hi! I help you find products to pitch."
    # system prompt + the user question, no history context message
    roles = [m["role"] for m in records[0]]
    assert roles == ["system", "user"]


def test_simple_reply_injects_history_context():
    model, tok, records = _pair("Sure, anything else about that motor?")
    simple_reply("and the price?", "• motors → LinMot P01", model, tok, _MODEL_ID)
    contents = [m["content"] for m in records[0]]
    assert any("LinMot P01" in c for c in contents)
    assert any("and the price?" == c for c in contents)

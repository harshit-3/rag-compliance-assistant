"""Fast unit tests (no models, no network)."""
from ragkit.ingest import Chunk, _normalise, split_long
from ragkit.rag import REFUSAL, RAGPipeline, validate_citations
from ragkit.retrieval import _REG_REF, BM25Retriever, HybridRetriever, rrf, tokenize


def _chunk(i, section, text, path=None):
    return Chunk(id=f"d#{i}", corpus="t", doc_id="d", doc_title="Doc", section=section,
                 path=path or section, page=i + 1, text=text)


CHUNKS = [
    _chunk(0, "617. Managing risks to health and safety",
           "A person conducting a business or undertaking at a mine must manage risks to health and safety."),
    _chunk(1, "653. Monitoring and testing of ventilation system",
           "The mine operator must keep a record of all monitoring and testing of the ventilation system "
           "for at least 7 years."),
    _chunk(2, "625B. Duty of mine operator to provide information to contractor",
           "The mine operator must provide the contractor with information about hazards at the mine."),
    _chunk(3, "3.1.1 Hand hygiene", "Hand hygiene is the single most important strategy in preventing infection."),
]


def test_normalise_keeps_list_items_but_not_inline_refs():
    raw = "Penalty for this subregulation:\n(a) for an individual, a fine of $7 000;\n(b) for a body corporate. " \
          "See subregulation (1) for details."
    out = _normalise(raw)
    lines = out.split("\n")
    assert lines[1].startswith("(a) for an individual")
    assert lines[2].startswith("(b) for a body corporate")
    assert "subregulation (1) for details" in out  # inline reference not split


def test_split_long_respects_limit_and_overlap():
    c = _chunk(0, "Big", "\n".join(f"({i}) " + "word " * 60 for i in range(1, 15)))
    parts = split_long(c, max_words=200, overlap=30)
    assert len(parts) > 1
    assert all(len(p.text.split()) <= 200 for p in parts)
    assert parts[0].section.endswith("(part 1)")


def test_tokenize_keeps_regulation_numbers():
    assert "625b" in tokenize("What does regulation 625B require?")
    assert "3.1.1" in tokenize("section 3.1.1 hand hygiene")
    assert "the" not in tokenize("the mine")


def test_reg_reference_regex():
    refs = [m.upper() for m in _REG_REF.findall("See regulation 617, reg. 625B and r 653(2).")]
    assert refs == ["617", "625B", "653"]


def test_rrf_rewards_agreement():
    fused = rrf([[1, 2, 3], [2, 1, 4]])
    assert fused[1] == fused[2] > fused[3] > 0
    assert max(fused, key=fused.get) in (1, 2)


def test_bm25_finds_exact_terms():
    r = BM25Retriever(CHUNKS)
    assert r.search("ventilation testing records years", 1)[0][0] == 1


def test_reference_pinning_without_dense_or_reranker():
    r = HybridRetriever(CHUNKS, dense_model=None, reranker=None)
    hits = r.search("What does regulation 625B require?", k=2, mode="hybrid")
    assert hits[0].chunk.section.startswith("625B.")
    assert hits[0].sources.get("reference") == 1


def test_validate_citations_strips_hallucinated_numbers():
    text, valid, invalid = validate_citations("Keep records for 7 years [2]. Also see [9].", n_sources=3)
    assert valid == [2] and invalid == [9]
    assert "[9]" not in text and "[2]" in text


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def generate(self, system, user, temperature=0.1, json_mode=False):
        from ragkit.llm import LLMResponse
        assert "Answer ONLY from the numbered sources" in system
        assert "[1]" in user
        return LLMResponse(self.reply, "fake", 0.0)


def test_pipeline_grounded_answer():
    r = HybridRetriever(CHUNKS, dense_model=None, reranker=None)
    a = RAGPipeline(r, FakeLLM("Records must be kept for at least 7 years [1]."), "mines", k=3,
                    mode="hybrid").ask("How long are ventilation records kept?")
    assert a.grounded and not a.refused
    assert a.citations[0]["section"].startswith("653.")


def test_pipeline_flags_uncited_answer_and_refusal():
    r = HybridRetriever(CHUNKS, dense_model=None, reranker=None)
    a = RAGPipeline(r, FakeLLM("Probably 5 years."), "mines", mode="hybrid").ask("ventilation records?")
    assert not a.grounded
    a = RAGPipeline(r, FakeLLM(REFUSAL), "mines", mode="hybrid").ask("hand hygiene in France?")
    assert a.refused and a.grounded


def test_no_hits_refuses_without_llm_call():
    class Boom:
        def generate(self, *a, **k):
            raise AssertionError("LLM must not be called")
    r = HybridRetriever(CHUNKS, dense_model=None, reranker=None)
    a = RAGPipeline(r, Boom(), "mines", mode="bm25").ask("zzzz qqqq")
    assert a.refused and a.model == "none"

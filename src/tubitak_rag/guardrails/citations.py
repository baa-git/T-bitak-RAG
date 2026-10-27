"""Citation validation: reject uncited answers and citations to sources not given (FR-GR3)."""

from pydantic import BaseModel, ConfigDict

from tubitak_rag.generation.schema import LlmOutput, cited_numbers

UNCITED = "atifsiz_cevap"
INVALID_CITATION = "gecersiz_atif"


class CitationVerdict(BaseModel):
    """Whether the output's citations are acceptable, with a reason code when not."""

    model_config = ConfigDict(frozen=True)

    ok: bool
    reason: str | None


def validate(output: LlmOutput, n_sources: int) -> CitationVerdict:
    """Reject an answer with no citations or with any number outside ``1..n_sources``."""
    if output.cevaplanabilir and not output.atiflar:
        return CitationVerdict(ok=False, reason=UNCITED)
    if any(not 1 <= number <= n_sources for number in cited_numbers(output)):
        return CitationVerdict(ok=False, reason=INVALID_CITATION)
    return CitationVerdict(ok=True, reason=None)

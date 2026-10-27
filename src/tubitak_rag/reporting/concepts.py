"""One-paragraph descriptions of the experiment ladder's rungs.

Every experiment config names its concept (``experiment.concept``). The report generator puts
the concept's paragraph at the top of ``report.md``: what this configuration is and
which question the rung answers. Fix the wording here, never in a generated report
(regenerate with ``rag report --all``).
"""

from pydantic import BaseModel, ConfigDict


class Concept(BaseModel):
    """One rung of the experiment ladder."""

    model_config = ConfigDict(frozen=True)

    key: str
    summary: str


_FIXED = Concept(
    key="fixed_chunking",
    summary=(
        "Belgeler içeriğe bakılmadan 512 tokenlık pencerelere bölünür (64 token örtüşme). Her "
        "pencere embedding modeliyle vektöre çevrilir; soruya kosinüs benzerliği en yüksek beş "
        "parça dil modeline verilir. Madde sınırı, belge adı ve program kodu parçaya eklenmez. "
        "Bu basamak, sonraki her tekniğin karşılaştırıldığı taban çizgisidir; amaç en basit RAG "
        "kurulumunun bu belgelerde ne kadar başarılı olduğunu görmektir."
    ),
)

_RERANKER = Concept(
    key="reranker",
    summary=(
        "Taban çizgisiyle aynı parçalar ve aynı dense arama; tek fark, aramanın getirdiği 30 "
        "adayın bir cross-encoder reranker (bge-reranker-v2-m3) ile yeniden sıralanmasıdır. "
        "Reranker soruyu ve parçayı birlikte okuduğu için daha isabetlidir ama daha yavaştır ve "
        "yalnız ilk aşamanın getirdiği havuzu sıralayabilir. Amaç, yeniden sıralamanın kötü "
        "parçalamayı ne ölçüde telafi edebildiğini görmektir."
    ),
)

_STRUCTURAL = Concept(
    key="structural_context",
    summary=(
        "Mevzuatta her madde, program sayfalarında her bölüm ve her SSS çifti ayrı parçadır; "
        "uzun maddeler fıkra sınırından bölünür. Aranan metnin başına bir bağlam başlığı eklenir: "
        'belge adı, program kodu ve hiyerarşi (ör. "1507 Uygulama Esasları (1507) › Madde 7 – '
        'Destek oranı ve sınırı"). Arama yine yalnız dense, reranker yok. Taban çizgisiyle '
        "karşılaştırılır; amaç parça sınırlarının ve bağlam başlığının etkisini görmektir."
    ),
)

_HYBRID = Concept(
    key="hybrid_rrf",
    summary=(
        "Parçalar bir önceki basamakla aynı; aramaya Türkçe kök bulmalı BM25 eklenir. Dense ve "
        "BM25 sonuç listeleri skorlarıyla değil sıralarıyla birleştirilir (Reciprocal Rank "
        'Fusion, k = 60). Dense anlamı, BM25 ise "1507", "4691", "%75" gibi kodları ve tam '
        "terimleri yakalar. Reranker yok. Amaç kelime aramasını eklemenin katkısını görmektir."
    ),
)

_FULL = Concept(
    key="full_pipeline",
    summary=(
        "Önceki basamakların hepsi birlikte: yapı bazlı parçalar ve bağlam başlığı, hibrit arama "
        "(dense + BM25, RRF) ve 30 adayı yeniden sıralayan reranker. Dil modeline en iyi beş "
        "parça gider. Kapsam kontrolü, eşik kapısı, atıf ve sayı denetimi bütün basamaklarda "
        "açıktır; burada eşik kapısı dense skoru yerine reranker skoruna bakar. Hibrit aramayla "
        "karşılaştırılır."
    ),
)

_EMBEDDING = Concept(
    key="embedding_comparison",
    summary=(
        "Tam sistemle birebir aynı; yalnız embedding modeli EmbeddingGemma 2 (768 boyut) yerine "
        "BAAI/bge-m3 (1024 boyut). Parçalar, BM25, reranker, eşik ve dil modeli değişmez. "
        "Reranker'lı bir sistemde embedding modelinin etkisi en çok aday havuzunda görünür; bu "
        "yüzden Recall@10 ve kanıtın 30 adayda hiç bulunmadığı soru sayısı özellikle izlenir."
    ),
)

CONCEPTS: dict[str, Concept] = {
    c.key: c for c in (_FIXED, _RERANKER, _STRUCTURAL, _HYBRID, _FULL, _EMBEDDING)
}
DEFAULT_CONCEPT = "full_pipeline"


def concept_for(key: str) -> Concept:
    """The concept named in the experiment config; unknown keys fall back to the full pipeline."""
    return CONCEPTS.get(key, CONCEPTS[DEFAULT_CONCEPT])

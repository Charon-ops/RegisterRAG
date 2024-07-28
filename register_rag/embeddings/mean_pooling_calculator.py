from typing import List, Dict, Any

from .embedding_getter import EmbeddingGetter
from ..documents import Document


class MeanPoolingCalculator:
    @classmethod
    async def mean_pooling(
        self,
        embeds=List[List[float]],
    ) -> List[float]:
        embed = [sum(x) / len(x) for x in zip(*embeds)]
        return embed

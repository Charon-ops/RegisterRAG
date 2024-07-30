from typing import List, Dict, Any

from .embedding_getter import EmbeddingGetter
from ..documents import Document


class MeanPoolingCalculator:
    """
    A class to calculate the mean pooling of the embeddings.

    You can use the `mean_pooling` method to calculate the mean pooling of the embeddings.
    """

    @classmethod
    async def mean_pooling(
        self,
        embeds: List[List[float]],
    ) -> List[float]:
        """
        A method to calculate the mean pooling of the embeddings.

        Args:
            embeds (List[List[float]]): A list of embeddings to calculate the mean pooling.

        Returns:
            List[float]: The mean pooled embedding of the embeddings.
        """
        embed = [sum(x) / len(x) for x in zip(*embeds)]
        return embed

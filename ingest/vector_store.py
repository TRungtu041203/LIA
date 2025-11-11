# vector_store.py
from __future__ import annotations
from typing import Dict, List, Iterable, Optional, Union, Sequence, Any, Tuple
from qdrant_client import QdrantClient, models
from dataclasses import dataclass

DistanceLike = Union[str, models.Distance]

def _to_distance(d: DistanceLike) -> models.Distance:
    if isinstance(d, models.Distance):
        return d
    d = str(d).strip().upper()
    if d in {"COSINE", "COS"}:
        return models.Distance.COSINE
    if d in {"DOT", "IP", "INNER"}:
        return models.Distance.DOT
    if d in {"EUCLID", "L2"}:
        return models.Distance.EUCLID
    raise ValueError(f"Unsupported distance: {d}")

@dataclass(frozen=True)
class VectorSpace:
    """Definition for one vector space."""
    size: int
    distance: models.Distance

class VectorStore:
    """
    A thin, safe wrapper around Qdrant for creating collections with either:
      - a single unnamed vector (classic), or
      - multiple named vectors (multimodal).
    Also handles: payload indexes, upserts, search, delete, info, etc.
    """

    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        url: Optional[str] = "http://localhost:6333",
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.client = client or QdrantClient(url=url, api_key=api_key, timeout=timeout)

    # ---------- COLLECTION CREATION ----------

    def create_or_recreate_collection(
        self,
        collection_name: str,
        vectors: Union[
            Tuple[int, DistanceLike],                                # single-vector: (size, distance)
            Dict[str, Union[Tuple[int, DistanceLike], VectorSpace]]  # named vectors
        ],
        *,
        on_disk_payload: bool = False,
        hnsw_config: Optional[models.HnswConfigDiff] = None,
        optimizers_config: Optional[models.OptimizersConfigDiff] = None,
        quantization_config: Optional[models.QuantizationConfig] = None,
        shard_number: Optional[int] = None,
        replication_factor: Optional[int] = None,
        write_consistency_factor: Optional[int] = None,
        force: bool = False,
        strict: bool = True,
    ) -> None:
        """
        Create a collection. If it exists:
          - force=True  -> recreate (drops data!)
          - force=False -> no-op
        """
        exists = self.client.collection_exists(collection_name)

        if exists and not force:
            return

        vectors_config = self._build_vectors_config(vectors, strict=strict)

        if exists and force:
            self.client.update_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                on_disk_payload=on_disk_payload,
                hnsw_config=hnsw_config,
                optimizers_config=optimizers_config,
                quantization_config=quantization_config,
                shard_number=shard_number,
                replication_factor=replication_factor,
                write_consistency_factor=write_consistency_factor,
            )
        else:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                on_disk_payload=on_disk_payload,
                hnsw_config=hnsw_config,
                optimizers_config=optimizers_config,
                quantization_config=quantization_config,
                shard_number=shard_number,
                replication_factor=replication_factor,
                write_consistency_factor=write_consistency_factor,
            )

    def _build_vectors_config(
        self,
        vectors: Union[Tuple[int, DistanceLike], Dict[str, Union[Tuple[int, DistanceLike], VectorSpace]]],
        strict: bool = True,
    ) -> Union[models.VectorParams, Dict[str, models.VectorParams]]:
        # Single-vector path
        if isinstance(vectors, tuple):
            size, dist = vectors
            if strict:
                assert isinstance(size, int) and size > 0, "size must be positive int"
            return models.VectorParams(size=size, distance=_to_distance(dist))

        # Named-vectors path
        cfg: Dict[str, models.VectorParams] = {}
        for name, spec in vectors.items():
            if strict:
                assert isinstance(name, str) and name, "vector name must be non-empty str"
            if isinstance(spec, tuple):
                size, dist = spec
                cfg[name] = models.VectorParams(size=int(size), distance=_to_distance(dist))
            elif isinstance(spec, VectorSpace):
                cfg[name] = models.VectorParams(size=int(spec.size), distance=spec.distance)
            else:
                raise TypeError(f"Unexpected vector spec for '{name}': {type(spec)}")
        if strict:
            assert len(cfg) >= 1, "must define at least one vector"
            assert len(cfg) == len(set(cfg.keys())), "duplicate vector names"
            for name, vp in cfg.items():
                assert vp.size > 0, f"size must be >0 for '{name}'"
        return cfg

    # ---------- PAYLOAD INDEXES ----------

    def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        schema: models.PayloadSchemaType = models.PayloadSchemaType.KEYWORD,
    ) -> None:
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=schema,
        )

    # ---------- UPSERT / DELETE / FETCH ----------

    def upsert_points(
        self,
        collection_name: str,
        ids: Iterable[Union[int, str]],
        vectors: Iterable[Union[List[float], Dict[str, List[float]]]],
        payloads: Optional[Iterable[Optional[Dict[str, Any]]]] = None,
        batch_size: int = 512,
    ) -> None:
        """
        Upsert in batches. For named-vectors collections, pass vector as dict
        e.g. {"image": [...], "caption": [...]}.
        You may omit some named vectors per point (Qdrant allows sparse named vectors).
        """
        from itertools import islice

        def _batched(iterable, n):
            it = iter(iterable)
            while True:
                batch = list(islice(it, n))
                if not batch:
                    break
                yield batch

        ids_iter = iter(ids)
        vecs_iter = iter(vectors)
        payl_iter = iter(payloads) if payloads is not None else None

        while True:
            _ids = list(islice(ids_iter, batch_size))
            if not _ids:
                break
            _vecs = [v for v in islice(vecs_iter, len(_ids))]
            _payl = [p for p in islice(payl_iter, len(_ids))] if payl_iter is not None else [None] * len(_ids)

            points = [
                models.PointStruct(id=_ids[i], vector=_vecs[i], payload=_payl[i])
                for i in range(len(_ids))
            ]
            self.client.upsert(collection_name=collection_name, points=points)

    def delete_points(
        self,
        collection_name: str,
        point_ids: Iterable[Union[int, str]],
    ) -> None:
        self.client.delete(collection_name=collection_name, points=point_ids)

    def retrieve_points(
        self,
        collection_name: str,
        point_ids: Iterable[Union[int, str]],
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> List[models.Record]:
        return self.client.retrieve(
            collection_name=collection_name,
            ids=list(point_ids),
            with_payload=with_payload,
            with_vectors=with_vectors,
        )

    # ---------- SEARCH ----------

    def search(
        self,
        collection_name: str,
        query_vector: Union[List[float], Dict[str, List[float]]],
        *,
        vector_name: Optional[str] = None,   # set for named vectors
        limit: int = 10,
        filter: Optional[models.Filter] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        score_threshold: Optional[float] = None,
        offset: Optional[int] = None,
    ) -> List[models.ScoredPoint]:
        """
        For named vectors, either pass:
          - vector_name="image", query_vector=[...], or
          - query_vector=models.NamedVector(name="image", vector=[...])  (we build it for you below)
        """
        # Allow either explicit name or a dict
        if vector_name and isinstance(query_vector, list):
            nv = models.NamedVector(name=vector_name, vector=query_vector)
            return self.client.search(
                collection_name=collection_name,
                query_vector=nv,
                limit=limit,
                filter=filter,
                with_payload=with_payload,
                with_vectors=with_vectors,
                score_threshold=score_threshold,
                offset=offset,
            )
        elif isinstance(query_vector, dict):
            # When you pass a dict, Qdrant expects exactly one named vector inside
            return self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                filter=filter,
                with_payload=with_payload,
                with_vectors=with_vectors,
                score_threshold=score_threshold,
                offset=offset,
            )
        else:
            # Single-vector collection
            return self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,  # plain list[float]
                limit=limit,
                filter=filter,
                with_payload=with_payload,
                with_vectors=with_vectors,
                score_threshold=score_threshold,
                offset=offset,
            )

    # ---------- UTILITIES ----------

    def collection_info(self, collection_name: str) -> models.CollectionInfo:
        return self.client.get_collection(collection_name=collection_name)

    def exists(self, collection_name: str) -> bool:
        return self.client.collection_exists(collection_name)

    def count(self, collection_name: str, exact: bool = False, filters: Optional[models.Filter] = None) -> int:
        res = self.client.count(collection_name=collection_name, exact=exact, filter=filters)
        return res.count

    def assert_vector_dim(
        self,
        collection_name: str,
        vector_name: Optional[str],
        sample_vector: List[float],
    ) -> None:
        """
        Quick guard: ensures your embedding dimension matches the collection contract.
        """
        info = self.collection_info(collection_name)
        if isinstance(info.config.params.vectors, dict):
            assert vector_name, "Must provide vector_name for named-vectors collection"
            expected = info.config.params.vectors[vector_name].size
        else:
            expected = info.config.params.vectors.size
        got = len(sample_vector)
        if got != expected:
            raise ValueError(f"Embedding dim mismatch for '{collection_name}'"
                             f"{'/' + vector_name if vector_name else ''}: expected {expected}, got {got}")

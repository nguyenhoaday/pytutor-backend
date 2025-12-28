"""
Qdrant Cloud RAG Module - Vector DB và Embedding
Tích hợp với Qdrant Cloud để lưu trữ và truy xuất code mẫu + reference code
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models
from infra.utils.normalize_code import normalize_code
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import uuid
import os
import re
import logging
import ast

logger = logging.getLogger(__name__)

MAX_CHUNK_SIZE = 800
VECTOR_SIZE = 384


@dataclass
class RetrievedCode:
    """Kết quả truy xuất từ Qdrant"""
    id: str
    problem_id: str
    code: str
    similarity: float
    chunk_idx: int
    is_passed: bool = False
    user_uuid: str = ""
    total_chunks: int = 1
    full_code: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class QdrantTutor:
    """
    Hệ thống RAG sử dụng Qdrant Cloud để lưu trữ và truy xuất code mẫu.
    Sử dụng local embedding model (all-MiniLM-L6-v2) để giảm chi phí.
    """

    COLLECTION_SUBMISSIONS = "student_submissions"
    
    def __init__(self):
        """Khởi tạo Qdrant client và embedding model"""
        qdrant_url = os.environ.get("QDRANT_URL", "")
        qdrant_api_key = os.environ.get("QDRANT_API_KEY", "")
        
        if qdrant_url and qdrant_api_key:
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
            )
            logger.info(f"Connected to Qdrant Cloud: {qdrant_url}")
        else:
            self.client = QdrantClient(":memory:")
            logger.warning("QDRANT_URL not set, using in-memory Qdrant")
        
        try:
            # Lazy import to keep FastAPI startup fast (torch/transformers can be slow to import)
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.vector_size = VECTOR_SIZE
            logger.info("Loaded embedding model: all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        self._init_collection()
    
    def _init_collection(self):
        """Khởi tạo collection với các payload indexes cần thiết"""
        try:
            existing = [c.name for c in self.client.get_collections().collections]
            
            if self.COLLECTION_SUBMISSIONS not in existing:
                self.client.create_collection(
                    collection_name=self.COLLECTION_SUBMISSIONS,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.COLLECTION_SUBMISSIONS}")
            
            # Tạo indexes cho filtering
            for field_name, field_type in [
                ("problem_id", models.PayloadSchemaType.KEYWORD),
                ("is_passed", models.PayloadSchemaType.BOOL),
                ("user_uuid", models.PayloadSchemaType.KEYWORD),
            ]:
                try:
                    self.client.create_payload_index(
                        collection_name=self.COLLECTION_SUBMISSIONS,
                        field_name=field_name,
                        field_schema=field_type
                    )
                except Exception:
                    pass  # Index có thể đã tồn tại
                    
        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            raise
    
    def _chunk_code(self, code: str) -> List[str]:
        """
        Chia code thông minh theo function (def) để giữ ngữ cảnh logic.
        Sử dụng AST để parse chính xác các function definitions.
        Nếu function quá dài, chia theo dòng để tránh cắt giữa dòng.
        """
        chunks = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Nếu code không parse được, chia theo size
            for i in range(0, len(code), MAX_CHUNK_SIZE):
                chunks.append(code[i:i + MAX_CHUNK_SIZE])
            return [c for c in chunks if c.strip()]
        
        # Extract function definitions
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Lấy source code của function
                func_code = ast.get_source_segment(code, node)
                if func_code:
                    functions.append(func_code.strip())
        
        if functions:
            # Chia theo function
            for func_code in functions:
                if len(func_code) <= MAX_CHUNK_SIZE:
                    chunks.append(func_code)
                else:
                    # Function quá dài, chia theo dòng để giữ ngữ cảnh
                    lines = func_code.split('\n')
                    current_chunk = ''
                    for line in lines:
                        if len(current_chunk + '\n' + line) > MAX_CHUNK_SIZE:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = line
                        else:
                            current_chunk += '\n' + line if current_chunk else line
                    if current_chunk:
                        chunks.append(current_chunk.strip())
            
            # Thêm phần code ngoài function (top-level statements)
            # Loại bỏ các function đã extract
            remaining = code
            for func in functions:
                remaining = remaining.replace(func, '', 1)  # Replace only first occurrence
            remaining = remaining.strip()
            if remaining:
                # Tương tự cho remaining
                lines = remaining.split('\n')
                current_chunk = ''
                for line in lines:
                    if len(current_chunk + '\n' + line) > MAX_CHUNK_SIZE:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = line
                    else:
                        current_chunk += '\n' + line if current_chunk else line
                if current_chunk:
                    chunks.append(current_chunk.strip())
        else:
            # Không có function, chia theo size
            for i in range(0, len(code), MAX_CHUNK_SIZE):
                chunks.append(code[i:i + MAX_CHUNK_SIZE])
        
        return [c for c in chunks if c.strip()]
    
    def add_submission(
        self,
        problem_id: str,
        code_content: str,
        is_passed: bool = False,
        user_uuid: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Thêm submission vào Qdrant với đầy đủ metadata.
        Sử dụng chunking thông minh theo function.
        
        Args:
            problem_id: ID của bài toán
            code_content: Code cần thêm
            is_passed: True nếu code đã pass tất cả test cases (dùng làm reference)
            user_uuid: UUID của user (để track lịch sử)
            metadata: Metadata bổ sung (error_type, hints_received, etc.)
            
        Returns:
            List các point IDs đã thêm
        """
        chunks = self._chunk_code(code_content)
        total_chunks = len(chunks)
        
        points = []
        point_ids = []
        
        for i, chunk in enumerate(chunks):
            point_id = str(uuid.uuid4())
            vector = self.model.encode(chunk).tolist()
            
            payload = {
                "problem_id": str(problem_id),
                "code": normalize_code(chunk),
                "chunk_idx": i,
                "total_chunks": total_chunks,
                "is_passed": is_passed,
                "user_uuid": user_uuid or "anonymous",
                "full_code": normalize_code(code_content),
                **(metadata or {})
            }
            
            points.append(models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            ))
            point_ids.append(point_id)
        
        self.client.upsert(
            collection_name=self.COLLECTION_SUBMISSIONS,
            points=points
        )
        
        logger.info(f"Added {len(points)} chunks for problem {problem_id} (passed={is_passed})")
        return point_ids
    
    def add_dataset(self, problem_id: str, code_content: str) -> List[str]:
        """
        Thêm code mẫu (dataset) vào hệ thống.
        Ghi chú: code mẫu luôn được đánh dấu is_passed=True.
        """
        return self.add_submission(
            problem_id=problem_id,
            code_content=code_content,
            is_passed=True,
            user_uuid="system_dataset"
        )
    
    
    def get_similar_wrong_submissions(
        self,
        student_code: str,
        problem_id: str,
        top_k: int = 3
    ) -> List[RetrievedCode]:
        """
        Tìm các submission sai tương tự - hữu ích để phát hiện lỗi phổ biến.
        
        Returns:
            List các submission sai tương tự
        """
        query_vector = self.model.encode(student_code).tolist()
        
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="problem_id",
                    match=models.MatchValue(value=str(problem_id))
                ),
                models.FieldCondition(
                    key="is_passed",
                    match=models.MatchValue(value=False)
                )
            ]
        )
        
        results = self.client.query_points(
            collection_name=self.COLLECTION_SUBMISSIONS,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k
        ).points
        
        return [
            RetrievedCode(
                id=str(hit.id),
                problem_id=hit.payload.get("problem_id", ""),
                code=hit.payload.get("code", ""),
                similarity=hit.score,
                chunk_idx=hit.payload.get("chunk_idx", 0),
                is_passed=False,
                user_uuid=hit.payload.get("user_uuid", ""),
                total_chunks=hit.payload.get("total_chunks", 1),
                full_code=hit.payload.get("full_code", ""),  # Thêm full_code
                metadata={k: v for k, v in hit.payload.items()
                         if k not in ["problem_id", "code", "chunk_idx", "is_passed", "user_uuid", "total_chunks", "full_code"]}
            )
            for hit in results
        ]
    
    
    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        problem_id: Optional[str] = None,
        only_passed: bool = False
    ) -> List[RetrievedCode]:
        """
        Tìm kiếm semantic trong collection.
        
        Args:
            query: Text query (code hoặc mô tả)
            top_k: Số kết quả
            problem_id: Optional filter theo problem_id
            only_passed: Chỉ lấy code đã pass
        """
        query_vector = self.model.encode(query).tolist()
        
        must_conditions = []
        if problem_id:
            must_conditions.append(
                models.FieldCondition(
                    key="problem_id",
                    match=models.MatchValue(value=str(problem_id))
                )
            )
        if only_passed:
            must_conditions.append(
                models.FieldCondition(
                    key="is_passed",
                    match=models.MatchValue(value=True)
                )
            )
        
        search_filter = models.Filter(must=must_conditions) if must_conditions else None
        
        results = self.client.query_points(
            collection_name=self.COLLECTION_SUBMISSIONS,
            query=query_vector,
            query_filter=search_filter,
            limit=top_k
        ).points
        
        # Prefer returning chunk-level code if available, otherwise fall back to full_code
        return [
            RetrievedCode(
                id=str(hit.id),
                problem_id=hit.payload.get("problem_id", ""),
                code=hit.payload.get("code", hit.payload.get("full_code", "")),
                similarity=hit.score,
                chunk_idx=hit.payload.get("chunk_idx", 0),
                is_passed=hit.payload.get("is_passed", False),
                user_uuid=hit.payload.get("user_uuid", ""),
                total_chunks=hit.payload.get("total_chunks", 1),
                full_code=hit.payload.get("full_code", ""),
                metadata={k: v for k, v in hit.payload.items()
                         if k not in ["problem_id", "code", "chunk_idx", "is_passed", "user_uuid", "total_chunks", "full_code"]}
            )
            for hit in results
        ]
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Lấy thống kê về collection"""
        try:
            info = self.client.get_collection(self.COLLECTION_SUBMISSIONS)

            # Extract attributes safely because different qdrant-client versions
            # expose different fields on CollectionInfo.
            points_count = getattr(info, "points_count", None)
            vectors_count = getattr(info, "vectors_count", None)

            # Some client versions may return nested structures or different names
            if vectors_count is None:
                # Try common alternatives
                vectors_count = getattr(info, "vectors", None)
                if isinstance(vectors_count, dict) and "vectors_count" in vectors_count:
                    vectors_count = vectors_count.get("vectors_count")

            # Fallback: if vectors_count still None, assume one vector per point
            if vectors_count is None and points_count is not None:
                vectors_count = points_count

            status = None
            status_attr = getattr(info, "status", None)
            if status_attr is not None:
                status = getattr(status_attr, "value", status_attr)

            return {
                self.COLLECTION_SUBMISSIONS: {
                    "points_count": points_count,
                    "vectors_count": vectors_count,
                    "status": status or "unknown"
                }
            }
        except Exception as e:
            return {self.COLLECTION_SUBMISSIONS: {"error": str(e)}}
    
    def delete_by_problem(self, problem_id: str):
        """
        Xóa tất cả dữ liệu của một problem_id.
        
        Args:
            problem_id: ID bài toán cần xóa
        """
        try:
            self.client.delete(
                collection_name=self.COLLECTION_SUBMISSIONS,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="problem_id",
                                match=models.MatchValue(value=problem_id)
                            )
                        ]
                    )
                )
            )
            logger.info(f"Deleted all data for problem {problem_id}")
        except Exception as e:
            logger.error(f"Error deleting data: {e}")


# Singleton instance
_qdrant_tutor: Optional[QdrantTutor] = None


def get_qdrant_tutor() -> QdrantTutor:
    """Get singleton instance of QdrantTutor"""
    global _qdrant_tutor
    if _qdrant_tutor is None:
        _qdrant_tutor = QdrantTutor()
    return _qdrant_tutor

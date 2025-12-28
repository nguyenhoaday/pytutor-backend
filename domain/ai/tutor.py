"""
Hybrid Tutor - Hệ thống Gia sư AI sử dụng Qdrant RAG + AST Analysis.
Kết hợp truy xuất code mẫu và phương pháp Socratic để hướng dẫn sinh viên.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import os
import time
import json

from .qdrant_rag import get_qdrant_tutor
from .analyzer import get_hybrid_analyzer, HybridAnalysisResult
from infra.utils.normalize_code import normalize_code
from infra.utils.llm_utils import get_groq_client
import re

logger = logging.getLogger(__name__)


@dataclass
class TutorFeedback:
    """Kết quả phản hồi từ gia sư AI"""
    # Kết quả phân tích
    syntax_valid: bool
    error_type: str
    error_message: str
    error_line: Optional[int] = None
    
    # Kết quả phân tích Hybrid
    code_structure: Dict[str, Any] = None
    
    # Kết quả truy xuất từ Qdrant
    reference_code: Optional[str] = None
    reference_similarity: float = 0.0
    
    # Gợi ý (Socratic method)
    hint: str = ""
    hint_level: int = 1
    
    # Câu hỏi theo dõi
    follow_up_question: str = ""
    concepts_to_review: List[str] = None
    
    # Độ tin cậy và metadata
    confidence: float = 0.5
    strategy: str = "socratic"
    
    def __post_init__(self):
        if self.concepts_to_review is None:
            self.concepts_to_review = []
        if self.code_structure is None:
            self.code_structure = {}


class HybridTutor:
    """
    Gia sư AI kết hợp RAG (Qdrant) và phương pháp Socratic.
    
    Features:
    1. Truy xuất code mẫu tương tự từ Qdrant
    2. Phân tích AST để hiểu cấu trúc code
    3. Phân tích trong Sandbox để bắt lỗi runtime
    4. Sinh gợi ý theo phương pháp Socratic (đặt câu hỏi dẫn dắt)
    5. Hỗ trợ cả tiếng Việt và tiếng Anh
    """
    
    def __init__(self):
        self.qdrant = get_qdrant_tutor()
        self.analyzer = get_hybrid_analyzer()
        self._llm_client = None
    
    def _get_llm_client(self):
        """Lazy load Groq client"""
        if self._llm_client is None:
            try:
                self._llm_client = get_groq_client()
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
                self._llm_client = None
        return self._llm_client
    
    def generate_feedback(
        self,
        student_code: str,
        problem_id: str,
        problem_description: str = "",
        hint_level: int = 1,
        previous_hints: Optional[List[str]] = None,
        language: str = "vi",
        use_llm: bool = True,
        run_sandbox: bool = False
    ) -> TutorFeedback:
        """
        Sinh phản hồi gia sư kết hợp RAG và Socratic method.
        
        Args:
            student_code: Code của sinh viên
            problem_id: ID bài toán
            problem_description: Mô tả bài toán
            hint_level: Mức độ gợi ý (1-5, 1=mơ hồ, 5=gần đáp án)
            previous_hints: Các gợi ý đã đưa trước đó
            language: Ngôn ngữ output (vi/en)
            use_llm: Có sử dụng LLM không
            run_sandbox: Có chạy sandbox không
        """
        previous_hints = previous_hints or []

        try:
            # 1. Phân tích (phân tích AST và chạy sandbox)
            analysis = self.analyzer.analyze_hybrid(student_code, run_sandbox=run_sandbox)

            # 2. Qdrant retrieval (top_k configurable)
            top_k = 1
            retrieved = self.qdrant.semantic_search(query=student_code, top_k=top_k, problem_id=problem_id)

            ref_code = retrieved[0].full_code if retrieved else None
            ref_similarity = retrieved[0].similarity if retrieved else 0.0

            # 3. Nếu sử dụng LLM, gọi với JSON payload
            if use_llm:
                client = self._get_llm_client()
                if not client:
                    hint_text = self._generate_template_hint(analysis, hint_level, language)
                    follow_up = self._generate_follow_up(analysis, language)
                    confidence = self._calculate_confidence(analysis, ref_similarity, use_llm)
                    return TutorFeedback(
                        syntax_valid=analysis.ast_analysis.valid_syntax,
                        error_type=analysis.error_type,
                        error_message=analysis.error_message,
                        error_line=analysis.error_line,
                        code_structure=self.analyzer.get_code_structure_summary(student_code),
                        reference_code=ref_code,
                        reference_similarity=ref_similarity,
                        hint=hint_text,
                        hint_level=hint_level,
                        follow_up_question=follow_up,
                        concepts_to_review=analysis.concepts_involved,
                        confidence=confidence,
                        strategy="template"
                    )

                # Build JSON user payload theo spec
                user_payload = {
                    "student_code": normalize_code(student_code),
                    "problem_statement": problem_description or "",
                    "reference_code": ref_code,
                    "reference_similarity": ref_similarity,
                    "error_type": analysis.error_type,
                    "error_message": analysis.error_message,
                    "concepts": analysis.concepts_involved,
                    "hint_level": hint_level,
                    "previous_hints": previous_hints,
                    "constraints": "Do not give full solution code. Provide one next-step action the student should try."
                }

                # Tạo system prompt theo ngôn ngữ
                if language == "vi":
                    system_prompt = (
                        "Trả lời bằng tiếng Việt.\n"
                        "Bạn là một Gia sư Python thông minh, sử dụng phương pháp Socratic kết hợp với code tham khảo từ hệ thống.\n\n"
                        "QUAN TRỌNG:\n"
                        "- KHÔNG cho đáp án trực tiếp hay viết code hoàn chỉnh thay sinh viên\n"
                        "- SỬ DỤNG reference_code (code tham khảo đúng) để so sánh với code sinh viên và tìm điểm khác biệt\n"
                        "- Đặt câu hỏi dẫn dắt để sinh viên TỰ TÌM RA bước giải tiếp theo\n"
                        "- So sánh cấu trúc, logic, cách tiếp cận giữa code sinh viên và code tham khảo\n\n"
                        "Điều chỉnh mức độ gợi ý theo hint_level:\n"
                        "- Level 1-2: Hỏi về concept chung, rất mơ hồ, không nhắc đến code tham khảo\n"
                        "- Level 3-4: Gợi ý vị trí lỗi bằng cách so sánh với code tham khảo, hỏi về điều kiện cụ thể\n"
                        "- Level 5: Chỉ ra điểm khác biệt cụ thể với code tham khảo nhưng vẫn để sinh viên hoàn thành\n\n"
                        "Trả về JSON hợp lệ: {\"hint\": \"...\", \"next_step\": \"...\"}. KHÔNG thêm text ngoài JSON."
                    )
                else:
                    system_prompt = (
                        "Respond in English.\n"
                        "You are an intelligent Python Tutor using the Socratic method combined with reference code from the system.\n\n"
                        "IMPORTANT:\n"
                        "- DO NOT give direct answers or write complete code for the student\n"
                        "- USE reference_code (correct reference code) to compare with student code and find differences\n"
                        "- Ask guiding questions to help students DISCOVER the solution themselves\n"
                        "- Compare structure, logic, and approach between student code and reference code\n\n"
                        "Adjust hint specificity based on hint_level:\n"
                        "- Level 1-2: Ask about general concepts, very vague, don't mention reference code\n"
                        "- Level 3-4: Hint at error location by comparing with reference code, ask about specific conditions\n"
                        "- Level 5: Point out specific differences with reference code but let student complete it\n\n"
                        "Return valid JSON: {\"hint\": \"...\", \"next_step\": \"...\"}. DO NOT include extra text outside JSON."
                    )

                try:
                    response = client.chat.completions.create(
                        model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
                        ],
                        max_tokens=1024,
                        temperature=0.0
                    )

                    response_text = response.choices[0].message.content.strip()

                    # Parse JSON response
                    try:
                        parsed = json.loads(response_text)
                        hint_text = parsed.get("hint", "").strip()
                        next_step = parsed.get("next_step", "").strip()
                    except json.JSONDecodeError:
                        # Nếu parse trực tiếp lỗi, thử trích xuất JSON từ text
                        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                        if json_match:
                            try:
                                parsed = json.loads(json_match.group())
                                hint_text = parsed.get("hint", "").strip()
                                next_step = parsed.get("next_step", "").strip()
                            except json.JSONDecodeError:
                                hint_text = response_text.strip()
                                next_step = ""
                        else:
                            hint_text = response_text.strip()
                            next_step = ""

                    if not hint_text:
                        hint_text = self._generate_template_hint(analysis, hint_level, language)

                    if not hint_text or not hint_text.strip():
                        hint_text = self._generate_template_hint(analysis, hint_level, language)

                except Exception as e:
                    logger.error(f"LLM call failed: {e}")
                    hint_text = self._generate_template_hint(analysis, hint_level, language)
                    next_step = self._generate_follow_up(analysis, language)

                # Tính toán độ tin cậy và trả về
                confidence = self._calculate_confidence(analysis, ref_similarity, use_llm)

                return TutorFeedback(
                    syntax_valid=analysis.ast_analysis.valid_syntax,
                    error_type=analysis.error_type,
                    error_message=analysis.error_message,
                    error_line=analysis.error_line,
                    code_structure=self.analyzer.get_code_structure_summary(student_code),
                    reference_code=ref_code if ref_code else None,
                    reference_similarity=ref_similarity,
                    hint=hint_text,
                    hint_level=hint_level,
                    follow_up_question=next_step,
                    concepts_to_review=analysis.concepts_involved,
                    confidence=confidence,
                    strategy="qdrant_llm"
                )
            
            # Nếu không sử dụng LLM, sử dụng template hints
            hint_text = self._generate_template_hint(analysis, hint_level, language)
            follow_up = self._generate_follow_up(analysis, language)
            confidence = self._calculate_confidence(analysis, ref_similarity, use_llm)
            
            return TutorFeedback(
                syntax_valid=analysis.ast_analysis.valid_syntax,
                error_type=analysis.error_type,
                error_message=analysis.error_message,
                error_line=analysis.error_line,
                code_structure=self.analyzer.get_code_structure_summary(student_code),
                reference_code=ref_code,
                reference_similarity=ref_similarity,
                hint=hint_text,
                hint_level=hint_level,
                follow_up_question=follow_up,
                concepts_to_review=analysis.concepts_involved,
                confidence=confidence,
                strategy="template"
            )

        except Exception as e:
            logger.exception("Error generating feedback")
            return self._generate_fallback_feedback(hint_level, language)
    
    def _build_socratic_prompt(
        self,
        student_code: str,
        problem_description: str,
        analysis: HybridAnalysisResult,
        reference_code: Optional[str],
        hint_level: int,
        previous_hints: List[str],
        language: str
    ) -> str:
        """
        Xây dựng prompt Socratic cho LLM.
        Phương pháp: Đặt câu hỏi để sinh viên tự tìm ra lỗi.
        """
        if language == "vi":
            system_instruction = """Bạn là một Gia sư Python thông minh, sử dụng phương pháp Socratic.
QUAN TRỌNG: 
- KHÔNG cho đáp án trực tiếp
- KHÔNG viết code hoàn chỉnh thay sinh viên
- Đặt câu hỏi dẫn dắt để sinh viên TỰ TÌM RA bước giải tiếp theo
- Điều chỉnh mức độ gợi ý theo hint_level (1=rất mơ hồ, 5=gần đáp án)"""
            
            ref_section = ""
            if reference_code:
                ref_section = f"\n\nCode tham khảo đúng (KHÔNG cho sinh viên thấy):\n```python\n{reference_code}\n```"
            
            prev_hints_text = ""
            if previous_hints:
                prev_hints_text = f"\n\nGợi ý đã đưa trước đó:\n" + "\n".join(f"- {h}" for h in previous_hints[-3:])
            
            error_info = ""
            if analysis.error_type != "none":
                error_info = f"\n\nLoại lỗi phát hiện: {analysis.error_type}\nChi tiết: {analysis.error_message}"
            
            prompt = f"""{system_instruction}

Bài toán: {problem_description or 'Giải bài tập Python'}

Code sinh viên:
```python
{student_code}
```{error_info}{ref_section}{prev_hints_text}

Mức độ gợi ý: {hint_level}/5
Các concept liên quan: {', '.join(analysis.concepts_involved) if analysis.concepts_involved else 'chưa xác định'}

Hãy đưa ra một câu hỏi hoặc gợi ý theo phương pháp Socratic phù hợp với mức độ {hint_level}/5.
Nếu level 1-2: Hỏi về concept chung
Nếu level 3-4: Gợi ý vị trí lỗi, hỏi về điều kiện cụ thể
Nếu level 5: Gợi ý gần đáp án nhưng vẫn để sinh viên hoàn thành"""
            
        else:  # English
            system_instruction = """You are an intelligent Python Tutor using the Socratic method.
IMPORTANT:
- DO NOT give direct answers
- DO NOT write complete code for the student
- Ask guiding questions to help students DISCOVER the solution themselves
- Adjust hint specificity based on hint_level (1=very vague, 5=almost answer)"""
            
            ref_section = ""
            if reference_code:
                ref_section = f"\n\nReference solution (DO NOT show to student):\n```python\n{reference_code}\n```"
            
            prev_hints_text = ""
            if previous_hints:
                prev_hints_text = f"\n\nPrevious hints given:\n" + "\n".join(f"- {h}" for h in previous_hints[-3:])
            
            error_info = ""
            if analysis.error_type != "none":
                error_info = f"\n\nDetected error type: {analysis.error_type}\nDetails: {analysis.error_message}"
            
            prompt = f"""{system_instruction}

Problem: {problem_description or 'Python exercise'}

Student code:
```python
{student_code}
```{error_info}{ref_section}{prev_hints_text}

Hint level: {hint_level}/5
Related concepts: {', '.join(analysis.concepts_involved) if analysis.concepts_involved else 'undetermined'}

Provide a Socratic question or hint appropriate for level {hint_level}/5.
Level 1-2: Ask about general concepts
Level 3-4: Hint at error location, ask about specific conditions
Level 5: Give near-answer hint but let student complete it"""
        
        return prompt
    
    def _generate_from_llm(
        self,
        prompt: str,
        language: str,
        hint_level: int
    ) -> str:
        """Gọi LLM để sinh hint"""
        client = self._get_llm_client()
        
        if not client:
            return self._generate_template_hint(None, hint_level, language)
        
        try:
            # Sử dụng system message để model trả lời theo ngôn ngữ
            if language == "vi":
                sys_msg = (
                    "Trả lời bằng tiếng Việt. Bạn là một Gia sư Python theo phương pháp Socratic. KHÔNG đưa code hoàn chỉnh."
                )
            else:
                sys_msg = (
                    "Respond in English. You are a Socratic Python tutor. DO NOT provide complete code."
                )

            response = client.chat.completions.create(
                model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return self._generate_template_hint(None, hint_level, language)
    
    def _generate_template_hint(
        self,
        analysis: Optional[HybridAnalysisResult],
        hint_level: int,
        language: str
    ) -> str:
        """Sinh hint từ template khi không dùng LLM"""
        
        templates = {
            "vi": {
                "syntax": {
                    1: "Có vẻ như có lỗi cú pháp trong code của bạn. Bạn đã kiểm tra lại cách viết chưa?",
                    2: "Hãy kiểm tra lại các dấu ngoặc, dấu hai chấm và thụt lề trong code.",
                    3: "Lỗi cú pháp thường xảy ra ở dấu ngoặc hoặc thụt lề. Xem lại dòng được báo lỗi.",
                    4: "Kiểm tra dòng có lỗi: có đủ dấu ngoặc đóng không? Thụt lề có đúng không?",
                    5: "Cú pháp Python yêu cầu: dấu hai chấm sau if/for/while/def, thụt lề 4 spaces."
                },
                "logic": {
                    1: "Kết quả có vẻ chưa đúng. Bạn đã thử với các trường hợp khác nhau chưa?",
                    2: "Hãy nghĩ về logic của thuật toán. Các điều kiện đã đầy đủ chưa?",
                    3: "Kiểm tra lại các điều kiện trong vòng lặp và câu lệnh if.",
                    4: "Chú ý đến giá trị biên. Vòng lặp bắt đầu và kết thúc đúng chỗ chưa?",
                    5: "Kiểm tra range(): range(n) cho 0 đến n-1, range(1, n+1) cho 1 đến n."
                },
                "runtime": {
                    1: "Code gặp lỗi khi chạy. Bạn đã kiểm tra các biến chưa?",
                    2: "Có biến nào đang được sử dụng mà chưa được tạo không?",
                    3: "Kiểm tra tên biến: có viết đúng không? Có tạo trước khi dùng không?",
                    4: "Lỗi NameError thường do biến chưa được gán giá trị hoặc viết sai tên.",
                    5: "Thêm dòng khởi tạo biến trước khi sử dụng."
                },
                "infinite_loop": {
                    1: "Code có vẻ chạy mãi. Vòng lặp của bạn có điểm dừng không?",
                    2: "Vòng lặp while cần có điều kiện dừng. Bạn đã kiểm tra chưa?",
                    3: "Biến điều kiện có được thay đổi trong vòng lặp không?",
                    4: "Với while True, cần có break hoặc return để thoát.",
                    5: "Thêm điều kiện if và break để thoát vòng lặp khi cần."
                },
                "none": {
                    1: "Code của bạn có vẻ OK. Hãy thử với nhiều test case hơn.",
                    2: "Kiểm tra lại logic với các trường hợp đặc biệt.",
                    3: "Xem xét các edge cases: list rỗng, số âm, số 0...",
                    4: "So sánh output với kết quả mong đợi.",
                    5: "Nếu bạn vẫn cần giúp, hãy mô tả vấn đề cụ thể hơn."
                }
            },
            "en": {
                "syntax": {
                    1: "There seems to be a syntax error. Have you checked your code structure?",
                    2: "Check your brackets, colons, and indentation.",
                    3: "Syntax errors often occur with brackets or indentation. Review the error line.",
                    4: "Check the error line: are brackets balanced? Is indentation correct?",
                    5: "Python syntax requires: colon after if/for/while/def, 4-space indentation."
                },
                "logic": {
                    1: "The result doesn't seem right. Have you tried different test cases?",
                    2: "Think about the algorithm logic. Are all conditions covered?",
                    3: "Review conditions in your loops and if statements.",
                    4: "Pay attention to boundary values. Does the loop start/end correctly?",
                    5: "Check range(): range(n) gives 0 to n-1, range(1, n+1) gives 1 to n."
                },
                "runtime": {
                    1: "The code encounters an error when running. Have you checked your variables?",
                    2: "Is there a variable being used before it's defined?",
                    3: "Check variable names: spelled correctly? Defined before use?",
                    4: "NameError usually means a variable wasn't assigned or is misspelled.",
                    5: "Add a line to initialize the variable before using it."
                },
                "infinite_loop": {
                    1: "The code seems to run forever. Does your loop have a stopping point?",
                    2: "While loops need a stopping condition. Have you checked?",
                    3: "Is the condition variable being modified inside the loop?",
                    4: "With while True, you need break or return to exit.",
                    5: "Add an if condition with break to exit the loop when needed."
                },
                "none": {
                    1: "Your code looks OK. Try testing with more test cases.",
                    2: "Review the logic with special cases.",
                    3: "Consider edge cases: empty list, negative numbers, zero...",
                    4: "Compare output with expected results.",
                    5: "If you still need help, describe your issue more specifically."
                }
            }
        }
        
        lang_templates = templates.get(language, templates["vi"])
        
        error_type = "none"
        if analysis:
            error_type = analysis.error_type or "none"
        
        type_templates = lang_templates.get(error_type, lang_templates["none"])
        
        return type_templates.get(hint_level, type_templates[1])
    
    def _generate_follow_up(
        self,
        analysis: HybridAnalysisResult,
        language: str
    ) -> str:
        """Tạo câu hỏi follow-up"""
        if language == "vi":
            if analysis.error_type == "syntax":
                return "Bạn có thể chỉ ra dòng nào có lỗi không?"
            elif analysis.error_type == "logic":
                return "Kết quả bạn mong đợi là gì? Kết quả thực tế là gì?"
            elif analysis.error_type == "runtime":
                return "Lỗi xảy ra ở dòng nào? Thông báo lỗi nói gì?"
            elif analysis.error_type == "infinite_loop":
                return "Điều kiện dừng của vòng lặp là gì?"
            else:
                return "Bạn có câu hỏi gì thêm không?"
        else:
            if analysis.error_type == "syntax":
                return "Can you identify which line has the error?"
            elif analysis.error_type == "logic":
                return "What output do you expect? What do you actually get?"
            elif analysis.error_type == "runtime":
                return "Which line causes the error? What does the error message say?"
            elif analysis.error_type == "infinite_loop":
                return "What is the stopping condition for your loop?"
            else:
                return "Do you have any other questions?"
    
    def _calculate_confidence(
        self,
        analysis: HybridAnalysisResult,
        ref_similarity: float,
        use_llm: bool
    ) -> float:
        """Tính điểm confidence cho feedback"""
        confidence = 0.5
        
        # Có reference code tương đồng cao
        if ref_similarity > 0.8:
            confidence += 0.3
        elif ref_similarity > 0.6:
            confidence += 0.2
        elif ref_similarity > 0.4:
            confidence += 0.1
        
        # Phát hiện được lỗi cụ thể
        if analysis.error_type != "none":
            confidence += 0.1
        
        # Sử dụng LLM
        if use_llm:
            confidence += 0.1
        
        return min(0.95, confidence)
    
    def _generate_fallback_feedback(
        self,
        hint_level: int,
        language: str
    ) -> TutorFeedback:
        """Tạo feedback fallback khi có lỗi"""
        if language == "vi":
            hint = "Xin lỗi, mình gặp chút vấn đề. Bạn có thể thử lại không? 🙏"
        else:
            hint = "Sorry, I encountered an issue. Could you try again? 🙏"
        
        return TutorFeedback(
            syntax_valid=True,
            error_type="unknown",
            error_message="",
            hint=hint,
            hint_level=hint_level,
            confidence=0.3,
            strategy="fallback"
        )
    
    def add_to_knowledge_base(
        self,
        problem_id: str,
        code: str,
        user_id: Optional[int] = None,
        is_passed: bool = True
    ):
        """
        Thêm code vào knowledge base.
        """
        if is_passed and user_id:
            self.qdrant.add_successful_submission(problem_id, code, user_id)
        else:
            self.qdrant.add_knowledge(problem_id, code)


# Singleton instance
_hybrid_tutor: Optional[HybridTutor] = None


def get_hybrid_tutor() -> HybridTutor:
    """Lấy instance của HybridTutor"""
    global _hybrid_tutor
    if _hybrid_tutor is None:
        _hybrid_tutor = HybridTutor()
    return _hybrid_tutor

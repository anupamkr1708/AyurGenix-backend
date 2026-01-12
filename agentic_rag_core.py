# -*- coding: utf-8 -*-
"""
Agentic RAG Core with Groq API (Cloud-based LLM)
Optimized for deployment on Render free tier
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import json
import re
import os

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Groq API
from groq import Groq

ENABLE_LANGSMITH = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

if ENABLE_LANGSMITH:
    from langsmith import traceable
else:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator



@dataclass
class Message:
    role: str
    content: str
    timestamp: str
    metadata: Dict = field(default_factory=dict)


class ConversationalMemory:
    """Enhanced memory with relevance-based retrieval"""

    def __init__(self, max_messages: int = 20):
        self.messages = deque(maxlen=max_messages)
        self.profile = {"conditions": [], "preferences": {}, "doshas_mentioned": []}

    def add(self, role: str, content: str, metadata: Dict = None):
        self.messages.append(
            Message(
                role=role,
                content=content,
                timestamp=datetime.now().isoformat(),
                metadata=metadata or {},
            )
        )

        if role == "user":
            self._extract_profile(content)

    def _extract_profile(self, text: str):
        """Extract health signals from user messages"""
        conditions = [
            "vata", "pitta", "kapha", "diabetes", "anxiety", "arthritis",
            "indigestion", "constipation", "stress", "sleep", "pain",
            "fever", "cold", "headache",
        ]
        text_lower = text.lower()

        for condition in conditions:
            if condition in text_lower:
                if condition not in self.profile["conditions"]:
                    self.profile["conditions"].append(condition)

                if condition in ["vata", "pitta", "kapha"]:
                    if condition not in self.profile["doshas_mentioned"]:
                        self.profile["doshas_mentioned"].append(condition)

    def get_history(self, n: int = 5) -> str:
        """Get recent messages"""
        recent = list(self.messages)[-n:]
        return "\n".join([f"{m.role.upper()}: {m.content}" for m in recent])

    def get_relevant(self, query: str, n: int = 3) -> str:
        """Get most relevant past messages"""
        if not self.messages:
            return ""

        query_words = set(query.lower().split())
        scored = []

        for msg in self.messages:
            msg_words = set(msg.content.lower().split())
            overlap = len(query_words & msg_words)
            if overlap >= 2:
                scored.append((overlap, msg))

        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored:
            return ""

        context = ["RELEVANT HISTORY:"]
        for _, msg in scored[:n]:
            context.append(f"{msg.role.upper()}: {msg.content[:150]}...")

        return "\n".join(context)

    def get_user_context(self) -> str:
        """Get user profile summary"""
        if not self.profile["conditions"]:
            return ""

        context_parts = []

        if self.profile["doshas_mentioned"]:
            context_parts.append(
                f"Doshas: {', '.join(self.profile['doshas_mentioned'])}"
            )

        other_conditions = [
            c for c in self.profile["conditions"] if c not in ["vata", "pitta", "kapha"]
        ]
        if other_conditions:
            context_parts.append(f"Concerns: {', '.join(other_conditions[:5])}")

        return "USER PROFILE: " + " | ".join(context_parts)

    def clear(self):
        self.messages.clear()
        self.profile = {"conditions": [], "preferences": {}, "doshas_mentioned": []}


class QueryProcessor:
    """Enhanced query processing with intent classification"""

    def __init__(self):
        self.intents = {
            "treatment": ["treat", "cure", "remedy", "medicine", "therapy"],
            "prevention": ["prevent", "avoid", "stop", "reduce risk"],
            "diet": ["eat", "avoid", "diet", "food", "meal", "nutrition"],
            "lifestyle": ["lifestyle", "routine", "habits", "daily", "practice"],
            "symptoms": ["symptom", "sign", "feel", "experience"],
            "causes": ["cause", "why", "reason", "origin"],
            "diagnosis": ["diagnose", "identify", "assess"],
            "comparison": ["difference", "compare", "versus", "vs"],
            "definition": ["what is", "meaning", "define", "explain"],
            "recommendation": ["suggest", "recommend", "advise", "should"],
        }

        self.ayur_terms = [
            "vata", "pitta", "kapha", "dosha", "agni", "ama", "ojas",
            "prana", "tejas", "dhatu", "srotas", "mala", "triphala",
            "ashwagandha", "ginger", "turmeric", "brahmi", "panchakarma",
            "abhyanga", "shirodhara",
        ]

    def classify_intent(self, query: str) -> str:
        """Classify query intent"""
        query_lower = query.lower()

        scores = {}
        for intent, keywords in self.intents.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[intent] = score

        return max(scores, key=scores.get) if scores else "general"

    def extract_entities(self, query: str) -> List[str]:
        """Extract Ayurvedic entities"""
        query_lower = query.lower()
        return [term for term in self.ayur_terms if term in query_lower]

    def expand_query(self, query: str, intent: str, entities: List[str]) -> List[str]:
        """Rule-based query expansion"""
        expanded = [query]

        if intent == "treatment":
            expanded.append(f"Ayurvedic treatment for {query}")
            if entities:
                expanded.append(f"{entities[0]} remedies protocol")

        elif intent == "diet":
            expanded.append(f"Ayurvedic diet recommendations {query}")
            expanded.append(f"Foods to eat and avoid {query}")

        elif intent == "lifestyle":
            expanded.append(f"Daily routine Dinacharya {query}")
            if entities:
                expanded.append(f"{entities[0]} lifestyle modifications")

        elif intent == "symptoms":
            expanded.append(f"Signs and symptoms {query}")
            if entities:
                expanded.append(f"{entities[0]} imbalance indicators")

        for entity in entities[:2]:
            expanded.append(f"{entity} classical Ayurvedic texts")

        return list(dict.fromkeys(expanded))[:4]

    def process(self, query: str, history: str) -> Dict:
        """Complete query processing"""
        intent = self.classify_intent(query)
        entities = self.extract_entities(query)
        expanded = self.expand_query(query, intent, entities)
        is_followup = self._detect_followup(query, history)

        return {
            "original": query,
            "intent": intent,
            "entities": entities,
            "expanded_queries": expanded,
            "is_followup": is_followup,
        }

    def _detect_followup(self, query: str, history: str) -> bool:
        """Detect if query is a follow-up"""
        followup_words = ["also", "what about", "and", "more", "else", "another"]
        pronouns = ["this", "that", "it", "them"]

        query_lower = query.lower()
        has_reference = any(p in query_lower for p in pronouns)
        has_followup = any(f in query_lower for f in followup_words)
        is_short = len(query.split()) < 5

        return (has_reference or has_followup or is_short) and len(history) > 0


class AdvancedReranker:
    """Multi-factor reranking: semantic + intent + diversity"""

    def __init__(self, embedder):
        self.embedder = embedder

    @traceable(name="rerank")
    def rerank(
        self, query: str, contexts: List[Dict], intent: str, top_k: int = 5
    ) -> List[Dict]:
        """Advanced multi-factor reranking"""
        if not contexts:
            return []

        max_score = max(c["score"] for c in contexts)
        for ctx in contexts:
            ctx["semantic_score"] = ctx["score"] / max_score if max_score > 0 else 0

        intent_keywords = {
            "treatment": ["remedy", "treatment", "medicine", "cure", "therapy"],
            "diet": ["food", "diet", "eat", "avoid", "nutrition", "meal"],
            "lifestyle": ["routine", "lifestyle", "daily", "practice", "habit"],
            "symptoms": ["symptom", "sign", "manifest", "indicate"],
            "causes": ["cause", "reason", "origin", "due to"],
        }

        keywords = intent_keywords.get(intent, [])

        for ctx in contexts:
            text_lower = ctx["text"].lower()
            matches = sum(1 for kw in keywords if kw in text_lower)
            ctx["intent_score"] = min(matches / max(len(keywords), 1), 1.0)

        source_counts = {}
        for ctx in contexts:
            source = ctx["source"]
            source_counts[source] = source_counts.get(source, 0) + 1

        for ctx in contexts:
            ctx["diversity_score"] = 1.0 / source_counts[ctx["source"]]

        for ctx in contexts:
            text_len = len(ctx["text"])
            if 200 <= text_len <= 1000:
                ctx["quality_score"] = 1.0
            elif text_len < 200:
                ctx["quality_score"] = text_len / 200.0
            else:
                ctx["quality_score"] = 0.8

        for ctx in contexts:
            ctx["final_score"] = (
                ctx["semantic_score"] * 0.50
                + ctx["intent_score"] * 0.25
                + ctx["diversity_score"] * 0.15
                + ctx["quality_score"] * 0.10
            )

        contexts.sort(key=lambda x: x["final_score"], reverse=True)
        return contexts[:top_k]


class RobustAyurvedicRAG:
    """
    Production-ready Agentic RAG with Groq API
    """

    def __init__(self, pinecone_key: str, index_name: str, groq_api_key: str):
        
        self.enable_eval = os.getenv("ENABLE_RAG_EVAL", "false").lower() == "true"

        print("\n" + "=" * 70)
        print("🚀 INITIALIZING ROBUST AGENTIC RAG (GROQ API)")
        print("=" * 70 + "\n")

        self.memories: Dict[str, ConversationalMemory] = {}

        print("📦 Loading embedder...")
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("   ✅ Embedder ready")

        print("🔗 Connecting to Pinecone...")
        self.pc = Pinecone(api_key=pinecone_key)
        self.index = self.pc.Index(index_name)
        stats = self.index.describe_index_stats()
        print(f"   ✅ Connected: {stats.total_vector_count} vectors")

        print("🤖 Connecting to Groq API...")
        self.groq_client = Groq(api_key=groq_api_key)
        self.model_name = "llama-3.3-70b-versatile"  # Fast and powerful
        print(f"   ✅ Groq ready: {self.model_name}")

        self.query_processor = QueryProcessor()
        self.reranker = AdvancedReranker(self.embedder)

        print("\n" + "=" * 70)
        print("✅ SYSTEM READY!")
        print("=" * 70 + "\n")

    def _get_memory(self, session_id: str) -> ConversationalMemory:
        """Returns session-specific conversational memory"""
        if session_id not in self.memories:
            self.memories[session_id] = ConversationalMemory(max_messages=20)
        return self.memories[session_id]

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.4,
        system_prompt: str = None,
    ) -> str:
        """Generate response using Groq API"""
        
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })

        try:
            completion = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9,
                stream=False,
            )
            
            return completion.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ Groq API error: {e}")
            return "I apologize, but I encountered an error generating a response. Please try again."

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 512,
        temperature: float = 0.4,
    ):
        """Stream response tokens using Groq API"""
        
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })

        try:
            stream = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9,
                stream=True,
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            print(f"❌ Groq streaming error: {e}")
            yield "Error generating response."

    @traceable(name="retrieve")
    def retrieve(self, queries: List[str], top_k: int = 5) -> List[Dict]:
        """Multi-query retrieval"""
        all_results = {}

        for query in queries:
            query_vec = self.embedder.encode(query).tolist()
            results = self.index.query(
                vector=query_vec, top_k=top_k, include_metadata=True
            )

            for match in results.matches:
                doc_id = match.id
                if (
                    doc_id not in all_results
                    or match.score > all_results[doc_id]["score"]
                ):
                    all_results[doc_id] = {
                        "id": doc_id,
                        "text": match.metadata.get("text", ""),
                        "source": match.metadata.get("source", "unknown"),
                        "page": match.metadata.get("page", -1),
                        "score": float(match.score),
                    }

        results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        return results[: top_k * 2]

    @traceable(name="generate_answer")
    def _generate_answer(
        self, query: str, query_info: Dict, contexts: List[Dict], history: str
    ) -> str:
        """Generate answer with intent-based system prompts"""

        context_text = "\n\n".join(
            [
                f"[Source {i+1}: {c['source']}, Page {c['page']}]\n{c['text']}"
                for i, c in enumerate(contexts)
            ]
        )

        intent_prompts = {
            "treatment": "You are an Ayurvedic physician. Provide treatment recommendations with herbs, dosages, diet, and lifestyle changes. Always cite sources.",
            "diet": "You are an Ayurvedic nutritionist. Provide detailed dietary guidance with specific foods to eat and avoid. Be practical.",
            "lifestyle": "You are an Ayurvedic lifestyle consultant. Provide daily routines (Dinacharya) with specific timings and practices.",
            "symptoms": "You are an Ayurvedic diagnostician. Describe symptoms with dosha correlation and classical signs.",
            "general": "You are an Ayurvedic scholar. Provide clear, evidence-based answers from classical texts.",
        }

        intent = query_info["intent"]
        system_prompt = intent_prompts.get(intent, intent_prompts["general"])

        prompt = f"""Answer based on these Ayurvedic texts:

CONTEXT:
{context_text}

"""

        if history:
            prompt += f"{history}\n\n"

        prompt += f"""QUESTION: {query}

Provide a comprehensive answer with:
- Clear explanation
- Practical application
- Source citations [Source N]
- Sanskrit terms with English translations

ANSWER:"""

        answer = self.generate(
            prompt, max_tokens=600, temperature=0.4, system_prompt=system_prompt
        )

        if not answer or len(answer.strip()) < 40:
            answer = (
                "According to Ayurvedic texts, this concept relates to the balance of doshas "
                "and the principles outlined in classical references."
            )

        return answer.strip()

    def _calculate_confidence(
        self, query_info: Dict, contexts: List[Dict], answer: str
    ) -> float:
        """Multi-factor confidence scoring"""
        if not contexts:
            return 0.0

        avg_retrieval = np.mean([c.get("final_score", c["score"]) for c in contexts])
        source_score = min(len(contexts) / 5.0, 1.0)
        length_score = min(len(answer) / 400.0, 1.0)
        citation_count = answer.count("[Source")
        citation_score = min(citation_count / 3.0, 1.0)
        entities_in_answer = sum(
            1 for entity in query_info["entities"] if entity.lower() in answer.lower()
        )
        entity_score = (
            entities_in_answer / len(query_info["entities"])
            if query_info["entities"]
            else 0.5
        )

        confidence = (
            avg_retrieval * 0.40
            + source_score * 0.20
            + length_score * 0.15
            + citation_score * 0.15
            + entity_score * 0.10
        )

        return min(confidence, 1.0)

    @traceable(name="ayurgenix_rag_chat")
    def chat(
        self,
        user_query: str,
        session_id: str,
        use_memory: bool = True,
        verbose: bool = False,
    ) -> Dict:
        """Main agentic chat with 6-step reasoning"""
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"💬 USER [{session_id}]: {user_query}")
            print(f"{'='*70}\n")

        reasoning_steps = []
        start_time = datetime.now()

        memory = self._get_memory(session_id)

        if verbose:
            print("📝 Step 1: Retrieving conversation context...")

        conversation_context = ""
        if use_memory and memory.messages:
            conversation_context = memory.get_relevant(user_query, n=3)
            user_profile = memory.get_user_context()
            if user_profile:
                conversation_context = f"{user_profile}\n\n{conversation_context}"

        reasoning_steps.append(f"Memory chars: {len(conversation_context)}")

        if verbose:
            print("🔍 Step 2: Processing & expanding query...")

        query_info = self.query_processor.process(user_query, conversation_context)
        reasoning_steps.append(
            f"Intent={query_info['intent']} | "
            f"Entities={query_info['entities']} | "
            f"Expanded={len(query_info['expanded_queries'])}"
        )

        if verbose:
            print("🔎 Step 3: Multi-query retrieval...")

        raw_contexts = self.retrieve(query_info["expanded_queries"], top_k=5)
        reasoning_steps.append(f"Retrieved={len(raw_contexts)}")

        if verbose:
            print("⚡ Step 4: Advanced reranking...")

        reranked_contexts = self.reranker.rerank(
            user_query, raw_contexts, intent=query_info["intent"], top_k=5
        )
        reasoning_steps.append(f"Reranked={len(reranked_contexts)}")

        if verbose:
            print("✨ Step 5: Generating answer...")

        answer = self._generate_answer(
            user_query, query_info, reranked_contexts, conversation_context
        )
        reasoning_steps.append(f"Answer chars={len(answer)}")

        confidence = self._calculate_confidence(query_info, reranked_contexts, answer)
        reasoning_steps.append(f"Confidence={confidence:.2f}")

        if use_memory:
            memory.add("user", user_query)
            memory.add(
                "assistant",
                answer,
                {"confidence": confidence, "intent": query_info["intent"]},
            )

        elapsed = (datetime.now() - start_time).total_seconds()

        if verbose:
            print(f"\n{'='*70}")
            print(f"✅ Done in {elapsed:.2f}s | Confidence {confidence:.1%}")
            print(f"{'='*70}\n")

        return {
            "query": user_query,
            "answer": answer,
            "sources": [
                {
                    "source": c["source"],
                    "page": c["page"],
                    "score": c.get("final_score", c["score"]),
                    "text_preview": c["text"][:200],
                }
                for c in reranked_contexts
            ],
            "confidence": confidence,
            "reasoning": reasoning_steps,
            "intent": query_info["intent"],
            "entities": query_info["entities"],
            "response_time_seconds": elapsed,
        }

    def reset_conversation(self, session_id: Optional[str] = None):
        """Clear conversation memory"""
        if session_id:
            self.memories.pop(session_id, None)
        else:
            self.memories.clear()
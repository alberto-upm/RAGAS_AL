from langchain_ollama import OllamaLLM
from langchain_core.documents import Document as LCDocument
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses import Document as HaystackDocument 
import numpy as np
import os
import json
import typing as t
import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Sequence
from tqdm import tqdm
import random
from pydantic import BaseModel
from difflib import get_close_matches
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define basic structures needed
class NodeType(str, Enum):
    DOCUMENT = "document"
    SUMMARY = "summary"
    ENTITY = "entity"
    KEYPHRASE = "keyphrase"
    THEME = "theme"
    TOPIC = "topic"
    
class Node:
    """Node in the knowledge graph."""
    
    def __init__(self, type: NodeType, properties: Dict[str, Any] = None):
        self.type = type
        self.properties = properties or {}
        self.id = properties.get("id", str(id(self)))
        
class KnowledgeGraph:
    """Simple knowledge graph implementation."""
    
    def __init__(self, nodes: List[Node] = None):
        self.nodes = nodes or []
        
    def add_node(self, node: Node):
        self.nodes.append(node)
        
    def get_nodes_by_type(self, node_type: NodeType) -> List[Node]:
        return [node for node in self.nodes if node.type == node_type]

class Persona(BaseModel):
    """Persona model representing a user."""
    name: str
    role_description: str

class PersonaList(BaseModel):
    """List of personas with lookup capabilities."""
    personas: List[Persona]

    def __getitem__(self, key: str) -> Persona:
        # Normalize the search name
        key_normalized = key.split(" (")[0].strip()
        
        # Try exact match first
        for persona in self.personas:
            if persona.name == key_normalized:
                return persona

        # Try approximate match
        names = [persona.name for persona in self.personas]
        matches = get_close_matches(key_normalized, names, n=1, cutoff=0.6)

        if matches:
            for persona in self.personas:
                if persona.name == matches[0]:
                    return persona
        raise KeyError(f"No persona found with name '{key_normalized}'")

class TestsetSample(BaseModel):
    """Sample in the evaluation dataset."""
    query: str
    ground_truth: str
    persona: Persona
    synthesizer_name: str = ""
    source_documents: List[Dict[str, Any]] = []
    
class Testset(BaseModel):
    """Evaluation dataset."""
    samples: List[TestsetSample]

class QueryType(str, Enum):
    SINGLE_HOP = "single_hop_specific_query_synthesizer"
    MULTI_HOP = "multi_hop_specific_query_synthesizer"

@dataclass
class OllamaTestsetGenerator:
    """
    Testset generator using OllamaLLM for all generation tasks.
    
    Attributes:
    -----------
    llm: OllamaLLM
        The language model to use for generation tasks
    knowledge_graph: KnowledgeGraph
        Knowledge graph containing document nodes and metadata
    persona_list: Optional[List[Persona]]
        List of personas to use in the testset generation
    """
    
    llm: OllamaLLM
    knowledge_graph: KnowledgeGraph = field(default_factory=KnowledgeGraph)
    persona_list: Optional[List[Persona]] = None
    
    async def extract_summary(self, text: str) -> str:
        """
        Extract a summary from the text using the LLM.
        
        Args:
            text: Text to summarize
            
        Returns:
            Summary of the text
        """
        prompt = f"""Summarize the given text in less than 10 sentences. 
        
        Text: {text}
        
        Return ONLY a JSON object with a field 'text'.
        Example: {{"text": "This document discusses the key aspects of machine learning..."}}
        """
        
        try:
            response = self.llm.invoke(prompt)
            # Parse JSON response
            result = json.loads(response)
            return result.get("text", "")
        except Exception as e:
            logger.error(f"Error extracting summary: {e}")
            return ""
    
    async def extract_keyphrases(self, text: str, max_num: int = 5) -> List[str]:
        """
        Extract key phrases from the text.
        
        Args:
            text: Text to extract from
            max_num: Maximum number of keyphrases
            
        Returns:
            List of keyphrases
        """
        prompt = f"""Extract top {max_num} keyphrases from the given text.
        
        Text: {text}
        
        Return ONLY a JSON object with a field 'keyphrases' as a list of strings.
        Example: {{"keyphrases": ["Artificial intelligence", "automating tasks", "healthcare"]}}
        """
        
        try:
            response = self.llm.invoke(prompt)
            # Parse JSON response
            result = json.loads(response)
            return result.get("keyphrases", [])
        except Exception as e:
            logger.error(f"Error extracting keyphrases: {e}")
            return []
    
    async def extract_entities(self, text: str, max_num: int = 10) -> List[str]:
        """
        Extract entities from the text.
        
        Args:
            text: Text to extract from
            max_num: Maximum number of entities
            
        Returns:
            List of entities
        """
        prompt = f"""Extract the named entities from the given text, limiting to the top {max_num} entities.
        
        Text: {text}
        
        Return ONLY a JSON object with a field 'entities' as a list of strings.
        Example: {{"entities": ["Elon Musk", "Tesla", "SpaceX"]}}
        """
        
        try:
            response = self.llm.invoke(prompt)
            # Parse JSON response
            result = json.loads(response)
            return result.get("entities", [])
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []

    async def extract_themes(self, text: str, max_num: int = 5) -> List[str]:
        """
        Extract themes from the text.
        
        Args:
            text: Text to extract from
            max_num: Maximum number of themes
            
        Returns:
            List of themes
        """
        prompt = f"""Extract the main themes or topics from the given text, limiting to the top {max_num}.
        
        Text: {text}
        
        Return ONLY a JSON object with a field 'themes' as a list of strings.
        Example: {{"themes": ["Artificial Intelligence Ethics", "Technology Regulation", "Future of Work"]}}
        """
        
        try:
            response = self.llm.invoke(prompt)
            # Parse JSON response
            result = json.loads(response)
            return result.get("themes", [])
        except Exception as e:
            logger.error(f"Error extracting themes: {e}")
            return []
    
    async def generate_persona_from_summary(self, summary: str) -> Persona:
        """
        Generate a persona from a document summary.
        
        Args:
            summary: Document summary
            
        Returns:
            Generated persona
        """
        prompt = f"""Using the provided summary, generate a single persona who would likely interact with or benefit from the content.
        Provide a short, unique name (only one or two words) and a concise role description.
        
        Summary: {summary}
        
        Return ONLY a JSON object with fields 'name' and 'role_description'.
        Example: {{"name": "Digital Marketing Specialist", "role_description": "Focuses on engaging audiences and growing the brand online."}}
        """
        
        try:
            response = self.llm.invoke(prompt)
            # Parse JSON response
            result = json.loads(response)
            return Persona(
                name=result.get("name", "Unknown Persona"),
                role_description=result.get("role_description", "No description provided")
            )
        except Exception as e:
            logger.error(f"Error generating persona: {e}")
            return Persona(name="Generic User", role_description="A general user of the system.")
    
    async def generate_embeddings(self, text: str) -> List[float]:
        """
        Generate simple embeddings for the text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        prompt = f"""Generate a simplified embedding vector for the following text. 
        The vector should capture the semantic meaning of the text.
        
        Text: {text}
        
        Return ONLY a JSON object with a field 'embedding' containing a list of 10 float values.
        Example: {{"embedding": [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9, -1.0]}}
        """
        
        try:
            response = self.llm.invoke(prompt)
            # Parse JSON response
            result = json.loads(response)
            return result.get("embedding", [0.0] * 10)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return default embedding
            return [0.0] * 10
    
    async def generate_query(self, summary: str, persona: Persona, query_type: QueryType) -> str:
        """
        Generate a query based on a summary and persona.
        
        Args:
            summary: Document summary
            persona: User persona
            query_type: Type of query to generate (single-hop or multi-hop)
            
        Returns:
            Generated query
        """
        if query_type == QueryType.SINGLE_HOP:
            prompt = f"""Generate a simple single-hop question that {persona.name} ({persona.role_description}) might ask about the following content:
            
            Content: {summary}
            
            The question should be specific and ask about a single concept, term, entity or fact that appears in the content.
            
            Return ONLY a JSON object with a field 'query'.
            Example: {{"query": "What is a microservices architecture?"}}
            """
        else:  # MULTI_HOP
            prompt = f"""Generate a complex multi-hop question that {persona.name} ({persona.role_description}) might ask about the following content:
            
            Content: {summary}
            
            The question should require connecting multiple pieces of information from the content, making comparisons, or require deeper analysis.
            For example, it should ask about relationships between concepts, comparisons between methods/approaches,
            or require synthesis of information from different parts of the content.
            
            Return ONLY a JSON object with a field 'query'.
            Example: {{"query": "How do the benefits of microservices architecture compare to traditional monolithic systems in terms of scalability and maintenance costs?"}}
            """
        
        try:
            response = self.llm.invoke(prompt)
            # Parse JSON response
            result = json.loads(response)
            return result.get("query", "")
        except Exception as e:
            logger.error(f"Error generating query: {e}")
            return ""
    
    async def generate_ground_truth(self, query: str, summary: str) -> str:
        """
        Generate ground truth answer for a query.
        
        Args:
            query: User query
            summary: Document summary
            
        Returns:
            Ground truth answer
        """
        prompt = f"""Based on the following content, provide a comprehensive and accurate answer to the query.
        
        Content: {summary}
        
        Query: {query}
        
        Return ONLY a JSON object with a field 'answer' containing a comprehensive response.
        Example: {{"answer": "Microservices architecture offers several benefits for enterprise applications including..."}}
        """
        
        try:
            response = self.llm.invoke(prompt)
            # Parse JSON response
            result = json.loads(response)
            return result.get("answer", "")
        except Exception as e:
            logger.error(f"Error generating ground truth: {e}")
            return ""
    
    async def generate_personas_from_kg(self, num_personas: int = 3) -> List[Persona]:
        """
        Generate personas from the knowledge graph.
        
        Args:
            num_personas: Number of personas to generate
            
        Returns:
            List of personas
        """
        # Get summary nodes from the knowledge graph
        summary_nodes = self.knowledge_graph.get_nodes_by_type(NodeType.SUMMARY)
        if not summary_nodes:
            # Fallback to document nodes if no summaries
            document_nodes = self.knowledge_graph.get_nodes_by_type(NodeType.DOCUMENT)
            summaries = []
            
            # Generate summaries for document nodes
            for node in tqdm(document_nodes[:min(10, len(document_nodes))], desc="Generating summaries"):
                content = node.properties.get("page_content", "")
                if content:
                    summary = await self.extract_summary(content)
                    if summary:
                        summaries.append(summary)
        else:
            # Use existing summaries
            summaries = [node.properties.get("summary", "") for node in summary_nodes]
            summaries = [s for s in summaries if s]
        
        # Ensure we have summaries
        if not summaries:
            logger.warning("No summaries available for persona generation.")
            return [
                Persona(name="Generic User", role_description="A general user of the system."),
                Persona(name="Domain Expert", role_description="A specialist with deep knowledge in the field."),
                Persona(name="Beginner", role_description="A newcomer to the subject with basic understanding.")
            ]
        
        # Generate embeddings for clustering
        embeddings = []
        for summary in tqdm(summaries, desc="Generating embeddings"):
            embedding = await self.generate_embeddings(summary)
            embeddings.append(embedding)
        
        # Cluster summaries
        embeddings_array = np.array(embeddings)
        groups = []
        visited = set()
        threshold = 0.75
        
        for i in range(len(summaries)):
            if i in visited:
                continue
            
            group = [i]
            visited.add(i)
            
            for j in range(i + 1, len(summaries)):
                if j in visited:
                    continue
                    
                # Calculate similarity
                vec1 = embeddings_array[i]
                vec2 = embeddings_array[j]
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                
                if similarity > threshold:
                    group.append(j)
                    visited.add(j)
                    
            groups.append(group)
        
        # Get representative summaries
        representative_summaries = []
        for group in groups:
            representative_idx = max(group, key=lambda idx: len(summaries[idx]))
            representative_summaries.append(summaries[representative_idx])
        
        # Handle case with fewer groups than requested personas
        if len(representative_summaries) < num_personas:
            additional_needed = num_personas - len(representative_summaries)
            additional_summaries = random.choices(representative_summaries, k=additional_needed)
            representative_summaries.extend(additional_summaries)
        
        # Limit to requested number
        representative_summaries = representative_summaries[:num_personas]
        
        # Generate personas
        personas = []
        for summary in tqdm(representative_summaries, desc="Generating personas"):
            persona = await self.generate_persona_from_summary(summary)
            personas.append(persona)
        
        return personas
    
    async def build_knowledge_graph(self, documents: List[Dict[str, Any]]) -> KnowledgeGraph:
        """
        Build a knowledge graph from documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Knowledge graph
        """
        kg = KnowledgeGraph()
        
        # Add document nodes
        for i, doc in enumerate(tqdm(documents, desc="Building knowledge graph")):
            content = doc["content"]
            metadata = doc["metadata"]
            
            # Create document node
            doc_node = Node(
                type=NodeType.DOCUMENT,
                properties={
                    "id": f"doc_{i}",
                    "page_content": content,
                    "document_metadata": metadata
                }
            )
            kg.add_node(doc_node)
            
            # Extract and add summary
            summary = await self.extract_summary(content)
            if summary:
                summary_node = Node(
                    type=NodeType.SUMMARY,
                    properties={
                        "id": f"summary_{i}",
                        "summary": summary,
                        "doc_id": f"doc_{i}"
                    }
                )
                kg.add_node(summary_node)
            
            # Extract and add keyphrases
            keyphrases = await self.extract_keyphrases(content)
            for j, keyphrase in enumerate(keyphrases):
                keyphrase_node = Node(
                    type=NodeType.KEYPHRASE,
                    properties={
                        "id": f"keyphrase_{i}_{j}",
                        "keyphrase": keyphrase,
                        "doc_id": f"doc_{i}"
                    }
                )
                kg.add_node(keyphrase_node)
            
            # Extract and add entities
            entities = await self.extract_entities(content)
            for j, entity in enumerate(entities):
                entity_node = Node(
                    type=NodeType.ENTITY,
                    properties={
                        "id": f"entity_{i}_{j}",
                        "entity": entity,
                        "doc_id": f"doc_{i}"
                    }
                )
                kg.add_node(entity_node)
            
            # Extract and add themes
            themes = await self.extract_themes(content)
            for j, theme in enumerate(themes):
                theme_node = Node(
                    type=NodeType.THEME,
                    properties={
                        "id": f"theme_{i}_{j}",
                        "theme": theme,
                        "doc_id": f"doc_{i}"
                    }
                )
                kg.add_node(theme_node)
        
        return kg
    
    async def generate_sample(self, summary_node: Node, persona: Persona, query_type: QueryType) -> TestsetSample:
        """
        Generate a testset sample from a summary and persona.
        
        Args:
            summary_node: Summary node
            persona: User persona
            query_type: Type of query to generate
            
        Returns:
            Testset sample
        """
        summary = summary_node.properties.get("summary", "")
        doc_id = summary_node.properties.get("doc_id")
        
        # Generate query based on query type
        query = await self.generate_query(summary, persona, query_type)
        
        # Generate ground truth
        ground_truth = await self.generate_ground_truth(query, summary)
        
        # Find source document
        source_docs = []
        if doc_id:
            for node in self.knowledge_graph.nodes:
                if node.type == NodeType.DOCUMENT and node.properties.get("id") == doc_id:
                    source_docs.append({
                        "content": node.properties.get("page_content", ""),
                        "metadata": node.properties.get("document_metadata", {})
                    })
        
        return TestsetSample(
            query=query,
            ground_truth=ground_truth,
            persona=persona,
            synthesizer_name=query_type.value,  # Use the query type value as synthesizer name
            source_documents=source_docs
        )
    
    async def _process_and_generate_samples(
        self, 
        testset_size: int, 
        persona_list: List[Persona],
        single_hop_ratio: float = 0.5
    ) -> List[TestsetSample]:
        """
        Process the knowledge graph and generate samples.
        
        Args:
            testset_size: Number of samples to generate
            persona_list: List of personas to use
            single_hop_ratio: Ratio of single-hop to multi-hop questions
            
        Returns:
            List of testset samples
        """
        # Get summary nodes
        summary_nodes = self.knowledge_graph.get_nodes_by_type(NodeType.SUMMARY)
        if not summary_nodes:
            logger.error("No summary nodes found in the knowledge graph.")
            return []
        
        # Calculate number of single-hop and multi-hop questions
        num_single_hop = int(testset_size * single_hop_ratio)
        num_multi_hop = testset_size - num_single_hop
        
        # Randomly select summary nodes if we have more than we need
        if len(summary_nodes) > testset_size:
            selected_summaries = random.sample(summary_nodes, testset_size)
        else:
            # Otherwise use all and possibly duplicate some
            selected_summaries = random.choices(summary_nodes, k=testset_size)
        
        # Generate samples
        samples = []
        
        # Generate single-hop samples
        for i in range(num_single_hop):
            # Select a persona (cycling through the list)
            persona_idx = i % len(persona_list)
            persona = persona_list[persona_idx]
            
            # Generate sample
            sample = await self.generate_sample(
                selected_summaries[i], 
                persona, 
                QueryType.SINGLE_HOP
            )
            samples.append(sample)
            
        # Generate multi-hop samples
        for i in range(num_multi_hop):
            # Select a persona (cycling through the list)
            persona_idx = (i + num_single_hop) % len(persona_list)
            persona = persona_list[persona_idx]
            
            # Generate sample
            sample = await self.generate_sample(
                selected_summaries[i + num_single_hop], 
                persona, 
                QueryType.MULTI_HOP
            )
            samples.append(sample)
        
        # Shuffle samples
        random.shuffle(samples)
        
        return samples
    
    async def generate_with_langchain_docs_async(
        self,
        documents: Sequence[LCDocument],
        testset_size: int,
        num_personas: int = 3,
        chunk_size: int = 1024,
        chunk_overlap: int = 100,
        single_hop_ratio: float = 0.5
    ) -> Testset:
        """
        Generate a testset from LangChain documents.
        
        Args:
            documents: LangChain documents
            testset_size: Number of samples to generate
            num_personas: Number of personas to generate
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            single_hop_ratio: Ratio of single-hop to multi-hop questions
            
        Returns:
            Generated testset
        """
        # 1. Split documents into chunks
        logger.info(f"Splitting documents into chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")
        
        splitter = DocumentSplitter(
            split_by="word",
            split_length=chunk_size,
            split_overlap=chunk_overlap,
        )
        
        chunks = []
        for doc in tqdm(documents, desc="Splitting documents"):
            # Convert to Haystack document format
            haystack_doc = HaystackDocument(
                content=doc.page_content,
                meta=doc.metadata
            )
            split_results = splitter.run(documents=[haystack_doc])
            chunks.extend(split_results["documents"])
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # 2. Convert chunks to format for processing
        doc_list = []
        for chunk in chunks:
            try:
                metadata = chunk.metadata if hasattr(chunk, 'metadata') else {}
                
                # Create document dictionary
                doc_list.append({
                    "content": chunk.content,
                    "metadata": metadata
                })
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
                # Add document with empty metadata to avoid failures
                doc_list.append({
                    "content": chunk.content,
                    "metadata": {}
                })
        
        # 3. Build knowledge graph
        logger.info("Building knowledge graph")
        self.knowledge_graph = await self.build_knowledge_graph(doc_list)
        
        # 4. Generate personas if not provided
        if self.persona_list is None or len(self.persona_list) < num_personas:
            logger.info(f"Generating {num_personas} personas")
            self.persona_list = await self.generate_personas_from_kg(num_personas)
        
        # 5. Generate samples with specified ratio of single-hop to multi-hop questions
        logger.info(f"Generating {testset_size} samples (single-hop ratio: {single_hop_ratio})")
        samples = await self._process_and_generate_samples(
            testset_size=testset_size,
            persona_list=self.persona_list[:num_personas],
            single_hop_ratio=single_hop_ratio
        )
        
        # 6. Create testset
        return Testset(samples=samples)
    
    def generate_with_langchain_docs(
        self,
        documents: Sequence[LCDocument],
        testset_size: int,
        num_personas: int = 3,
        chunk_size: int = 1024,
        chunk_overlap: int = 100,
        single_hop_ratio: float = 0.5
    ) -> Testset:
        """
        Generate a testset from LangChain documents (synchronous wrapper).
        
        Args:
            documents: LangChain documents
            testset_size: Number of samples to generate
            num_personas: Number of personas to generate
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            single_hop_ratio: Ratio of single-hop to multi-hop questions
            
        Returns:
            Generated testset
        """
        return asyncio.run(self.generate_with_langchain_docs_async(
            documents=documents,
            testset_size=testset_size,
            num_personas=num_personas,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            single_hop_ratio=single_hop_ratio
        ))
    
    async def generate_async(
        self,
        testset_size: int,
        num_personas: int = 3,
        single_hop_ratio: float = 0.5
    ) -> Testset:
        """
        Generate a testset from the knowledge graph.
        
        Args:
            testset_size: Number of samples to generate
            num_personas: Number of personas to generate
            single_hop_ratio: Ratio of single-hop to multi-hop questions
            
        Returns:
            Generated testset
        """
        # Check if knowledge graph has nodes
        if not self.knowledge_graph.nodes:
            raise ValueError("Knowledge graph is empty. Add documents first.")
        
        # Generate personas if not provided
        if self.persona_list is None or len(self.persona_list) < num_personas:
            logger.info(f"Generating {num_personas} personas")
            self.persona_list = await self.generate_personas_from_kg(num_personas)
        
        # Generate samples
        logger.info(f"Generating {testset_size} samples (single-hop ratio: {single_hop_ratio})")
        samples = await self._process_and_generate_samples(
            testset_size=testset_size,
            persona_list=self.persona_list[:num_personas],
            single_hop_ratio=single_hop_ratio
        )
        
        # Create testset
        return Testset(samples=samples)
    
    def generate(
        self,
        testset_size: int,
        num_personas: int = 3,
        single_hop_ratio: float = 0.5
    ) -> Testset:
        """
        Generate a testset from the knowledge graph (synchronous wrapper).
        
        Args:
            testset_size: Number of samples to generate
            num_personas: Number of personas to generate
            single_hop_ratio: Ratio of single-hop to multi-hop questions
            
        Returns:
            Generated testset
        """
        return asyncio.run(self.generate_async(
            testset_size=testset_size,
            num_personas=num_personas,
            single_hop_ratio=single_hop_ratio
        ))

def save_testset_as_csv(testset: Testset, output_file: str):
    """
    Save the testset as a CSV file with specified columns.
    
    Args:
        testset: Generated testset
        output_file: Path to save the CSV file
    """
    import csv
    
    # Change file extension to .csv if not already
    if not output_file.endswith('.csv'):
        output_file = output_file.replace('.json', '.csv')
        if not output_file.endswith('.csv'):
            output_file += '.csv'
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['user_input', 'reference_context', 'reference', 'synthesizer_name'])
        
        # Write data rows
        for sample in testset.samples:
            user_input = sample.query
            reference = sample.ground_truth
            
            # Get reference context from source documents
            reference_context = ""
            if sample.source_documents:
                for doc in sample.source_documents:
                    reference_context += doc.get("content", "") + " "
            
            # Include the synthesizer_name in the output
            synthesizer_name = sample.synthesizer_name
            
            writer.writerow([user_input, reference_context.strip(), reference, synthesizer_name])
    
    logger.info(f"Saved testset to CSV: {output_file}")

# Usage example
def create_testset_from_documents(
    documents_path: str,
    output_file: str = "testset.csv",
    testset_size: int = 10,
    num_personas: int = 3,
    chunk_size: int = 1024,
    chunk_overlap: int = 100,
    single_hop_ratio: float = 0.5,  # Added parameter for ratio
    ollama_model: str = "llama3.1:8b"
):
    """
    Create a testset from documents using OllamaLLM.
    
    Args:
        documents_path: Path to the documents
        output_file: Path to save the testset
        testset_size: Number of samples to generate
        num_personas: Number of personas to generate
        chunk_size: Size of document chunks
        chunk_overlap: Overlap between chunks
        single_hop_ratio: Ratio of single-hop to multi-hop questions (default: 0.5)
        ollama_model: Ollama model to use
    """
    from langchain_community.document_loaders import DirectoryLoader
    
    # Configure Ollama LLM
    llm = OllamaLLM(
        model=ollama_model,
        temperature=0.5,
        max_tokens=2600,
        format="json"
    )
    
    # Create testset generator
    generator = OllamaTestsetGenerator(llm=llm)
    
    # Load documents
    logger.info(f"Loading documents from {documents_path}")
    loader = DirectoryLoader(documents_path)
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents")
    
    # Generate testset
    testset = generator.generate_with_langchain_docs(
        documents=documents,
        testset_size=testset_size,
        num_personas=num_personas,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        single_hop_ratio=single_hop_ratio
    )
    
    # Save testset as CSV
    if output_file:
        # If CSV file already exists, remove it
        if os.path.exists(output_file):
            os.remove(output_file)
        save_testset_as_csv(testset, output_file)
        
        # Optional: also save as JSON for backup
        json_output = output_file.replace('.csv', '.json')
        if os.path.exists(json_output):
            os.remove(json_output)
        with open(json_output, "w") as f:
            f.write(testset.model_dump_json(indent=2))
        logger.info(f"Also saved testset to JSON: {json_output}")
    
    return testset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate testset from documents with single-hop and multi-hop questions")
    parser.add_argument("--documents_path", type=str, default="/Users/albertog.garcia/Documents/UPM/TFG/Documentos", help="Path to documents")
    parser.add_argument("--output_file", type=str, default="/Users/albertog.garcia/Documents/UPM/TFG/RAGAS_AL/output/documents_ollama_generation_testset_mixed.csv", help="Output file")
    parser.add_argument("--testset_size", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--num_personas", type=int, default=3, help="Number of personas to generate")
    parser.add_argument("--chunk_size", type=int, default=1024, help="Size of document chunks")
    parser.add_argument("--chunk_overlap", type=int, default=100, help="Overlap between chunks")
    parser.add_argument("--single_hop_ratio", type=float, default=0.5, help="Ratio of single-hop to multi-hop questions (0.0-1.0)")
    parser.add_argument("--ollama_model", type=str, default="llama3.1:8b", help="Ollama model to use")
    
    args = parser.parse_args()
    
    # Validate single_hop_ratio
    if not 0.0 <= args.single_hop_ratio <= 1.0:
        parser.error("single_hop_ratio must be between 0.0 and 1.0")
    
    testset = create_testset_from_documents(
        documents_path=args.documents_path,
        output_file=args.output_file,
        testset_size=args.testset_size,
        num_personas=args.num_personas,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        single_hop_ratio=args.single_hop_ratio,
        ollama_model=args.ollama_model
    )
    
    # Print summary
    print(f"\nGenerated testset with {len(testset.samples)} samples")
    print(f"Number of personas: {len(set([s.persona.name for s in testset.samples]))}")
    
    # Count each query type
    single_hop_count = sum(1 for s in testset.samples if s.synthesizer_name == QueryType.SINGLE_HOP.value)
    multi_hop_count = sum(1 for s in testset.samples if s.synthesizer_name == QueryType.MULTI_HOP.value)
    print(f"Single-hop questions: {single_hop_count} ({single_hop_count/len(testset.samples):.1%})")
    print(f"Multi-hop questions: {multi_hop_count} ({multi_hop_count/len(testset.samples):.1%})")
    
    # Print sample queries of each type
    print("\nSample single-hop queries:")
    single_hop_samples = [s for s in testset.samples if s.synthesizer_name == QueryType.SINGLE_HOP.value]
    for i, sample in enumerate(single_hop_samples[:2], 1):
        print(f"{i}. {sample.query} (by {sample.persona.name})")
    
    print("\nSample multi-hop queries:")
    multi_hop_samples = [s for s in testset.samples if s.synthesizer_name == QueryType.MULTI_HOP.value]
    for i, sample in enumerate(multi_hop_samples[:2], 1):
        print(f"{i}. {sample.query} (by {sample.persona.name})")
    
    # Optional: Add RAGAS integration if needed
    try:
        # Configure the token of authentication for Ragas
        os.environ["RAGAS_APP_TOKEN"] = "apt.44d7-55ca6e4e5ccd-d4a1-906b-0f447f55-80b6b"
        
        # Upload the dataset (make sure you have a valid account in Ragas)
        testset.upload()
        print("\nUploaded testset to RAGAS successfully")
    except Exception as e:
        print(f"\nFailed to upload to RAGAS: {e}")
        print("Please check your RAGAS credentials or configuration")

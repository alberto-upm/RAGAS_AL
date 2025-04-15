import typing as t
from dataclasses import dataclass

from pydantic import BaseModel

from ragas.prompt import PydanticPrompt, StringIO
from ragas.testset.graph import Node
from ragas.testset.transforms.base import LLMBasedExtractor
import json

class TextWithExtractionLimit(BaseModel):
    text: str
    max_num: int = 10


class SummaryExtractorPrompt(PydanticPrompt[StringIO, StringIO]):
    instruction: str = (
        "Summarize the given text in less than 10 sentences. "
        "Return ONLY a JSON object with a field 'text'. "
        'Example: {"text": "RAGAS is a framework for evaluating Retrieval Augmented Generation (RAG) systems..."}'
    )
    input_model: t.Type[StringIO] = StringIO
    output_model: t.Type[StringIO] = StringIO
    examples: t.List[t.Tuple[StringIO, StringIO]] = [
        (
            StringIO(
                text="Artificial intelligence\n\nArtificial intelligence is transforming various industries by automating tasks that previously required human intelligence. From healthcare to finance, AI is being used to analyze vast amounts of data quickly and accurately. This technology is also driving innovations in areas like self-driving cars and personalized recommendations."
            ),
            StringIO(
                text="AI is revolutionizing industries by automating tasks, analyzing data, and driving innovations like self-driving cars and personalized recommendations."
            ),
        )
    ]


class Keyphrases(BaseModel):
    keyphrases: t.List[str]


class KeyphrasesExtractorPrompt(PydanticPrompt[TextWithExtractionLimit, Keyphrases]):
    instruction: str = (
        "Extract top max_num keyphrases from the given text. "
        "Return ONLY a JSON object with a field 'keyphrases' as a list of strings. "
        'Example: {"keyphrases": ["Artificial intelligence", "automating tasks", "healthcare"]}'
    )
    input_model: t.Type[TextWithExtractionLimit] = TextWithExtractionLimit
    output_model: t.Type[Keyphrases] = Keyphrases
    examples: t.List[t.Tuple[TextWithExtractionLimit, Keyphrases]] = [
        (
            TextWithExtractionLimit(
                text="Artificial intelligence\n\nArtificial intelligence is transforming various industries by automating tasks that previously required human intelligence. From healthcare to finance, AI is being used to analyze vast amounts of data quickly and accurately. This technology is also driving innovations in areas like self-driving cars and personalized recommendations.",
                max_num=5,
            ),
            Keyphrases(
                keyphrases=[
                    "Artificial intelligence",
                    "automating tasks",
                    "healthcare",
                    "self-driving cars",
                    "personalized recommendations",
                ]
            ),
        )
    ]


class TitleExtractorPrompt(PydanticPrompt[StringIO, StringIO]):
    instruction: str = (
        "Extract the title of the given document. "
        "Return ONLY a JSON object with a field 'text'. "
        'Example: {"text": "Deep Learning for Natural Language Processing"}'
    )

    input_model: t.Type[StringIO] = StringIO
    output_model: t.Type[StringIO] = StringIO
    examples: t.List[t.Tuple[StringIO, StringIO]] = [
        (
            StringIO(
                text="Deep Learning for Natural Language Processing\n\nAbstract\n\nDeep learning has revolutionized the field of natural language processing (NLP). This paper explores various deep learning models and their applications in NLP tasks such as language translation, sentiment analysis, and text generation. We discuss the advantages and limitations of different models, and provide a comprehensive overview of the current state of the art in NLP."
            ),
            StringIO(text="Deep Learning for Natural Language Processing"),
        )
    ]


class Headlines(BaseModel):
    headlines: t.List[str]


class HeadlinesExtractorPrompt(PydanticPrompt[TextWithExtractionLimit, Headlines]):
    instruction: str = (
        "Extract the most important max_num headlines from the given text that can be used to split the text into independent sections. "
        "Focus on Level 2 and Level 3 headings. Return ONLY a JSON object with a field 'headlines'. "
        'Example: {"headlines": ["Introduction", "Main Concepts", "Detailed Analysis"]}'
    )


    input_model: t.Type[TextWithExtractionLimit] = TextWithExtractionLimit
    output_model: t.Type[Headlines] = Headlines
    examples: t.List[t.Tuple[TextWithExtractionLimit, Headlines]] = [
        (
            TextWithExtractionLimit(
                text="""\
                Introduction
                Overview of the topic...

                Main Concepts
                Explanation of core ideas...

                Detailed Analysis
                Techniques and methods for analysis...

                Subsection: Specialized Techniques
                Further details on specialized techniques...

                Future Directions
                Insights into upcoming trends...

                Subsection: Next Steps in Research
                Discussion of new areas of study...

                Conclusion
                Final remarks and summary.
                """,
                max_num=6,
            ),
            Headlines(
                headlines=[
                    "Introduction",
                    "Main Concepts",
                    "Detailed Analysis",
                    "Subsection: Specialized Techniques",
                    "Future Directions",
                    "Conclusion",
                ],
            ),
        ),
    ]


class NEROutput(BaseModel):
    entities: t.List[str]


class NERPrompt(PydanticPrompt[TextWithExtractionLimit, NEROutput]):
    instruction: str = (
        "Extract the named entities from the given text, limiting the output to the top entities. "
        "Ensure the number of entities does not exceed the specified maximum. "
        "Return ONLY a JSON object with a field 'entities'. "
        'Example: {"entities": ["Elon Musk", "Tesla", "Europe"]}'
    )

    input_model: t.Type[TextWithExtractionLimit] = TextWithExtractionLimit
    output_model: t.Type[NEROutput] = NEROutput
    examples: t.List[t.Tuple[TextWithExtractionLimit, NEROutput]] = [
        (
            TextWithExtractionLimit(
                text="""Elon Musk, the CEO of Tesla and SpaceX, announced plans to expand operations to new locations in Europe and Asia.
                This expansion is expected to create thousands of jobs, particularly in cities like Berlin and Shanghai.""",
                max_num=10,
            ),
            NEROutput(
                entities=[
                    "Elon Musk",
                    "Tesla",
                    "SpaceX",
                    "Europe",
                    "Asia",
                    "Berlin",
                    "Shanghai",
                ]
            ),
        ),
    ]


@dataclass
class SummaryExtractor(LLMBasedExtractor):
    """
    Extracts a summary from the given text.

    Attributes
    ----------
    property_name : str
        The name of the property to extract.
    prompt : SummaryExtractorPrompt
        The prompt used for extraction.
    """

    property_name: str = "summary"
    prompt: SummaryExtractorPrompt = SummaryExtractorPrompt()

    async def extract(self, node: Node) -> t.Tuple[str, t.Any]:
        node_text = node.get_property("page_content")
        if node_text is None:
            return self.property_name, None
        chunks = self.split_text_by_token_limit(node_text, self.max_token_limit)
        try:
            result = await self.prompt.generate(self.llm, data=StringIO(text=chunks[0]))
            print("\n[DEBUG] RAW OUTPUT from SummaryExtractor:")
            print(result)
            return self.property_name, result.text
        except Exception as e:
            print("[ERROR] SummaryExtractor failed. Using raw_output fallback.")
            try:
                parsed = json.loads(result.raw_output)
                return self.property_name, parsed.get("text", None)
            except Exception as e2:
                print("[FATAL] Could not parse raw_output in SummaryExtractor:", e2)
                return self.property_name, None


@dataclass
class KeyphrasesExtractor(LLMBasedExtractor):
    """
    Extracts top keyphrases from the given text.

    Attributes
    ----------
    property_name : str
        The name of the property to extract.
    prompt : KeyphrasesExtractorPrompt
        The prompt used for extraction.
    """

    property_name: str = "keyphrases"
    prompt: KeyphrasesExtractorPrompt = KeyphrasesExtractorPrompt()
    max_num: int = 5

    async def extract(self, node: Node) -> t.Tuple[str, t.Any]:
        node_text = node.get_property("page_content")
        if node_text is None:
            return self.property_name, None
        chunks = self.split_text_by_token_limit(node_text, self.max_token_limit)
        keyphrases = []
        for chunk in chunks:
            try:
                result = await self.prompt.generate(
                    self.llm, data=TextWithExtractionLimit(text=chunk, max_num=self.max_num)
                )
                print("\n[DEBUG] RAW OUTPUT from KeyphrasesExtractor:")
                print(result)
                keyphrases.extend(result.keyphrases)
            except Exception as e:
                print("[ERROR] KeyphrasesExtractor failed. Using raw_output fallback.")
                try:
                    parsed = json.loads(result.raw_output)
                    keyphrases.extend(parsed.get("keyphrases", []))
                except Exception as e2:
                    print("[FATAL] Could not parse raw_output in KeyphrasesExtractor:", e2)
        return self.property_name, keyphrases


@dataclass
class TitleExtractor(LLMBasedExtractor):
    """
    Extracts the title from the given text.

    Attributes
    ----------
    property_name : str
        The name of the property to extract.
    prompt : TitleExtractorPrompt
        The prompt used for extraction.
    """

    property_name: str = "title"
    prompt: TitleExtractorPrompt = TitleExtractorPrompt()

    async def extract(self, node: Node) -> t.Tuple[str, t.Any]:
        node_text = node.get_property("page_content")
        if node_text is None:
            return self.property_name, None
        chunks = self.split_text_by_token_limit(node_text, self.max_token_limit)
        try:
            result = await self.prompt.generate(self.llm, data=StringIO(text=chunks[0]))
            print("\n[DEBUG] RAW OUTPUT from TitleExtractor:")
            print(result)
            return self.property_name, result.text
        except Exception as e:
            print("[ERROR] TitleExtractor failed. Using raw_output fallback.")
            try:
                parsed = json.loads(result.raw_output)
                return self.property_name, parsed.get("text", None)
            except Exception as e2:
                print("[FATAL] Could not parse raw_output in TitleExtractor:", e2)
                return self.property_name, None


@dataclass
class HeadlinesExtractor(LLMBasedExtractor):
    """
    Extracts the headlines from the given text.

    Attributes
    ----------
    property_name : str
        The name of the property to extract.
    prompt : HeadlinesExtractorPrompt
        The prompt used for extraction.
    """

    property_name: str = "headlines"
    prompt: HeadlinesExtractorPrompt = HeadlinesExtractorPrompt()
    max_num: int = 5

    async def extract(self, node: Node) -> t.Tuple[str, t.Any]:
        node_text = node.get_property("page_content")
        if node_text is None:
            return self.property_name, None
        chunks = self.split_text_by_token_limit(node_text, self.max_token_limit)
        headlines = []
        for chunk in chunks:
            try:
                result = await self.prompt.generate(
                    self.llm, data=TextWithExtractionLimit(text=chunk, max_num=self.max_num)
                )
                print("\n[DEBUG] RAW OUTPUT from HeadlinesExtractor:")
                print(result)
                headlines.extend(result.headlines)
            except Exception as e:
                print("[ERROR] HeadlinesExtractor failed. Using raw_output fallback.")
                try:
                    parsed = json.loads(result.raw_output)
                    headlines.extend(parsed.get("headlines", []))
                except Exception as e2:
                    print("[FATAL] Could not parse raw_output in HeadlinesExtractor:", e2)
        return self.property_name, headlines


@dataclass
class NERExtractor(LLMBasedExtractor):
    """
    Extracts named entities from the given text.

    Attributes
    ----------
    property_name : str
        The name of the property to extract. Defaults to "entities".
    prompt : NERPrompt
        The prompt used for extraction.
    """

    property_name: str = "entities"
    prompt: PydanticPrompt[TextWithExtractionLimit, NEROutput] = NERPrompt()
    max_num_entities: int = 10

    async def extract(self, node: Node) -> t.Tuple[str, t.List[str]]:
        node_text = node.get_property("page_content")
        if node_text is None:
            return self.property_name, []
        chunks = self.split_text_by_token_limit(node_text, self.max_token_limit)
        entities = []
        for chunk in chunks:
            try:
                result = await self.prompt.generate(
                    self.llm,
                    data=TextWithExtractionLimit(text=chunk, max_num=self.max_num_entities),
                )
                print("\n[DEBUG] RAW OUTPUT from NERExtractor:")
                print(result)
                entities.extend(result.entities)
            except Exception as e:
                print("[ERROR] NERExtractor failed. Using raw_output fallback.")
                try:
                    parsed = json.loads(result.raw_output)
                    entities.extend(parsed.get("entities", []))
                except Exception as e2:
                    print("[FATAL] Could not parse raw_output in NERExtractor:", e2)
        return self.property_name, entities


class TopicDescription(BaseModel):
    description: str


class TopicDescriptionPrompt(PydanticPrompt[StringIO, TopicDescription]):
    instruction: str = (
        "Provide a concise description of the main topic(s) discussed in the following text. "
        "Return ONLY a JSON object with a field 'description'. "
        'Example: {"description": "An introduction to quantum computing and its potential..."}'
    )

    input_model: t.Type[StringIO] = StringIO
    output_model: t.Type[TopicDescription] = TopicDescription
    examples: t.List[t.Tuple[StringIO, TopicDescription]] = [
        (
            StringIO(
                text="Quantum Computing\n\nQuantum computing leverages the principles of quantum mechanics to perform complex computations more efficiently than classical computers. It has the potential to revolutionize fields like cryptography, material science, and optimization problems by solving tasks that are currently intractable for classical systems."
            ),
            TopicDescription(
                description="An introduction to quantum computing and its potential to outperform classical computers in complex computations, impacting areas such as cryptography and material science."
            ),
        )
    ]


@dataclass
class TopicDescriptionExtractor(LLMBasedExtractor):
    """
    Extracts a concise description of the main topic(s) discussed in the given text.

    Attributes
    ----------
    property_name : str
        The name of the property to extract.
    prompt : TopicDescriptionPrompt
        The prompt used for extraction.
    """

    property_name: str = "topic_description"
    prompt: PydanticPrompt = TopicDescriptionPrompt()

    async def extract(self, node: Node) -> t.Tuple[str, t.Any]:
        node_text = node.get_property("page_content")
        if node_text is None:
            return self.property_name, None
        chunks = self.split_text_by_token_limit(node_text, self.max_token_limit)
        try:
            result = await self.prompt.generate(self.llm, data=StringIO(text=chunks[0]))
            print("\n[DEBUG] RAW OUTPUT from TopicDescriptionExtractor:")
            print(result)
            return self.property_name, result.description
        except Exception as e:
            print("[ERROR] TopicDescriptionExtractor failed. Using raw_output fallback.")
            try:
                parsed = json.loads(result.raw_output)
                return self.property_name, parsed.get("description", None)
            except Exception as e2:
                print("[FATAL] Could not parse raw_output in TopicDescriptionExtractor:", e2)
                return self.property_name, None


class ThemesAndConcepts(BaseModel):
    output: t.List[str]


class ThemesAndConceptsExtractorPrompt(
    PydanticPrompt[TextWithExtractionLimit, ThemesAndConcepts]
):
    instruction: str = (
        "Extract the main themes and concepts from the given text. "
        "Return ONLY a JSON object with a field 'output' as a list of strings. "
        'Example: {"output": ["Artificial intelligence", "Automation", "Self-driving cars"]}'
    )

    input_model: t.Type[TextWithExtractionLimit] = TextWithExtractionLimit
    output_model: t.Type[ThemesAndConcepts] = ThemesAndConcepts
    examples: t.List[t.Tuple[TextWithExtractionLimit, ThemesAndConcepts]] = [
        (
            TextWithExtractionLimit(
                text="Artificial intelligence is transforming industries by automating tasks requiring human intelligence. AI analyzes vast data quickly and accurately, driving innovations like self-driving cars and personalized recommendations.",
                max_num=10,
            ),
            ThemesAndConcepts(
                output=[
                    "Artificial intelligence",
                    "Automation",
                    "Data analysis",
                    "Innovation",
                    "Self-driving cars",
                    "Personalized recommendations",
                ]
            ),
        )
    ]


@dataclass
class ThemesExtractor(LLMBasedExtractor):
    """
    Extracts themes from the given text.

    Attributes
    ----------
    property_name : str
        The name of the property to extract. Defaults to "themes".
    prompt : ThemesExtractorPrompt
        The prompt used for extraction.
    """

    property_name: str = "themes"
    prompt: ThemesAndConceptsExtractorPrompt = ThemesAndConceptsExtractorPrompt()
    max_num_themes: int = 10

    async def extract(self, node: Node) -> t.Tuple[str, t.List[str]]:
        node_text = node.get_property("page_content")
        if node_text is None:
            return self.property_name, []
        chunks = self.split_text_by_token_limit(node_text, self.max_token_limit)
        themes = []
        for chunk in chunks:
            try:
                result = await self.prompt.generate(
                    self.llm,
                    data=TextWithExtractionLimit(text=chunk, max_num=self.max_num_themes),
                )
                print("\n[DEBUG] RAW OUTPUT from ThemesExtractor:")
                print(result)
                themes.extend(result.output)
            except Exception as e:
                print("[ERROR] ThemesExtractor failed. Using raw_output fallback.")
                try:
                    parsed = json.loads(result.raw_output)
                    themes.extend(parsed.get("output", []))
                except Exception as e2:
                    print("[FATAL] Could not parse raw_output in ThemesExtractor:", e2)
        return self.property_name, themes
    
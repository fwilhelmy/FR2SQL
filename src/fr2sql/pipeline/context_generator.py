class ContextGenerator:
    """Create prompts or context from schemas and questions for LLMs."""

    def __init__(self, linker, extractor):
        """Store helper objects for building context.

        TODO: keep references to schema linker and extractor.
        """
        self.linker = linker
        self.extractor = extractor

    def build(self, question: str):
        """Compose a textual context for the language model.

        TODO: combine schema info and question into a prompt.
        """
        return ""

class Picard:
    """Apply PICARD constraints to generated SQL sequences."""

    def __init__(self, grammar_path: str):
        """Load SQL grammar for PICARD constraints.

        TODO: parse grammar and prepare incremental decoding hooks.
        """
        self.grammar_path = grammar_path

    def constrain(self, partial_sql: str):
        """Return constrained SQL according to grammar rules."""
        # TODO: implement PICARD constraint application
        return partial_sql

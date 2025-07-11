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

    def generate_llm_context(schema: dict, user_request: str, db_type: str = "PostgreSQL") -> str:
    
        context = "Vous êtes un assistant expert en bases de données SQL.\n\n"
        context += "Voici la structure de la base de données :\n\n"

        for table in schema["tables"]:
            context += f"- Table **{table['name']}**\n"
            for col in table["columns"]:
                info = f"   - {col['name']} ({col['type']})"
                if col.get("pk"):
                    info += ", clé primaire"
                if col.get("fk"):
                    info += f", clé étrangère vers {col['fk']}"
                context += info + "\n"
            context += "\n"

        context += "Tâche :\n"
        context += f"Générez une requête SQL pour répondre à cette demande utilisateur :\n"
        context += f"**\"{user_request}\"**\n\n"
        context += f"La requête doit être compatible avec {db_type}.\n"
        context += "Expliquez la logique si besoin."

        return context
    # Exemple d'utilisation
    if __name__ == "__main__":
        example_schema = {
            "tables": [
                {
                    "name": "clients",
                    "columns": [
                        {"name": "id", "type": "INT", "pk": True},
                        {"name": "nom", "type": "VARCHAR"},
                        {"name": "ville", "type": "VARCHAR"}
                    ]
                },
                {
                    "name": "commandes",
                    "columns": [
                        {"name": "id", "type": "INT", "pk": True},
                        {"name": "client_id", "type": "INT", "fk": "clients.id"},
                        {"name": "date", "type": "DATE"},
                        {"name": "montant", "type": "DECIMAL"}
                    ]
                }
            ]
        }

        question = "Afficher le total des ventes par ville pour l'année 2023."
        print(generate_llm_context(example_schema, question))
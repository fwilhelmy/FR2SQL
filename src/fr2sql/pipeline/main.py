"""Entry point for training and evaluation pipeline."""

# TODO: wire all components together here


from fr2sql.pipeline.ContextGenerator import ContextGenerator


def main():
    """Run the full FR2SQL pipeline."""
    llm = ContextGenerator()
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
    print(llm.generate_llm_context(example_schema, question))
    pass

if __name__ == "__main__":
    main()

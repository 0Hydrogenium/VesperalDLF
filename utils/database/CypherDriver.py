import os

from neo4j import GraphDatabase


class CypherDriver:
    driver = GraphDatabase.driver(
        uri=os.getenv("NEO4J_CONNECTOR_URI"),
        auth=(
            os.getenv("NEO4J_CONNECTOR_AUTH_USER"),
            os.getenv("NEO4J_CONNECTOR_AUTH_PASSWORD")
        )
    )

    @classmethod
    def _execute_read_cypher(cls, graph, cypher, params=None):
        result = graph.run(cypher, parameters=params)
        values_list = [{k: v for k, v in record.items()} for record in result]
        return values_list

    @classmethod
    def _execute_write_cypher(cls, graph, cypher, params=None):
        graph.run(cypher, parameters=params)

    @classmethod
    def execute_read(cls, cypher, params=None):
        session = cls.driver.session()
        try:
            results = session.execute_read(cls._execute_read_cypher, cypher, params)
        except Exception as e:
            print(e)
            session.close()
            return None
        session.close()
        return results

    @classmethod
    def execute_write(cls, cypher, params=None):
        session = cls.driver.session()
        try:
            session.execute_write(cls._execute_write_cypher, cypher, params)
        except Exception as e:
            print(e)
            session.close()
        session.close()

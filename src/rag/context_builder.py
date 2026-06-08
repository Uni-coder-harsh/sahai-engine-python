from psycopg2.extras import RealDictCursor

class TutorContextBuilder:
    """
    Assembles contextual academic frames for students based on their 
    lowest performing nodes in the database.
    """
    
    def __init__(self):
        pass

    def build_tutoring_context(self, user_id: str, pg_conn) -> dict:
        """
        Queries the database for weak nodes and formats a prompt helper block.
        """
        with pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Query lowest scoring nodes (Mastery < 0.60)
            cur.execute(
                """
                SELECT ucs.node_id, cn.concept_name, ucs.expected_mastery, ucs.alpha, ucs.beta
                FROM user_cognitive_states ucs
                JOIN concept_nodes cn ON ucs.node_id = cn.node_id
                WHERE ucs.user_id = %s
                ORDER BY ucs.expected_mastery ASC
                LIMIT 3;
                """,
                (user_id,)
            )
            weak_nodes = cur.fetchall()

        context_blocks = []
        for node in weak_nodes:
            context_blocks.append(
                f"- Concept: {node['concept_name']} ({node['node_id']}), "
                f"Mastery: {float(node['expected_mastery']):.2%}, "
                f"Alpha/Beta parameters: ({float(node['alpha']):.1f}, {float(node['beta']):.1f})"
            )

        tutor_frame = (
            "SYSTEM TUTORING CONTEXT CONFLICT DETECTION:\n"
            "The student requires assistance in these top 3 weak areas:\n"
            + "\n".join(context_blocks) + "\n"
            "Task: Adjust practice difficulty down, explain prerequisites, and avoid technical jargon."
        )

        return {
            "user_id": user_id,
            "weak_nodes": weak_nodes,
            "tutor_prompt_context": tutor_frame
        }

context_builder = TutorContextBuilder()

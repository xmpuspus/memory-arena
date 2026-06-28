// Memory Arena universal memory schema.
//
// Nodes:
//   User       - the human the agent is remembering for
//   Session    - one ordered chat session (a sequence of turns)
//   Turn       - one user-or-assistant message in a session
//   Fact       - an extracted fact with valid_at / invalid_at
//   Entity     - any named entity referenced by a fact (person, place, thing)
//
// Relationships:
//   HAS_TURN       (Session)-[HAS_TURN]->(Turn)
//   ASSERTED       (Turn)-[ASSERTED]->(Fact)
//   INVALIDATES    (Fact)-[INVALIDATES]->(Fact)            // explicit supersedes
//   REFERS_TO      (Fact)-[REFERS_TO]->(Entity)
//   CONTRADICTS    (Fact)-[CONTRADICTS]->(Fact)            // soft contradiction
//   EVOLVED_FROM   (Fact)-[EVOLVED_FROM]->(Fact)           // same entity, later version

CREATE CONSTRAINT user_id IF NOT EXISTS
  FOR (u:User) REQUIRE u.id IS UNIQUE;

CREATE CONSTRAINT session_id IF NOT EXISTS
  FOR (s:Session) REQUIRE s.id IS UNIQUE;

CREATE CONSTRAINT turn_id IF NOT EXISTS
  FOR (t:Turn) REQUIRE t.id IS UNIQUE;

CREATE CONSTRAINT fact_id IF NOT EXISTS
  FOR (f:Fact) REQUIRE f.id IS UNIQUE;

CREATE CONSTRAINT entity_id IF NOT EXISTS
  FOR (e:Entity) REQUIRE e.id IS UNIQUE;

// Indexes that speed up benchmark teardown and run-namespaced queries.
CREATE INDEX session_run_id IF NOT EXISTS
  FOR (s:Session) ON (s.run_id);

CREATE INDEX fact_run_id IF NOT EXISTS
  FOR (f:Fact) ON (f.run_id);

CREATE INDEX fact_valid_at IF NOT EXISTS
  FOR (f:Fact) ON (f.valid_at);

CREATE INDEX entity_run_id IF NOT EXISTS
  FOR (e:Entity) ON (e.run_id);

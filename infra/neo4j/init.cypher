CREATE CONSTRAINT event_id IF NOT EXISTS
FOR (e:Event) REQUIRE e.id IS UNIQUE;

CREATE CONSTRAINT character_uid IF NOT EXISTS
FOR (c:Character) REQUIRE c.uid IS UNIQUE;

CREATE INDEX character_doc_name IF NOT EXISTS
FOR (c:Character) ON (c.doc_id, c.name);

CREATE INDEX event_doc IF NOT EXISTS
FOR (e:Event) ON (e.doc_id);

CREATE INDEX event_order IF NOT EXISTS
FOR (e:Event) ON (e.doc_id, e.stage_order, e.first_page);

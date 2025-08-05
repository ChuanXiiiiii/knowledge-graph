#!/usr/bin/env python
# coding: utf-8

# # Create a Graph from a text

# This notebook demonstrates how to extract graph from any text using the graph maker
# 
# Steps:
# - Define an Ontology
# - Load a list of example text chunks. We will use the Lord of the Rings summary from this wikipedia page. 
# - Create Graph using an Open source model using Groq APIs. 
# - Save the graph to Neo4j db
# - Visualise
# 
# 
# 
# Loading the graph maker functions ->

# In[68]:


from knowledge_graph_maker import GraphMaker, Ontology, GroqClient, OpenAIClient
from knowledge_graph_maker import Document


# # Define the Ontology. 
# 
# The ontology is a pydantic model with the following schema. 
# 
# ```python
# class Ontology(BaseModel):
#     label: List[Union[str, Dict]]
#     relationships: List[str]
# ```
# 
# 

# For example lets use summaries of the LOTR books from the Wikipedia page. I have copied it into a file for easy import

# In[69]:


import json

file_path = './data/jsonl/hp.jsonl'

# 打开并读取 .jsonl 文件
with open(file_path, 'r', encoding='utf-8') as file:
    data = []
    # 每一行都是一个 JSON 对象，将其读取并解析为 Python 字典
    for line in file:
        json_object = json.loads(line.strip())  # strip() 去掉换行符
        data.append(json_object)

example_text_list_all = [json_object['text'] for json_object in data]


# In[70]:


example_text_list = example_text_list_all[:1000]


# Here is the ontology we will use for the LOTR summaries ->

# In[71]:


# ontology = Ontology(
#     labels=[
#         {"Person": "Person name without any adjectives, Remember a person may be referenced by their name or using a pronoun"},
#         {"Object": "Do not add the definite article 'the' in the object name"},
#         {"Event": "Event event involving multiple people. Do not include qualifiers or verbs like gives, leaves, works etc."},
#         "Place",
#         "Document",
#         "Organisation",
#         "Action",
#         {"Miscellaneous": "Any important concept can not be categorised with any other given label"},
#     ],
#     relationships=[
#         "Relation between any pair of Entities"
#         ],
# )


# In[72]:


# ontology = Ontology(
#     labels=[
#         # 实体类别
#         {"Person": "Any individual character in the Harry Potter universe, including wizards, witches, muggles, ghosts, house-elves, etc."},
#         {"MagicalCreature": "Any non-human character that is a magical creature, such as house-elves, goblins, centaurs, dementors, etc."},
#         {"Object": "Any magical or non-magical item without including articles like 'the'. Examples: 'Wand', 'Horcrux', 'Invisibility Cloak'."},
#         {"Event": "Any significant event involving characters or groups. Example: 'Triwizard Tournament', 'Battle of Hogwarts'."},
#         {"Spell": "Any magical spell or incantation. Example: 'Expelliarmus', 'Avada Kedavra', 'Lumos'."},
#         {"Place": "Locations in the wizarding world, both magical and non-magical. Example: 'Hogwarts', 'Diagon Alley', 'Forbidden Forest'."},
#         {"Organisation": "Groups or organizations within the universe. Example: 'Ministry of Magic', 'Dumbledore's Army', 'Death Eaters'."},
#         {"Document": "Written records or texts that play a significant role. Example: 'Marauder's Map', 'The Daily Prophet'."},
#         {"Potion": "Magical concoctions brewed in the wizarding world. Example: 'Polyjuice Potion', 'Felix Felicis'."},
#         {"House": "Hogwarts houses to which students belong. Example: 'Gryffindor', 'Slytherin'."},
#         {"QuidditchTeam": "Teams that participate in the sport of Quidditch. Example: 'Chudley Cannons', 'Holyhead Harpies'."},
#         {"Miscellaneous": "Any other important concept that does not fit into other categories, such as magical laws or principles."},
#     ],
# 
#     relationships=[
#         # 实体之间的关系
#         "Relation between any pair of Entities，Return no more than 2 words."
#     ]
# )


# In[73]:


ontology = Ontology(
    labels=[
        # 实体类别
        {"Person": "Any individual character in the Harry Potter universe, including wizards, witches, muggles, ghosts, house-elves, etc."},
        {"MagicalCreature": "Any non-human character that is a magical creature, such as house-elves, goblins, centaurs, dementors, etc."},
        {"Object": "Any magical or non-magical item without including articles like 'the'. Examples: 'Wand', 'Horcrux', 'Invisibility Cloak'."},
        {"Event": "Any significant event involving characters or groups. Example: 'Triwizard Tournament', 'Battle of Hogwarts'."},
        {"Spell": "Any magical spell or incantation. Example: 'Expelliarmus', 'Avada Kedavra', 'Lumos'."},
        {"Place": "Locations in the wizarding world, both magical and non-magical. Example: 'Hogwarts', 'Diagon Alley', 'Forbidden Forest'."},
        {"Organisation": "Groups or organizations within the universe. Example: 'Ministry of Magic', 'Dumbledore's Army', 'Death Eaters'."},
        {"Document": "Written records or texts that play a significant role. Example: 'Marauder's Map', 'The Daily Prophet'."},
        {"Potion": "Magical concoctions brewed in the wizarding world. Example: 'Polyjuice Potion', 'Felix Felicis'."},
        {"House": "Hogwarts houses to which students belong. Example: 'Gryffindor', 'Slytherin'."},
        {"QuidditchTeam": "Teams that participate in the sport of Quidditch. Example: 'Chudley Cannons', 'Holyhead Harpies'."},
        {"Miscellaneous": "Any other important concept that does not fit into other categories, such as magical laws or principles."},
    ],

    relationships=[
        "FamilyRelation: Indicates family ties between people, such as 'Father of', 'Sister of', 'Ancestor of', 'Descendant of'.",
        "Friendship: A friendly or mentor-like relationship between people, such as 'Best friend of', 'Mentor of'.",
        "Rivalry: A competitive or antagonistic relationship between characters, such as 'Rival of', 'Enemy of'.",
        "Marriage: Married relationship between characters, such as 'Spouse of', 'Wife of', 'Husband of'.",

        "Possession: Indicates ownership or possession of an object, such as 'Owner of', 'Possesses'.",
        "CreatorOfObject: Indicates the person who created or crafted an object, such as 'Creator of', 'Maker of'.",
        "HorcruxConnection: Relationship between a person and a Horcrux, such as 'Created Horcrux', 'Contains part of soul'.",
        "MasterOf: Indicates mastery or control over a magical object, such as 'Master of the Elder Wand'.",

        "PerformsSpell: Indicates when a person casts or performs a spell, such as 'Casts spell', 'Performs'.",
        "VictimOfSpell: Indicates a person who was affected by a spell, such as 'Victim of', 'Target of'.",
        "PotionBrewer: Indicates the person who brewed or created a potion, such as 'Brews potion'.",
        "PotionConsumer: Indicates the person who consumes or uses a potion, such as 'Consumes potion'.",
        "ParticipatesInEvent: Indicates participation in an event, such as 'Participant of', 'Fights in'.",
        "WinsEvent: Indicates a person or team that wins a specific event, such as 'Wins' the 'Triwizard Tournament'.",
        "LosesEvent: Indicates a person or team that loses a specific event, such as 'Loses' a Quidditch match.",
        "LeadsEvent: A person or group leading an event, such as 'Leads', 'In charge of' a battle or operation.",

        "LocatedIn: Indicates the location of an object, person, or event, such as 'Located at', 'In'.",
        "TravelTo: Indicates travel to a specific place, such as 'Travels to', 'Goes to' a location.",
        "BornIn: The place of birth of a character, such as 'Born in Godric's Hollow'.",
        "ResidesIn: The place where a person lives or resides, such as 'Lives in' Hogwarts.",
        "DestroyedAt: Indicates where an object or thing was destroyed, such as 'Destroyed at' the 'Chamber of Secrets'.",

        "Membership: Indicates membership in an organization, such as 'Member of', 'Leader of'.",
        "Leadership: Indicates leadership or authority within an organization, such as 'Leader of', 'Head of'.",
        "QuidditchTeamMember: Membership in a Quidditch team, such as 'Plays for', 'Captain of'.",
        "HouseAffiliation: Indicates the Hogwarts house a person belongs to, such as 'Belongs to Gryffindor'.",
        "TeachesAt: Indicates where a person teaches, such as 'Teaches at Hogwarts'.",
        "StudentAt: Indicates a person studying at a particular institution, such as 'Studies at Hogwarts'.",

        "BattleParticipation: Participation in a battle, such as 'Fights in Battle of Hogwarts'.",
        "WitnessesEvent: Indicates a person who witnessed an event, such as 'Witnesses the return of Voldemort'.",
        "DestroysObject: Indicates destruction of an object by someone, such as 'Destroys the Horcrux'.",
        "CreatesSpell: Indicates the person who invents or creates a spell, such as 'Creates spell' like 'Sectumsempra'.",

        "HasSpellEffect: The effect of a spell on a person or object, such as 'Affected by Expelliarmus'.",
        "BreaksCurse: Indicates a person who breaks a magical curse, such as 'Breaks curse on a person'.",
        "UnderCurse: Indicates when a person or object is under a curse, such as 'Cursed with the Imperius Curse'.",
        "TransformedInto: Denotes transformation from one state or being to another, such as 'Transformed into an Animagus form'.",
        "KilledBy: Indicates that a person or creature was killed by another, such as 'Killed by Voldemort'.",

        "WritesDocument: Indicates a person who writes or authors a document, such as 'Writes the Marauder's Map'.",
        "ReadsDocument: Indicates a person who reads or studies a document, such as 'Reads from a magical book'.",
        "DestroyedDocument: Indicates a document that is destroyed by someone, such as 'Destroys a cursed letter'.",

        "RevengeAgainst: Indicates a person seeking revenge against another, such as 'Seeks revenge against Voldemort'.",
        "Supports: Indicates a supportive or allied relationship between entities, such as 'Supports Dumbledore's Army'.",
        "Betrays: Indicates a betrayal between two entities, such as 'Betrays the Order of the Phoenix'.",
    ]
)


# In[73]:





# ## Select a Model
# 
# Groq support the following models at present. 
# 
# *LLaMA3 8b*
# Model ID: llama3-8b-8192
# 
# *LLaMA3 70b*
# Model ID: llama3-70b-8192
# 
# *Mixtral 8x7b*
# Model ID: mixtral-8x7b-32768
# 
# *Gemma 7b*
# Model ID: gemma-7b-it
# 
# 
# Selecting a model for this example ->
# 

# In[74]:


## Groq models
# model = "mixtral-8x7b-32768"
# model ="llama3-8b-8192"
# model = "llama3-70b-8192"
# model="gemma-7b-it"

## Open AI models
oai_model="gpt-3.5-turbo"

## Use Groq
# llm = GroqClient(model=model, temperature=0.1, top_p=0.5)
## OR Use OpenAI
llm = OpenAIClient(model=oai_model, temperature=0.1, top_p=0.5)


# ## Create documents out of text chumks. 
# Documents is a pydantic model with the following schema 
# 
# ```python
# class Document(BaseModel):
#     text: str
#     metadata: dict
# ```
# 
# The metadata we add to the document here is copied to every relation that is extracted out of the document. More often than not, the node pairs have multiple relation with each other. The metadata helps add more context to these relations
# 
# In this example I am generating a summary of the text chunk, and the timestamp of the run, to be used as metadata. 
# 

# In[75]:


import datetime
current_time = str(datetime.datetime.now())


graph_maker = GraphMaker(ontology=ontology, llm_client=llm, verbose=False)

def generate_summary(text):
    SYS_PROMPT = (
        "Succintly summarise the text provided by the user. "
        "Respond only with the summary and no other comments"
    )
    try:
        summary = llm.generate(user_message=text, system_message=SYS_PROMPT)
    except:
        summary = ""
    finally:
        return summary


docs = map(
    lambda t: Document(text=t, metadata={"summary": generate_summary(t), 'generated_at': current_time}),
    example_text_list
)


# ## Create Graph
# Finally run the Graph Maker to generate graph. 

# In[76]:


graph = graph_maker.from_documents(
    list(docs), 
    delay_s_between=0 ## delay_s_between because otherwise groq api maxes out pretty fast. 
    ) 
print("Total number of Edges", len(graph))


# In[77]:


for edge in graph:
    # print(edge.model_dump(exclude=['metadata']), "\n\n")
    print(edge.model_dump()['relationship'])


# # Save the Graph to Neo4j 

# In[78]:


# from knowledge_graph_maker import Neo4jGraphModel
# 
# create_indices = False
# neo4j_graph = Neo4jGraphModel(edges=graph, create_indices=create_indices)
# 
# neo4j_graph.save()
# print('构建图谱成功')


# In[79]:


from py2neo import Graph, Node, Relationship, NodeMatcher

URL = "neo4j://localhost:7687"
AUTH = ("neo4j", "neo4j123456")
py2neograph = Graph(URL, auth=AUTH, name='neo4j')
matcher = NodeMatcher(py2neograph)
py2neograph.delete_all()  # 导入前清空数据库


# In[80]:


for edge in graph:
    print(edge)
    node1 = Node(edge.node_1.label,name=edge.node_1.name)
    py2neograph.merge(node1,edge.node_1.label,'name')
    node2 = Node(edge.node_2.label,name=edge.node_2.name)
    py2neograph.merge(node2,edge.node_2.label,'name')
    
    relationship = Relationship(node1, edge.relationship, node2,**edge.metadata)
    py2neograph.merge(relationship)

print("Graph saved to Neo4j")
    


# In[82]:


import pickle
# 将对象保存到文件中
with open('./data/graph_object.pkl', 'wb') as file:
    pickle.dump(graph, file)


# In[85]:


# 删除多余的重复关系
query = """
MATCH (a)-[r]->(b)
WITH a, b, COLLECT(r) AS relationships
WHERE SIZE(relationships) > 1
FOREACH (r IN relationships[1..] | DELETE r)
"""
py2neograph.run(query)


# In[86]:


# 删除指向自己的关系
query = """
MATCH (a)-[r]->(b)
WHERE a=b
delete r
"""
py2neograph.run(query)


# In[87]:


# 删除反向的关系
query = """
MATCH (a)-[r]->(b),(a)<-[r2]-(b)
delete r2
"""
py2neograph.run(query)


# In[88]:


import json
with open('./data/merge_relationships.json', 'r',encoding='utf-8') as f:
    relationship_list = json.load(f)


# In[104]:


for key,value_list in relationship_list.items():
    for value in value_list:
        cql = f'''
        MATCH (n1)-[oldRel:`{value}`]->(n2)
        CREATE (n1)-[:`{key}`]->(n2)
        WITH oldRel
        DELETE oldRel
        '''
        print(cql)
        py2neograph.run(cql)
print("删除重复关系完成")


# In[ ]:





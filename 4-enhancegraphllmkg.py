import json
with open('data/new/all_list_result-new.json',encoding='utf-8') as f:
    all_list_kg = json.load(f)
#%%
all_list_kg
#%%
import json

file_path = './data/jsonl/hp.jsonl'

# 打开并读取 .jsonl 文件
with open(file_path, 'r', encoding='utf-8') as file:
    data = []
    # 每一行都是一个 JSON 对象，将其读取并解析为 Python 字典
    for line in file:
        json_object = json.loads(line.strip())  # strip() 去掉换行符
        data.append(json_object)

hp_text_list_all = [json_object['text'] for json_object in data]
#%%
entity_rel_map_val = {
    't': set(),
    'f': set(),
    'n':set()
}
#%%
for item in all_list_kg:
    result_list = item[-1]
    for result in result_list:
        if result[-1] == 1:
            entity_rel_map_val['t'].add(tuple(result[:-1]))
        elif result[-1] == 0:
            entity_rel_map_val['f'].add(tuple(result[:-1]))
        else:
            entity_rel_map_val['n'].add(tuple(result[:-1]))
#%%
entity_rel_map_val
#%%
from knowledge_graph_maker import GraphMaker, Ontology, GroqClient, OpenAIClient,OllamaClient
from knowledge_graph_maker import Document
#%%
ontology = Ontology(
    labels=[
        # Entity Categories
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
        {"Artifact": "Magical items with special significance or power. Example: 'Deathly Hallows', 'Time-Turner'."},
        {"Animal": "Ordinary animals without magical abilities. Example: 'Owl', 'Cat'."},
        {"Plant": "Plants with magical properties. Example: 'Mandrake', 'Dragon Blood Vine'."},
        {"Vehicle": "Modes of transportation in the wizarding world. Example: 'Broomstick', 'Knight Bus'."},
        {"Currency": "Currencies used in the wizarding world. Example: 'Galleon', 'Sickle', 'Knut'."},
        {"Title": "Titles or designations held by characters. Example: 'Headmaster', 'Professor', 'Minister'."},
        {"Curse": "Spells with negative effects. Example: 'Imperius Curse', 'Cruciatus Curse'."},
        {"Blessing": "Spells with positive effects. Example: 'Patronus Charm'."},
        {"Ritual": "Magical rituals or ceremonies. Example: 'Horcrux creation ritual'."},
        {"Prophecy": "Foretellings or predictions about the future. Example: 'The prophecy about Harry'."},
        {"Relic": "Items of historical or cultural importance. Example: 'Godric's Relics'."},
    ],

    relationships=[
        "Relation between any pair of Entities"]
)

#%%
# ## Open AI models
# oai_model="gpt-4-turbo-ca"
# llm = OpenAIClient(model=oai_model, temperature=0.1, top_p=0.5)
#%%
model="llama3.1:8b"
llm = OllamaClient(model=model, temperature=0.1, top_p=0.5)
#%%
use_llm_doc_list = hp_text_list_all
len(use_llm_doc_list)
#%%
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
        print("Summary:", summary)
        return summary


docs = map(
    lambda t: Document(text=t, metadata={"summary": generate_summary(t), 'generated_at': current_time}),
    use_llm_doc_list
)
#%%
graph = graph_maker.from_documents(
    list(docs),
    delay_s_between=0 ## delay_s_between because otherwise groq api maxes out pretty fast.
    )
print("Total number of Edges", len(graph))
#%%
import pickle
# 将对象保存到文件中
with open('./data/graph_object_all.pkl', 'wb') as file:
    pickle.dump(graph, file)
print("Graph saved")
input('请关闭窗口')
import chromadb
#craeting a client to access the features of chromadb
client = chromadb.PersistentClient("./data")

#this is the name of the collection similar to the name of table as in sql
collection_name = "vehicle"

collection = client.get_or_create_collection(collection_name)

print(f"Collection created: {collection.name}")

#adding data to the collection

datas = [
    {"id":"car", "text":"Car runs on the land"},
    {"id":"Plane", "text":"Plane flies in the sky"},
    {"id":"Boat", "text":"Boat travels on water"},
    {"id":"Bus", "text":"Bus is public transport on road"},
]

collection.add(
    documents = [
        "Car runs on the land",
        "Plane flies in the sky",
        "Boat travels on water",
        "Bus is public transport on road",
    ],
    ids = ["car", "plane", "boat", "bus"]


)
#query the collection
query_texts = ["Which vehicle run on the road?"]


results = collection.query(
    query_texts=[query_texts[0]],
    n_results= 2
)

print("This is the result: ",results)
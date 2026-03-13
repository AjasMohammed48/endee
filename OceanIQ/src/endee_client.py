from endee import Endee

class EndeeClient:

    def __init__(self, host="http://localhost:8080", index="ocean_vectors"):
        self.index_name = index
        self.client = Endee()
        self.client.set_base_url(f"{host}/api/v1")
        self._index = self.client.get_index(name=index)

    def insert_vectors(self, vectors):
        try:
            self._index.upsert(vectors)
            print(f"Insert status: 200")
            return "OK"
        except Exception as e:
            print(f"Insert error: {e}")
            return str(e)

    def search(self, vector, k=5):
        try:
            results = self._index.query(vector=vector, top_k=k)
            return {"results": results}
        except Exception as e:
            print(f"Search error: {e}")
            return {"results": []}
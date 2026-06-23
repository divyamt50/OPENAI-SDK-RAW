from pydantic import BaseModel



class Ask(BaseModel):
    query:str

class AskList(BaseModel):
    queries:list[str]
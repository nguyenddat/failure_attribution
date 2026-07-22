from pydantic import BaseModel


class AllAtOnceIn(BaseModel):
    trajectory: str
    failure_mode: str


def all_at_once():
    pass

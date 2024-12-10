from pydantic import BaseModel
from typing import Optional 

class UserBase(BaseModel):
	username : str 
	class Config:
		from_attributes = True
	
class UserCreation(UserBase):
	password:str
	
class UserUpdate(UserBase):
	pass
	
class UserDB(UserBase):
	id: int
	hashed_password: str
	
# Pydantic Models


class TokenData(BaseModel):
    username: Optional[str] = None

##class User(BaseModel):
##    username: str

##class UserCreate(BaseModel):
##    username: str
##    password: str

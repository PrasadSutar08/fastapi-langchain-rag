from sqlmodel import Field, SQLModel


class UserBase(SQLModel):
    email: str = Field(unique=True, index=True)
    is_active: bool = True
    is_superuser: bool = False
    full_name: str | None = None


class UserCreate(UserBase):
    password: str


class UserRegister(SQLModel):
    email: str
    password: str
    full_name: str | None = None


class UserUpdate(UserBase):
    email: str | None = None
    password: str | None = None


class UserUpdateMe(SQLModel):
    full_name: str | None = None
    email: str | None = None


class UpdatePassword(SQLModel):
    current_password: str
    new_password: str


class User(UserBase, table=True):
    __tablename__ = "users"
    id: int | None = Field(default=None, primary_key=True)
    hashed_password: str


class ItemBase(SQLModel):
    title: str
    description: str | None = None


class ItemCreate(ItemBase):
    pass


class ItemUpdate(ItemBase):
    title: str | None = None


class Item(ItemBase, table=True):
    __tablename__ = "items"
    id: int | None = Field(default=None, primary_key=True)
    owner_id: int = Field(foreign_key="users.id", nullable=False)


class UserOut(UserBase):
    id: int


class UsersOut(SQLModel):
    data: list[UserOut]
    count: int


class ItemOut(ItemBase):
    id: int
    owner_id: int


class ItemsOut(SQLModel):
    data: list[ItemOut]
    count: int


class Message(SQLModel):
    message: str


class Token(SQLModel):
    access_token: str
    token_type: str = "bearer"


class TokenPayload(SQLModel):
    sub: int | None = None
    exp: int | None = None
    iat: int | None = None


class NewPassword(SQLModel):
    token: str
    new_password: str

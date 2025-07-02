from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, Optional

from passlib.context import CryptContext


class UserManagerService:
    """Manage user accounts and authentication tokens."""

    def __init__(self, users_file: str = "users.json") -> None:
        self.path = Path(users_file)
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.users: Dict[str, str] = {}
        self.tokens: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            self.users = json.loads(self.path.read_text())

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.users))

    def create_user(self, username: str, password: str) -> bool:
        if username in self.users:
            return False
        self.users[username] = self.pwd_context.hash(password)
        self._save()
        return True

    def authenticate(self, username: str, password: str) -> Optional[str]:
        hashed = self.users.get(username)
        if not hashed or not self.pwd_context.verify(password, hashed):
            return None
        token = str(uuid.uuid4())
        self.tokens[token] = username
        return token

    def validate_token(self, token: str) -> bool:
        return token in self.tokens

    def get_user(self, username: str) -> Optional[Dict[str, str]]:
        if username not in self.users:
            return None
        return {"username": username}

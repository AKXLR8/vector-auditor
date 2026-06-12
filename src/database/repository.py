"""Repository pattern over SQLAlchemy models. Falls back to in-memory stores when DB unavailable.

All field names match the actual Neon schema in `src/database/models.py`.
"""
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
    BlacklistedToken,
    DeadLetter,
    Document,
    Feedback,
    Message,
    Query,
    Session,
    UploadJob,
    User,
)

logger = logging.getLogger("rga_auditor.repo")

_PERSIST_DIR = Path(".data")
_PERSIST_DIR.mkdir(exist_ok=True)


# ── In-memory fallback (persisted to disk as JSONL) ──────────────────────────

class _MemoryStore:
    def __init__(self, name: str) -> None:
        self.name = name
        self._path = _PERSIST_DIR / f"{name}.jsonl"
        self._items: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with self._path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        self._items[item["id"]] = item
            except Exception as e:
                logger.warning("memory store %s load failed: %s", self.name, e)

    def _persist(self) -> None:
        try:
            with self._path.open("w", encoding="utf-8") as f:
                for item in self._items.values():
                    f.write(json.dumps(item, default=str) + "\n")
        except Exception as e:
            logger.warning("memory store %s persist failed: %s", self.name, e)

    def add(self, item: dict) -> None:
        self._items[item["id"]] = item
        self._persist()

    def update(self, item_id: str, **fields: Any) -> Optional[dict]:
        item = self._items.get(item_id)
        if not item:
            return None
        item.update(fields)
        self._persist()
        return item

    def get(self, item_id: str) -> Optional[dict]:
        return self._items.get(item_id)

    def list(self, **filters: Any) -> list[dict]:
        items = list(self._items.values())
        for k, v in filters.items():
            items = [i for i in items if i.get(k) == v]
        return items

    def delete(self, item_id: str) -> None:
        self._items.pop(item_id, None)
        self._persist()


_users = _MemoryStore("users")
_docs = _MemoryStore("documents")
_blacklist = _MemoryStore("blacklisted_tokens")
_dlq = _MemoryStore("dead_letter_queue")
_sessions = _MemoryStore("chat_sessions")
_messages = _MemoryStore("chat_messages")
_queries = _MemoryStore("queries")
_uploads = _MemoryStore("upload_progress")


# ── Helpers ─────────────────────────────────────────────────────────────────

def _display_name(user_or_dict: dict) -> Optional[str]:
    if user_or_dict.get("display_name"):
        return user_or_dict["display_name"]
    if user_or_dict.get("email"):
        return user_or_dict["email"].split("@", 1)[0]
    return None


def _parse_roles(value: Any) -> list[str]:
    if not value:
        return ["user"]
    if isinstance(value, list):
        return [str(r) for r in value if r]
    if isinstance(value, str):
        try:
            v = json.loads(value)
            if isinstance(v, list):
                return [str(r) for r in v if r]
            return [str(v)] if v else ["user"]
        except Exception:
            return [value] if value else ["user"]
    return [str(value)]


# ── Users ───────────────────────────────────────────────────────────────────

async def get_user_by_email(session: Optional[AsyncSession], email: str) -> Optional[dict]:
    if session is not None:
        r = await session.execute(select(User).where(User.email == email))
        u = r.scalar_one_or_none()
        return _user_to_dict(u) if u else None
    matches = _users.list(email=email)
    return matches[0] if matches else None


async def get_user_by_id(session: Optional[AsyncSession], user_id: str) -> Optional[dict]:
    if session is not None:
        r = await session.execute(select(User).where(User.id == user_id))
        u = r.scalar_one_or_none()
        return _user_to_dict(u) if u else None
    return _users.get(user_id)


async def create_user(
    session: Optional[AsyncSession],
    email: str,
    hashed_password: str,
    username: Optional[str] = None,
    display_name: Optional[str] = None,
    roles: Optional[list[str]] = None,
) -> dict:
    if roles is None or not roles:
        roles = ["user"]
    user = {
        "id": uuid.uuid4().hex,
        "email": email,
        "hashed_password": hashed_password,
        "roles": json.dumps(roles),
        "mfa_enabled": False,
        "mfa_secret": None,
        "username": username,
        "display_name": display_name,
        "created_at": datetime.utcnow(),
    }
    if session is not None:
        u = User(**user)
        session.add(u)
        await session.commit()
        await session.refresh(u)
        return _user_to_dict(u)
    _users.add(user)
    return user


def _user_to_dict(u: User) -> dict:
    roles = _parse_roles(getattr(u, "roles", None))
    return {
        "id": u.id,
        "email": u.email,
        "hashed_password": u.hashed_password,
        "roles": roles,
        "mfa_enabled": u.mfa_enabled,
        "mfa_secret": u.mfa_secret,
        "username": getattr(u, "username", None),
        "display_name": getattr(u, "display_name", None),
        "created_at": u.created_at.isoformat() if u.created_at else None,
    }


# ── Token blacklist ─────────────────────────────────────────────────────────

async def blacklist_token(session: Optional[AsyncSession], jti: str, user_id: str, expires_at: Optional[datetime] = None) -> None:
    if session is not None:
        session.add(BlacklistedToken(
            id=uuid.uuid4().hex,
            token_jti=jti,
            expires_at=expires_at or datetime.utcnow(),
        ))
        await session.commit()
        return
    _blacklist.add({
        "id": uuid.uuid4().hex,
        "token_jti": jti,
        "expires_at": (expires_at or datetime.utcnow()).isoformat(),
        "created_at": datetime.utcnow().isoformat(),
    })


async def is_token_blacklisted(session: Optional[AsyncSession], jti: str) -> bool:
    if session is not None:
        r = await session.execute(select(BlacklistedToken).where(BlacklistedToken.token_jti == jti))
        return r.scalar_one_or_none() is not None
    return any(item.get("token_jti") == jti for item in _blacklist.list())


# ── Documents ───────────────────────────────────────────────────────────────

async def create_document(session: Optional[AsyncSession], **fields: Any) -> dict:
    doc = {
        "id": uuid.uuid4().hex,
        "status": "processing",
        "has_pii": False,
        **fields,
    }
    if session is not None:
        d = Document(**doc)
        session.add(d)
        await session.commit()
        return _doc_to_dict(d)
    _docs.add(doc)
    return doc


async def get_document(session: Optional[AsyncSession], doc_id: str) -> Optional[dict]:
    if session is not None:
        r = await session.execute(select(Document).where(Document.id == doc_id))
        d = r.scalar_one_or_none()
        return _doc_to_dict(d) if d else None
    return _docs.get(doc_id)


async def list_documents(session: Optional[AsyncSession], user_id: Optional[str] = None) -> list[dict]:
    if session is not None:
        stmt = select(Document).order_by(Document.created_at.desc())
        if user_id:
            stmt = stmt.where(Document.uploaded_by == user_id)
        r = await session.execute(stmt)
        return [_doc_to_dict(d) for d in r.scalars().all()]
    items = _docs.list()
    if user_id:
        items = [i for i in items if i.get("uploaded_by") == user_id]
    return sorted(items, key=lambda x: x.get("created_at") or "", reverse=True)


async def get_document_by_sha256(session: AsyncSession, sha256: str) -> Optional[dict]:
    r = await session.execute(select(Document).where(Document.sha256 == sha256))
    d = r.scalar_one_or_none()
    return _doc_to_dict(d) if d else None


async def update_document(session: Optional[AsyncSession], doc_id: str, **fields: Any) -> Optional[dict]:
    if session is not None:
        r = await session.execute(select(Document).where(Document.id == doc_id))
        d = r.scalar_one_or_none()
        if d is None:
            return None
        for k, v in fields.items():
            setattr(d, k, v)
        await session.commit()
        return _doc_to_dict(d)
    return _docs.update(doc_id, **fields)


async def delete_document(session: Optional[AsyncSession], doc_id: str) -> None:
    if session is not None:
        await session.execute(delete(Document).where(Document.id == doc_id))
        await session.commit()
        return
    _docs.delete(doc_id)


def _doc_to_dict(d: Document) -> dict:
    return {
        "id": d.id,
        "uploaded_by": d.uploaded_by,
        "filename": d.filename,
        "status": d.status,
        "has_pii": d.has_pii,
        "sha256": d.sha256,
        "cloudinary_url": d.cloudinary_url,
        "cloudinary_public_id": d.cloudinary_public_id,
        "created_at": d.created_at.isoformat() if d.created_at else None,
    }


# ── Dead-letter queue ───────────────────────────────────────────────────────

async def push_dlq(session: Optional[AsyncSession], source: str, error: str, payload: str = "", filename: str = "") -> None:
    item = {
        "id": uuid.uuid4().hex,
        "source": source,
        "filename": filename,
        "error": error,
        "payload": payload,
        "retry_count": 0,
        "created_at": datetime.utcnow(),
    }
    if session is not None:
        session.add(DeadLetter(
            id=item["id"],
            source=source,
            filename=filename,
            error=error,
            payload=payload,
            retry_count=0,
        ))
        await session.commit()
        return
    _dlq.add(item)


async def list_dlq(session: Optional[AsyncSession], limit: int = 100) -> list[dict]:
    if session is not None:
        r = await session.execute(select(DeadLetter).order_by(DeadLetter.created_at.desc()).limit(limit))
        return [
            {
                "id": d.id,
                "source": d.source,
                "filename": d.filename,
                "error": d.error,
                "payload": d.payload,
                "retry_count": d.retry_count,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in r.scalars().all()
        ]
    return sorted(_dlq.list(), key=lambda x: x.get("created_at") or "", reverse=True)[:limit]


async def dlq_size(session: Optional[AsyncSession]) -> int:
    if session is not None:
        from sqlalchemy import func
        r = await session.execute(select(func.count()).select_from(DeadLetter))
        return int(r.scalar() or 0)
    return len(_dlq.list())


# ── Feedback ────────────────────────────────────────────────────────────────

async def add_feedback(
    session: Optional[AsyncSession],
    user_id: str,
    query_id: str,
    thumbs_up: bool,
    comment: Optional[str] = None,
) -> None:
    if session is not None:
        session.add(Feedback(
            id=uuid.uuid4().hex,
            user_id=user_id,
            query_id=query_id,
            thumbs_up=thumbs_up,
            comment=comment,
        ))
        await session.commit()


# ── Sessions & messages ─────────────────────────────────────────────────────

async def list_sessions(session: Optional[AsyncSession], user_id: str) -> list[dict]:
    if session is not None:
        r = await session.execute(select(Session).where(Session.user_id == user_id).order_by(Session.updated_at.desc()))
        return [_session_to_dict(s) for s in r.scalars().all()]
    return sorted(
        [s for s in _sessions.list() if s.get("user_id") == user_id],
        key=lambda x: x.get("updated_at") or "",
        reverse=True,
    )


async def get_session(session: Optional[AsyncSession], sid: str, user_id: str) -> Optional[dict]:
    if session is not None:
        r = await session.execute(select(Session).where(Session.id == sid, Session.user_id == user_id))
        s = r.scalar_one_or_none()
        return _session_to_dict(s) if s else None
    for s in _sessions.list():
        if s.get("id") == sid and s.get("user_id") == user_id:
            return s
    return None


async def create_session(session: Optional[AsyncSession], sid: str, user_id: str, title: str) -> dict:
    now = datetime.utcnow()
    s = {"id": sid, "user_id": user_id, "title": title, "created_at": now, "updated_at": now}
    if session is not None:
        row = Session(id=sid, user_id=user_id, title=title)
        session.add(row)
        await session.commit()
        return s
    _sessions.add({**s, "created_at": now.isoformat(), "updated_at": now.isoformat()})
    return s


async def update_session_title(session: Optional[AsyncSession], sid: str, user_id: str, title: str) -> Optional[dict]:
    if session is not None:
        r = await session.execute(select(Session).where(Session.id == sid, Session.user_id == user_id))
        row = r.scalar_one_or_none()
        if row is None:
            return None
        row.title = title
        row.updated_at = datetime.utcnow()
        await session.commit()
        return _session_to_dict(row)
    s = await get_session(session, sid, user_id)
    if s is None:
        return None
    s["title"] = title
    s["updated_at"] = datetime.utcnow().isoformat()
    _sessions.update(sid, title=title, updated_at=s["updated_at"])
    return s


async def delete_session(session: Optional[AsyncSession], sid: str, user_id: str) -> None:
    if session is not None:
        from sqlalchemy import delete as _d
        await session.execute(_d(Session).where(Session.id == sid, Session.user_id == user_id))
        await session.commit()
        return
    s = await get_session(session, sid, user_id)
    if s is not None:
        _sessions.delete(sid)
    for m in list(_messages.list()):
        if m.get("session_id") == sid:
            _messages.delete(m["id"])


async def list_messages(session: Optional[AsyncSession], sid: str, user_id: str) -> list[dict]:
    if session is not None:
        r = await session.execute(select(Session).where(Session.id == sid, Session.user_id == user_id))
        if r.scalar_one_or_none() is None:
            return []
        mr = await session.execute(select(Message).where(Message.session_id == sid).order_by(Message.created_at.asc()))
        return [_message_to_dict(m) for m in mr.scalars().all()]
    sess = await get_session(session, sid, user_id)
    if sess is None:
        return []
    return sorted(
        [m for m in _messages.list() if m.get("session_id") == sid],
        key=lambda x: x.get("created_at") or "",
    )


async def add_message(session: Optional[AsyncSession], sid: str, user_id: str, **fields) -> Optional[dict]:
    sess = await get_session(session, sid, user_id)
    if sess is None:
        return None
    fields.setdefault("session_id", sid)
    fields.setdefault("id", uuid.uuid4().hex)
    fields.setdefault("created_at", datetime.utcnow())
    if session is not None:
        m = Message(**{k: v for k, v in fields.items() if hasattr(Message, k)})
        session.add(m)
        await session.commit()
        await session.refresh(m)
        return _message_to_dict(m)
    payload = {**fields, "created_at": fields["created_at"].isoformat() if isinstance(fields["created_at"], datetime) else fields["created_at"]}
    _messages.add(payload)
    return payload


def _session_to_dict(s: Session) -> dict:
    return {
        "id": s.id,
        "user_id": s.user_id,
        "title": s.title,
        "created_at": s.created_at.isoformat() if s.created_at else None,
        "updated_at": s.updated_at.isoformat() if s.updated_at else None,
    }


def _message_to_dict(m: Message) -> dict:
    return {
        "id": m.id,
        "session_id": m.session_id,
        "role": m.role,
        "content": m.content,
        "citations": json.loads(m.citations) if m.citations else None,
        "reasoning_path": json.loads(m.reasoning_path) if m.reasoning_path else None,
        "tokens_used": m.tokens_used,
        "cost_usd": m.cost_usd,
        "query_id": m.query_id,
        "feedback": m.feedback,
        "verification": m.verification,
        "created_at": m.created_at.isoformat() if m.created_at else None,
    }

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter(prefix="/api/admin", tags=["admin"])


def require_admin(payload: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    user = db.query(User).get(int(payload["sub"]))
    if not user or not user.is_admin:
        raise HTTPException(403, "需要管理员权限")
    return user


@router.get("/users")
def list_users(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [
        {"id": u.id, "username": u.username, "email": u.email,
         "is_admin": u.is_admin, "created_at": u.created_at.isoformat() if u.created_at else None}
        for u in users
    ]


@router.delete("/users/{user_id}")
def delete_user(user_id: int, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    if user_id == admin.id:
        raise HTTPException(400, "不能删除自己")
    user = db.query(User).get(user_id)
    if not user:
        raise HTTPException(404, "用户不存在")
    db.delete(user)
    db.commit()
    return {"message": f"用户 {user.username} 已删除"}


@router.put("/users/{user_id}/role")
def toggle_admin(user_id: int, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    if user_id == admin.id:
        raise HTTPException(400, "不能修改自己的角色")
    user = db.query(User).get(user_id)
    if not user:
        raise HTTPException(404, "用户不存在")
    user.is_admin = not user.is_admin
    db.commit()
    return {"message": f"用户 {user.username} 管理员状态: {user.is_admin}"}

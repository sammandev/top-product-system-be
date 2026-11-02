from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db import Base
from app.models.app_config import AppConfig, GuestCredential, IplasToken, SfistspConfig
from app.models.cached_test_item import CachedTestItem
from app.models.menu_access import MenuDefinition, MenuRoleAccess
from app.models.rbac import Permission, Role
from app.models.top_product import TopProduct, TopProductMeasurement
from app.models.user import User, UserRole
from app.routers.admin_access_control import AccessControlUserSchema
from app.routers.admin_users import _serialize_user


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        yield db
    engine.dispose()


@pytest.mark.parametrize(
    ("table", "nullable", "required"),
    [
        ("users", ["email", "worker_id", "is_admin", "is_ptb_admin", "is_superuser", "is_staff", "is_active", "menu_permissions", "last_login"], ["id", "username", "password_hash", "role", "token_version", "created_at"]),
        ("roles", ["description"], ["id", "name"]),
        ("permissions", ["description"], ["id", "name"]),
        ("menu_definitions", ["icon", "sort_order", "is_active", "parent_key", "description"], ["id", "menu_key", "title", "path", "section"]),
        ("menu_role_access", ["can_view"], ["id", "menu_id", "role_name"]),
        ("top_products", ["pass_count", "fail_count", "retest_count", "test_date", "project_name"], ["id", "dut_isn", "station_name", "score", "created_at"]),
        ("top_product_measurements", ["usl", "lsl", "actual_value"], ["id", "top_product_id", "test_item"]),
        ("app_config", ["description", "favicon_path", "primeui_license_key", "updated_by"], ["id", "name", "version", "updated_at"]),
    ],
)
def test_typed_columns_preserve_schema_nullability(table, nullable, required):
    columns = Base.metadata.tables[table].columns
    assert all(columns[name].nullable for name in nullable)
    assert all(not columns[name].nullable for name in required)


def test_user_and_rbac_roundtrip_preserves_defaults_and_response(session: Session):
    permission = Permission(name="read")
    role = Role(name="reader", permissions=[permission])
    user = User(username="typed-user", password_hash="test-hash", roles=[role])
    session.add(user)
    session.commit()
    session.expire_all()

    loaded = session.scalars(select(User).where(User.username == "typed-user")).one()
    assert isinstance(loaded.id, int)
    assert loaded.role is UserRole.user
    assert loaded.is_active is True
    assert loaded.is_admin is False
    assert loaded.token_version == 1
    assert isinstance(loaded.created_at, datetime)
    assert loaded.roles[0].permissions[0].name == "read"
    assert permission.roles[0].users[0] is loaded
    assert _serialize_user(loaded).model_dump()["role"] == "user"
    assert _serialize_user(loaded).roles == ["reader"]
    response = AccessControlUserSchema.model_validate(loaded, from_attributes=True)
    assert response.role == "user"
    assert response.is_active is True


def test_user_response_rejects_missing_active_status():
    user = User(id=1, username="invalid-user", role=UserRole.user, is_active=None, created_at=datetime.now(UTC))
    with pytest.raises(ValueError, match="has no active status"):
        _serialize_user(user)
    with pytest.raises(ValidationError):
        AccessControlUserSchema.model_validate(user, from_attributes=True)


def test_configuration_and_cache_roundtrip(session: Session):
    config = AppConfig(name="Test", version="1")
    token = IplasToken(site="PTB", base_url="http://example.test", token_value="encrypted")
    sfis = SfistspConfig(base_url="http://example.test", program_id="test", program_password="encrypted")
    guest = GuestCredential(username="encrypted", password="encrypted")
    item = CachedTestItem(site="PTB", project="test", station="test", test_item_name="power")
    session.add_all([config, token, sfis, guest, item])
    session.commit()
    session.expire_all()

    assert config.description is None
    assert token.is_active is True
    assert sfis.timeout == 30.0
    assert guest.is_active is True
    assert item.is_value is False
    assert isinstance(item.updated_at, datetime)


def test_menu_and_product_relationships_roundtrip(session: Session):
    menu = MenuDefinition(menu_key="test", title="Test", path="/test")
    access = MenuRoleAccess(menu=menu, role_name="user")
    product = TopProduct(dut_isn="test", station_name="test", score=9.0)
    measurement = TopProductMeasurement(top_product=product, test_item="power", actual_value=3.0)
    session.add_all([access, measurement])
    session.commit()
    session.expire_all()

    assert menu.icon == "mdi-circle-small"
    assert menu.sort_order == 0
    assert menu.role_access[0].can_view is True
    assert product.pass_count == 0
    assert product.measurements[0].actual_value == 3.0
    assert measurement.top_product is product

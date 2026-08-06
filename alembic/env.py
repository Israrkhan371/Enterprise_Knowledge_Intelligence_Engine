from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# Make `app` importable when alembic is invoked from the repo root (as it
# is in the "alembic upgrade head" calls below and in the Dockerfile).
from app.core.config import settings
from app.core.database import Base

# Import all model modules here so Base.metadata is fully populated before
# autogenerate compares it against the live DB. app.core.models is the only
# module that defines tables today; if a new model module is added, import
# it here too or autogenerate will silently miss its tables.
from app.core import models  # noqa: F401

# this is the Alembic Config object, which provides access to values within
# the .ini file in use.
config = context.config

# Single source of truth for the DB URL: app.core.config.settings, which
# itself reads DATABASE_URL from .env — the same value app/core/database.py
# uses to build the app's own engine. Overriding it here (rather than
# hardcoding sqlalchemy.url in alembic.ini) means alembic always points at
# whatever DB the running app is actually using.
config.set_main_option("sqlalchemy.url", settings.database_url)

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here for 'autogenerate' support
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine, though
    an Engine is acceptable here as well. By skipping the Engine creation
    we don't even need a DBAPI to be available.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine and associate a connection
    with the context.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

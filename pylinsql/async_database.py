from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import logging
import os
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import asyncpg

from pylinsql.core import DEFAULT, is_dataclass_type
from pylinsql.query import insert_or_select, select

T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class ConnectionParameters:
    user: str = None
    password: str = None
    database: str = None
    host: str = None
    port: int = None
    command_timeout: int = 60

    def __post_init__(self):
        # use object.__setattr__ to avoid dataclasses.FrozenInstanceError
        if self.user is None:
            user = os.getenv("PSQL_USERNAME", "postgres")
            object.__setattr__(self, "user", user)
        if self.password is None:
            password = os.getenv("PSQL_PASSWORD", "")
            object.__setattr__(self, "password", password)
        if self.database is None:
            database = os.getenv("PSQL_DATABASE", "postgres")
            object.__setattr__(self, "database", database)
        if self.host is None:
            host = os.getenv("PSQL_HOSTNAME", "localhost")
            object.__setattr__(self, "host", host)
        if self.port is None:
            port = int(os.getenv("PSQL_PORT", "5432"))
            object.__setattr__(self, "port", port)

    def as_dict(self):
        return dataclasses.asdict(self)


class DatabasePool:
    pool: asyncpg.pool.Pool

    def __init__(self, pool):
        self.pool = pool

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[DatabaseConnection]:
        conn = await self.pool.acquire()
        try:
            yield DatabaseConnection(conn)
        finally:
            await self.pool.release(conn)

    async def release(self):
        return await self.pool.close()


async def _create_pool(params: ConnectionParameters) -> asyncpg.Pool:
    return await asyncpg.create_pool(**params.as_dict())


@asynccontextmanager
async def pool(
    params: ConnectionParameters = None,
) -> AsyncIterator[DatabasePool]:
    if params is None:
        params = ConnectionParameters()
    pool = await _create_pool(params)
    try:
        yield DatabasePool(pool)
    finally:
        await pool.close()
        pool.terminate()


class DatabaseClient:
    conn: asyncpg.Connection

    def __init__(self, conn):
        self.conn = conn

    @staticmethod
    def _unwrap(record: asyncpg.Record):
        if len(record) > 1:
            return tuple(record.values())
        elif len(record) > 0:
            return record[0]
        else:
            return None

    async def select_first(
        self, sql_generator_expr: Generator, *args
    ) -> Union[Any, Tuple]:
        "Returns the first row of the resultset produced by a SELECT query."

        query = select(sql_generator_expr)
        stmt = await self.conn.prepare(str(query))
        logging.debug("executing query: %s", query)
        return self._unwrap(await stmt.fetchrow(*args))

    async def select(
        self, sql_generator_expr: Generator, *args
    ) -> List[asyncpg.Record]:
        "Returns all rows of the resultset produced by a SELECT query."

        query = select(sql_generator_expr)
        stmt = await self.conn.prepare(str(query))
        logging.debug("executing query: %s", query)
        return await stmt.fetch(*args)

    async def insert_or_select(
        self, insert_obj: T, sql_generator_expr: Generator, *args
    ) -> Union[Any, Tuple]:
        "Queries the database and inserts a new row if the query returns an empty resultset."

        query = insert_or_select(insert_obj, sql_generator_expr)
        stmt = await self.conn.prepare(str(query))

        # append parameters for object to insert
        fields = dataclasses.fields(insert_obj)
        fetch_args = list(args)
        for field in fields:
            value = getattr(insert_obj, field.name)
            if value is not DEFAULT:
                fetch_args.append(value)

        logging.debug("executing query: %s", query)
        return self._unwrap(await stmt.fetchrow(*fetch_args))

    async def insert_or_ignore(self, insert_obj: T) -> None:
        table_name = type(insert_obj)
        fields = dataclasses.fields(insert_obj)
        column_list = ", ".join(field.name for field in fields)
        placeholder_list = ", ".join(f"${index+1}" for index in range(len(fields)))
        query = f"INSERT INTO {table_name} ({column_list}) VALUES ({placeholder_list}) ON CONFLICT DO NOTHING"

        values = []
        for field in fields:
            value = getattr(insert_obj, field.name)
            if value is not DEFAULT:
                values.append(value)

        logging.debug("executing query: %s", query)
        stmt = await self.conn.prepare(query)
        await stmt.execute(values)

    async def typed_fetch(self, typ: Type, query: str, *args) -> List:
        """Maps all columns of a database record to a Python data class."""

        if not is_dataclass_type(typ):
            raise TypeError(f"{typ} must be a dataclass type")

        records = await self.conn.fetch(query, *args)
        return self._typed_fetch(typ, records)

    async def typed_fetch_column(
        self, typ: Type, query: str, *args, column: int = 0
    ) -> List:
        """Maps a single column of a database record to a Python class."""

        records = await self.conn.fetch(query, *args)
        return [record[column] for record in records]

    def _typed_fetch(self, typ: Type, records: List[asyncpg.Record]) -> List:
        results = []
        for record in records:
            result = object.__new__(typ)

            if is_dataclass_type(typ):
                for field in dataclasses.fields(typ):
                    key = field.name
                    value = record.get(key, None)
                    if value is not None:
                        setattr(result, key, value)
                    elif field.default:
                        setattr(result, key, field.default)
                    else:
                        raise RuntimeError(
                            f"object field {key} without default value is missing a corresponding database record column"
                        )
            else:
                for key, value in record.items():
                    setattr(result, key, value)

            results.append(result)
        return results

    async def raw_fetch(
        self, query: str, *args, timeout=None, record_class=None
    ) -> List[asyncpg.Record]:

        return await self.conn.fetch(
            query, *args, timeout=timeout, record_class=record_class
        )

    async def raw_fetchval(
        self, query: str, *args, column: int = 0, timeout: Optional[float] = None
    ) -> Any:
        return await self.conn.fetchval(query, *args, column=column, timeout=timeout)

    async def raw_execute(
        self, query: str, *args, timeout: Optional[float] = None
    ) -> str:
        return await self.conn.execute(query, *args, timeout=timeout)

    async def raw_executemany(
        self, command: str, args: Iterable, timeout: Optional[float] = None
    ) -> None:
        return await self.conn.executemany(command, args, timeout=timeout)


class DatabaseTransaction(DatabaseClient):
    def __init__(self, conn, transaction):
        super().__init__(conn)
        self.transaction = transaction


class DatabaseConnection(DatabaseClient):
    def __init__(self, conn):
        super().__init__(conn)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[DatabaseTransaction]:
        transaction = self.conn.transaction()
        await transaction.start()
        try:
            yield DatabaseTransaction(self.conn, transaction)
        except:
            await transaction.rollback()
            raise
        else:
            await transaction.commit()


@asynccontextmanager
async def connection(
    params: ConnectionParameters = None,
) -> AsyncIterator[DatabaseConnection]:
    if params is None:
        params = ConnectionParameters()
    conn = await asyncpg.connect(**params.as_dict())
    try:
        yield DatabaseConnection(conn)
    finally:
        await conn.close()


_connection_pools = contextvars.ContextVar("pool")


async def shared_pool(params: ConnectionParameters = None) -> DatabasePool:
    "A database connection pool shared across coroutines in the asynchronous execution context."

    if params is None:
        params = ConnectionParameters()

    pools: Dict[ConnectionParameters, DatabasePool] = _connection_pools.get(None)
    if pools is None:
        pools = {}
        _connection_pools.set(pools)

    pool: DatabasePool = pools.get(params, None)
    if pool is None:
        pool = DatabasePool(await _create_pool(params))
        pools[params] = pool

    return pool


class DataAccess:
    params: ConnectionParameters

    def __init__(self, params: ConnectionParameters = None):
        self.params = ConnectionParameters() if params is None else params

    @asynccontextmanager
    async def get_connection(self) -> AsyncIterator[DatabaseConnection]:
        pool = await shared_pool(self.params)
        async with pool.connection() as connection:
            yield connection

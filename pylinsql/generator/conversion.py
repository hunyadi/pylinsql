import datetime
import decimal
import logging
import re
import typing
import uuid
from dataclasses import dataclass
from ipaddress import IPv4Address, IPv6Address
from typing import Any, List, Optional, Tuple, Type, TypeVar

from strong_typing.auxiliary import (
    Annotated,
    MaxLength,
    Precision,
    SpecialConversion,
    Storage,
    TimePrecision,
    float32,
    float64,
    int16,
    int32,
    int64,
)
from strong_typing.inspection import (
    is_dataclass_type,
    is_generic_list,
    is_type_enum,
    is_type_literal,
    is_type_union,
    unwrap_generic_list,
    unwrap_literal_types,
    unwrap_union_types,
)

T = TypeVar("T")


def cast_if_not_none(typ: Type[T], value: Optional[Any]) -> Optional[T]:
    "Coerces an optional value into the specified type unless the value is None."

    if value is None:
        return None
    else:
        return typ(value)  # type: ignore


def sql_quoted_id(name: str) -> str:
    id = name.replace('"', '""')
    return f'"{id}"'


_sql_quoted_str_table = str.maketrans(
    {
        "\\": "\\\\",
        "'": "\\'",
        "\b": "\\b",
        "\f": "\\f",
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
    }
)


def sql_quoted_str(text: str) -> str:
    if re.search(r"[\b\f\n\r\t]", text):
        string = text.translate(_sql_quoted_str_table)
        return f"E'{string}'"
    else:
        string = text.replace("'", "''")
        return f"'{string}'"


class SqlDataType:
    def _unrecognized_meta(self, meta: Any) -> None:
        logging.warning(
            "unrecognized Python type annotation for %s: %s",
            type(self).__name__,
            repr(meta),
        )


@dataclass
class SqlCharacterType(SqlDataType):
    max_len: Optional[int] = None
    compact: bool = False

    def __init__(self, metadata, compact: bool = False):
        for meta in metadata:
            if isinstance(meta, MaxLength):
                self.max_len = meta.value
            elif isinstance(meta, SpecialConversion):
                pass
            else:
                self._unrecognized_meta(meta)
        self.compact = compact

    def __str__(self) -> str:
        type_name = "varchar" if self.compact else "character varying"
        if self.max_len is not None:
            return f"{type_name}({self.max_len})"
        else:
            return type_name


@dataclass
class SqlDecimalType(SqlDataType):
    precision: Optional[int] = None
    scale: Optional[int] = None

    def __init__(self, metadata):
        for meta in metadata:
            if isinstance(meta, Precision):
                self.precision = meta.significant_digits
                self.scale = meta.decimal_digits
            else:
                self._unrecognized_meta(meta)

    def __str__(self) -> str:
        if self.precision is not None and self.scale is not None:
            return f"decimal({self.precision}, {self.scale})"
        elif self.precision is not None:
            return f"decimal({self.precision})"
        else:
            return "decimal"


@dataclass
class SqlTimestampType(SqlDataType):
    precision: Optional[int] = None

    def __init__(self, metadata):
        for meta in metadata:
            if isinstance(meta, TimePrecision):
                self.precision = meta.decimal_digits
            else:
                self._unrecognized_meta(meta)

    def __str__(self) -> str:
        if self.precision is not None:
            return f"timestamp({self.precision})"
        else:
            return "timestamp"


@dataclass
class SqlTimeType(SqlDataType):
    precision: Optional[int] = None

    def __init__(self, metadata):
        for meta in metadata:
            if isinstance(meta, TimePrecision):
                self.precision = meta.decimal_digits
            else:
                self._unrecognized_meta(meta)

    def __str__(self) -> str:
        if self.precision is not None:
            return f"time({self.precision})"
        else:
            return "time"


def sql_to_python_type(sql_type: str) -> type:
    "Maps a PostgreSQL type to a native Python type."

    if sql_type.endswith("[]"):
        return List[sql_to_python_type(sql_type[:-2])]  # type: ignore

    sql_type = sql_type.lower()

    if sql_type == "boolean":
        return bool
    if sql_type == "smallint":
        return int16
    if sql_type in ["int", "integer"]:
        return int32
    if sql_type == "bigint":
        return int64
    if sql_type == "real":
        return Annotated[float, Storage(4)]
    if sql_type in ["double", "double precision"]:
        return Annotated[float, Storage(8)]
    if sql_type in ["character varying", "text"]:
        return str
    if sql_type in ["decimal", "numeric"]:
        return decimal.Decimal
    if sql_type == "date":
        return datetime.date
    if sql_type in ["time", "time with time zone", "time without time zone"]:
        return datetime.time
    if sql_type == "interval":
        return datetime.timedelta
    if sql_type in [
        "timestamp",
        "timestamp with time zone",
        "timestamp without time zone",
    ]:
        return datetime.datetime
    if sql_type in ["json", "jsonb"]:
        return str
    if sql_type == "uuid":
        return uuid.UUID

    m = re.match(r"^character varying[(](\d+)[)]$", sql_type)
    if m is not None:
        len = int(m.group(1))
        return Annotated[str, MaxLength(len)]

    m = re.match(r"^(?:decimal|numeric)[(](\d+)(?:,\s*(\d+))?[)]$", sql_type)
    if m is not None:
        precision = int(m.group(1))
        scale = int(m.group(2)) if m.group(2) else 0
        return Annotated[decimal.Decimal, Precision(precision, scale)]

    m = re.match(r"^time[(](\d+)[)](?: with(?:out)? time zone)?$", sql_type)
    if m is not None:
        precision = int(m.group(1))
        return Annotated[datetime.time, TimePrecision(precision)]

    m = re.match(r"^timestamp[(](\d+)[)](?: with(?:out)? time zone)?$", sql_type)
    if m is not None:
        precision = int(m.group(1))
        return Annotated[datetime.datetime, TimePrecision(precision)]

    raise NotImplementedError(f"unrecognized database type: {sql_type}")


def python_to_sql_type(typ: type, compact: bool = False, custom: bool = True) -> str:
    "Maps a native Python type to a PostgreSQL type."

    if typ is bool:
        return "boolean"
    if typ is int16:
        return "smallint"
    if typ is int32 or typ is int:
        return "int"
    if typ is int64:
        return "bigint"
    if typ is float32:
        return "real"
    if typ is float64 or typ is float:
        if compact:
            return "double"
        else:
            return "double precision"
    if typ is str:
        return "text"
    if typ is decimal.Decimal:
        return "decimal"
    if typ is datetime.datetime:
        return "timestamp without time zone"
    if typ is datetime.date:
        return "date"
    if typ is datetime.time:
        return "time without time zone"
    if typ is datetime.timedelta:
        return "interval"
    if typ is uuid.UUID:
        return "uuid"
    if typ is IPv4Address or typ is IPv6Address:
        return "inet"

    metadata = getattr(typ, "__metadata__", None)
    if metadata is not None:
        # type is Annotated[T, ...]
        inner_type = typing.get_args(typ)[0]

        if inner_type is str:
            return str(SqlCharacterType(metadata, compact=compact))
        elif inner_type is decimal.Decimal:
            return str(SqlDecimalType(metadata))
        elif inner_type is datetime.datetime:
            return str(SqlTimestampType(metadata))
        elif inner_type is datetime.time:
            return str(SqlTimeType(metadata))
        else:
            logging.warning("cannot map annotated Python type: %s", typ)
            return python_to_sql_type(inner_type, compact=compact)

    if custom:
        if is_type_enum(typ):
            return typ.__name__
        if is_dataclass_type(typ):
            return typ.__name__

    if is_generic_list(typ):
        list_item_type = python_to_sql_type(unwrap_generic_list(typ), compact=compact)
        return f"{list_item_type}[]"
    elif is_type_literal(typ):
        return _get_common_sql_type(unwrap_literal_types(typ), compact=compact)
    elif is_type_union(typ):
        return _get_common_sql_type(unwrap_union_types(typ), compact=compact)

    raise NotImplementedError(f"cannot map Python type: {repr(typ)}")


def _get_common_sql_type(data_types: Tuple[type], compact: bool = False) -> str:
    sql_types = set(python_to_sql_type(t, compact=compact) for t in data_types)
    if len(sql_types) != 1:
        raise NotImplementedError(
            f"cannot map union of incompatible types: {repr(data_types)}"
        )
    return sql_types.pop()

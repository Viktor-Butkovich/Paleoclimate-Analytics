import sqlalchemy

conn_str = "sqlite:///../Data/Temp12k-sqlite.db"  # SQLite connection string

engine: sqlalchemy.Engine = None
conn: sqlalchemy.Connection = None


def table_exists(table_name: str) -> bool:
    inspector = sqlalchemy.inspect(engine)
    return table_name in inspector.get_table_names()


def connect():
    global engine, conn
    engine = sqlalchemy.create_engine(conn_str)
    conn = engine.connect()
    conn.execute(sqlalchemy.text("PRAGMA journal_mode=WAL;"))


def close():
    conn.commit()
    conn.close()
    engine.dispose()


def drop_table(table_name: str):
    conn.execute(sqlalchemy.text(f"DROP TABLE {table_name}"))


def create_table(table_name: str, columns: dict):
    conn.execute(
        sqlalchemy.text(
            f"CREATE TABLE {table_name} ({', '.join([f'{k} {v}' for k, v in columns.items()])})"
        )
    )


def insert(table_name: str, values: dict):
    placeholders = ", ".join(["?" for _ in values.values()])
    conn.execute(
        sqlalchemy.text(
            f"INSERT INTO {table_name} ({', '.join(values.keys())}) VALUES ({placeholders})",
            tuple(values.values()),
        )
    )


def read_table(table_name: str):
    return conn.execute(sqlalchemy.text(f"SELECT * FROM {table_name}")).fetchall()


connect()
print("Connected to SQLite database")
close()

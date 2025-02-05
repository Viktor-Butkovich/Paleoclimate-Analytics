import sqlalchemy

conn_str = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=(LocalDB)\\MSSQLLocalDB;"
    "DATABASE=Temp12k;"
    "Trusted_Connection=yes;"
)  # Use the connection string for your local SQL Server instance

engine: sqlalchemy.Engine = None
conn: sqlalchemy.Connection = None


def table_exists(table_name: str) -> bool:
    result = conn.exec_driver_sql(
        f"SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{table_name}'"
    ).fetchone()
    return result[0] > 0


def connect():
    global engine, conn
    engine = sqlalchemy.create_engine(f"mssql+pyodbc:///?odbc_connect={conn_str}")
    conn = engine.connect()


def close():
    conn.commit()
    conn.close()


def drop_table(table_name: str):
    conn.exec_driver_sql(f"DROP TABLE IF EXISTS {table_name}")


def create_table(table_name: str, columns: dict):
    conn.exec_driver_sql(
        f"CREATE TABLE {table_name} ({', '.join([f'{k} {v}' for k, v in columns.items()])})"
    )


def insert(table_name: str, values: dict):
    conn.exec_driver_sql(
        f"INSERT INTO {table_name} ({', '.join(values.keys())}) VALUES ({', '.join([str(v) for v in values.values()])})"
    )


def read_table(table_name: str):
    return conn.exec_driver_sql(f"SELECT * FROM {table_name}").fetchall()


connect()
server_name = conn.exec_driver_sql("SELECT @@SERVERNAME").fetchone()[0]
print(f"Connected to SQL server: {server_name}")
close()

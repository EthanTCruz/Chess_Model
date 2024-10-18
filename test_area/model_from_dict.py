from sqlalchemy import Column, Integer, String, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# SQLAlchemy base class
Base = declarative_base()

# Example dictionary
data_dict = {
    "id": int,
    "name": str,
    "age": int,
    "salary": float
}

# Function to create an SQLAlchemy model dynamically
def create_model_from_dict(table_name, data_dict):
    columns = {
        '__tablename__': table_name,
        'id': Column(Integer, primary_key=True, autoincrement=True)  # Always have an ID column
    }
    
    # Iterate over the dictionary to add columns
    for key, value_type in data_dict.items():
        if value_type == int:
            columns[key] = Column(Integer)
        elif value_type == str:
            columns[key] = Column(String)
        elif value_type == float:
            columns[key] = Column(Float)
        else:
            raise TypeError(f"Unsupported type: {value_type}")
    
    # Dynamically create the class
    model_class = type(table_name.capitalize(), (Base,), columns)
    
    return model_class

# Create the SQLite engine and session
engine = create_engine('sqlite:///example.db', echo=True)
Session = sessionmaker(bind=engine)
session = Session()

# Create model from dictionary
MyModel = create_model_from_dict('my_table', data_dict)

# Create tables in the database
Base.metadata.create_all(engine)

# Now you can use `MyModel` to interact with the table
new_record = MyModel(id=1, name="John Doe", age=30, salary=50000.00)
session.add(new_record)
session.commit()

# Query the table
records = session.query(MyModel).all()
for record in records:
    print(record.name, record.age, record.salary)

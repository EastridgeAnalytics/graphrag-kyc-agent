CREATE TABLE customers (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255),
    email VARCHAR(255)
);

INSERT INTO customers (id, name, email) VALUES
('1', 'John Doe', 'john.doe@email.com'),
('2', 'Jane Smith', 'jane.smith@email.com');

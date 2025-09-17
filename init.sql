CREATE TABLE customers (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255),
    email VARCHAR(255),
    phone VARCHAR(255),
    address VARCHAR(255),
    risk_score INTEGER
);

CREATE TABLE transactions (
    id VARCHAR(255) PRIMARY KEY,
    from_customer VARCHAR(255),
    to_customer VARCHAR(255),
    amount DECIMAL(15,2),
    transaction_date DATE
);

INSERT INTO customers (id, name, email, phone, address, risk_score) VALUES
('CUST_06595', 'John Smith', 'john.smith@email.com', '555-0123', '123 Main St, NY', 8),
('CUST_01488', 'Sarah Johnson', 'sarah.j@email.com', '555-0456', '456 Oak Ave, CA', 6),
('CUST_01819', 'Michael Chen', 'michael.c@email.com', '555-0789', '789 Pine Rd, TX', 9);

INSERT INTO transactions (id, from_customer, to_customer, amount, transaction_date) VALUES
('TXN_097210', 'CUST_06595', 'CUST_01488', 15000.00, '2025-09-15'),
('TXN_097208', 'CUST_01488', 'CUST_01819', 25000.00, '2025-09-14');
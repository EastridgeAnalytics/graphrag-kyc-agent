import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
import random
import numpy as np
import time
import uuid
from datetime import datetime, timedelta

# reads .env into os.environ
load_dotenv()  

# Seed the RNG—use any constant (e.g. 42)
random.seed(42)

# ———————————————
# 0. Neo4j Aura Connection Setup
# ———————————————
# (Set these in your shell or CI/CD pipeline—never check secrets into Git!)
# Read Neo4j environment variables into variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# 3. (Optional) Debug print — remove or comment out before committing!
print(f"Using URI={NEO4J_URI}")
print(f"Using USER={NEO4J_USER}")
print(f"Using DATABASE={NEO4J_DATABASE}")


# By using the neo4j+s:// scheme, encryption & certificate trust are automatic
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# When you open sessions you may want to specify the default database (usually "neo4j"):
def get_session():
    return driver.session(database=NEO4J_DATABASE)

def clear_database():
    print("⚠️ Clearing the database...")
    with get_session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("✅ Database cleared.")

# Clear the database before generating new data
clear_database()

# ———————————————
# 1. Create uniqueness constraints (once)
# ———————————————
with get_session() as sess:
    
    for label in ('Customer','Account','Company','Address',
                  'Device','IP_Address','Payment_Method','Transaction','Alert','SAR_Draft','PhoneNumber'):
        sess.execute_write(
            lambda tx, L=label: tx.run(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{L}) REQUIRE n.id IS UNIQUE"
            )
        )


# ———————————————
# 2. Configuration & ID lists
# ———————————————
random.seed(42)
np.random.seed(42)

n_customers = 8_000
mean_accounts_per_customer   = 1.5
mean_devices_per_customer    = 2
mean_addresses_per_customer  = 1.2
mean_payment_methods_per_customer = 1
mean_transactions_per_account     = 10
p_pep       = 0.01
p_watchlist = 0.02

# IDs
customers   = [f"CUST_{i:05d}" for i in range(1, n_customers+1)]
n_companies = int(n_customers * 0.2)
companies   = [f"COMP_{i:05d}" for i in range(1, n_companies+1)]


# Prepare payloads
customer_rows = [
    {"id": cust,
     "pep": (random.random() < p_pep),
     "wl":  (random.random() < p_watchlist),
     "name": cust
    }
    for cust in customers
]

company_rows = [
    {"id": comp,
     "ind": random.choice(['Finance','Tech','Manufacturing','Retail']),
     "name": comp
     }
    for comp in companies
]

# ———————————————
# 3. Push Customers & Companies
# ———————————————

print(f"loading start...")
start_time = time.perf_counter()

batch_size=50

with get_session() as sess:
    # Customers in implicit transactions of 50 rows each
    sess.run(
        """
        UNWIND $rows AS row
        MERGE (c:Customer {id: row.id})
        SET c.is_pep       = row.pep,
            c.on_watchlist = row.wl,
            c.name = row.name
        """,
        rows=customer_rows
    )

    # Companies in implicit transactions of 50 rows each
    sess.run(
        """
        UNWIND $rows AS row
        MERGE (c:Company {id: row.id})
        SET c.industry = row.ind,
            c.name = row.name
        """,
        rows=company_rows
    )

    
end_time = time.perf_counter()
elapsed = end_time - start_time
print(f"⌛ Loading Customers & Companies took {elapsed:.2f} seconds")



# ———————————————
# 3. Push Accounts, Addresses, Devices, IP addresses, Payment Methods and Transactions
# ———————————————

# Build payloads
acct_counter = addr_counter = dev_counter = ip_counter = pm_counter = txn_counter = 0

account_rows     = []
employed_rows    = []
address_rows     = []
device_rows      = []
ip_rows          = []
payment_rows     = []
transaction_rows = []


# 1.1 Accounts & OWNS
for cust in customers:
    for _ in range(np.random.poisson(mean_accounts_per_customer)):
        acct_counter += 1
        aid = f"ACCT_{acct_counter:05d}"
        account_rows.append({"cust": cust, "acct": aid,"name":aid})

# 1.2 EMPLOYED_BY
for cust in customers:
    if random.random() < 0.8:
        comp = random.choice(companies)
        employed_rows.append({"cust": cust, "co": comp})

# 1.3 Addresses & LIVES_AT
for cust in customers:
    for _ in range(max(1, np.random.poisson(mean_addresses_per_customer))):
        addr_counter += 1
        aid = f"ADDR_{addr_counter:05d}"
        city = random.choice(['London','Manchester','Birmingham','Leeds'])
        address_rows.append({"cust": cust, "addr": aid, "city": city,"name":aid, "country": "UK"})

# 1.4 Devices & USES_DEVICE → ASSOCIATED_WITH IP_Address
for cust in customers:
    for _ in range(np.random.poisson(mean_devices_per_customer)):
        dev_counter += 1
        did = f"DEV_{dev_counter:05d}"
        osys = random.choice(['Android','iOS','Windows','MacOS'])
        device_rows.append({"cust": cust, "dev": did, "os": osys,"name":did})

        ip_counter += 1
        iid = f"IP_{ip_counter:05d}"
        ip_rows.append({"dev": did, "ip": iid,"name":iid})

# 1.5 Payment Methods & HAS_METHOD
for cust in customers:
    for _ in range(np.random.poisson(mean_payment_methods_per_customer)):
        pm_counter += 1
        pid = f"PM_{pm_counter:05d}"
        ptype = random.choice(['Credit_Card','Debit_Card','EWallet'])
        cnum = ''.join(random.choice('0123456789') for _ in range(16)) \
               if ptype in ('Credit_Card','Debit_Card') \
               else uuid.uuid4().hex[:16]
        payment_rows.append({
            "cust": cust,
            "pid": pid,
            "ptype": ptype,
            "cnum": cnum,
            "name": pid
        })


# 1.6 Transactions & FROM/TO
all_accts = [r["acct"] for r in account_rows]
for src in all_accts:
    for _ in range(np.random.poisson(mean_transactions_per_account)):
        txn_counter += 1
        tid = f"TXN_{txn_counter:06d}"
        amt = round(np.random.lognormal(mean=3, sigma=1), 2)
        ts  = (datetime(2025,1,1) + timedelta(days=random.randint(0,120))).isoformat()
        dst = random.choice(all_accts)
        transaction_rows.append({
            "src": src, "tid": tid, "amt": amt, "ts": ts, "dst": dst, "name":tid
        })


# 2. Push in batches
start_time = time.perf_counter()
with get_session() as sess:
    # 2.1 Accounts
    sess.run(
        """
        UNWIND $rows AS row
        MERGE (a:Account {id: row.acct})
        SET a.name = row.name
        WITH a, row
        MATCH (c:Customer {id: row.cust})
        MERGE (c)-[:OWNS]->(a)
        """,
        rows=account_rows    )

    # 2.2 Employed
    sess.run(
        """
        UNWIND $rows AS row
        MATCH (c:Customer {id: row.cust})
        MATCH (co:Company  {id: row.co})
        MERGE (c)-[:EMPLOYED_BY]->(co)
        """,
        rows=employed_rows    )

    # 2.3 Addresses
    sess.run(
        """
        UNWIND $rows AS row
        MERGE (a:Address {id: row.addr})
        SET a.city = row.city,
              a.name = row.name,
              a.country = row.country
        WITH a, row
        MATCH (c:Customer {id: row.cust})
        MERGE (c)-[:LIVES_AT]->(a)
        """,
        rows=address_rows    )

    # 2.4 Devices
    sess.run(
        """
        UNWIND $rows AS row
        MERGE (d:Device {id: row.dev})
        SET d.os = row.os,
            d.name = row.name
        WITH d, row
        MATCH (c:Customer {id: row.cust})
        MERGE (c)-[:USES_DEVICE]->(d)
        """,
        rows=device_rows    )
    # 2.5 IPs
    sess.run(
        """
        UNWIND $rows AS row
        MERGE (i:IP_Address {id: row.ip})
        SET i.name = row.name
        WITH i, row
        MATCH (d:Device {id: row.dev})
        MERGE (d)-[:ASSOCIATED_WITH]->(i)
        """,
        rows=ip_rows    )

    # 2.6 Payment Methods
    sess.run(
        """
        UNWIND $rows AS row
        MERGE (p:Payment_Method {id: row.pid})
        SET p.pm_type     = row.ptype,
              p.card_number = row.cnum,
              p.name = row.name
        WITH p, row
        MATCH (c:Customer {id: row.cust})
        MERGE (c)-[:HAS_METHOD]->(p)
        """,
        rows=payment_rows    )

    # 2.7 Transactions
    sess.run(
        """
        UNWIND $rows AS row
        MERGE (t:Transaction {id: row.tid})
        SET t.amount    = row.amt,
              t.timestamp = row.ts,
              t.name = row.name
        WITH t, row
        MATCH (a1:Account {id: row.src})
        MATCH (a2:Account {id: row.dst})
        MERGE (a1)-[:FROM]->(t)-[:TO]->(a2)
        """,
        rows=transaction_rows    )

end_time = time.perf_counter()
elapsed = end_time - start_time
print(f"⌛ Loading Account, Employed, Owns, Addresses, Devices, Payment Methods & Transactions took {elapsed:.2f} seconds")


# 5. Select anomalies
n_anomalies        = int(0.05 * len(customers))
anoms             = random.sample(customers, n_anomalies)
chunk             = n_anomalies // 5

# Prepare payload lists
super_rows        = []
ring_acct_rows    = []
ring_txn_rows     = []
bridge_rows       = []
isolate_rows      = []
dense_addr_rows   = []
dense_pm_rows     = []


# Super-hubs: 50 new accounts per customer
for cust in anoms[0:chunk]:
    for _ in range(50):
        acct_counter += 1
        aid = f"ACCT_{acct_counter:05d}"
        super_rows.append({"cust": cust, "acct": aid,"name":aid})



#Upload
with get_session() as sess:
    # 4.1 Super-hubs
    start_time = time.perf_counter()
    sess.run(
        """
        UNWIND $rows AS row
        MERGE (a:Account {id: row.acct})
        SET a.name = row.name
        WITH a, row
        MATCH (c:Customer {id: row.cust})
        MERGE (c)-[:OWNS]->(a)
        """,
        rows=super_rows, batch_size=50
    )
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"⌛ Loading Anomalies: Super Hubs took {elapsed:.2f} seconds")

# 2.2 Circular rings: 3-customer cycles
for i in range(chunk, 2*chunk, 3):
    trio = anoms[i : i+3]
    if len(trio) == 3:
        accts = []
        for c in trio:
            acct_counter += 1
            aid = f"ACCT_{acct_counter:05d}"
            ring_acct_rows.append({"cust": c, "acct": aid})
            accts.append(aid)
        for j in range(3):
            txn_counter += 1
            tid = f"TXN_{txn_counter:06d}"
            ring_txn_rows.append({
                "src":      accts[j],
                "dst":      accts[(j+1) % 3],
                "tid":      tid,
                "amount":   1000,
                "ts":       datetime(2025, 2, 1).isoformat()
            })

with get_session() as sess:
    # 4.2 Circular rings – ring transactions
    start_time = time.perf_counter()
    sess.run(
        """
        UNWIND $rows AS row
        MERGE (t:Transaction {id: row.tid})
        SET t.amount = row.amount, t.timestamp = row.ts
        WITH t, row
        MATCH (a1:Account {id: row.src}), (a2:Account {id: row.dst})
        MERGE (a1)-[:FROM]->(t)-[:TO]->(a2)
        """,
        rows=ring_txn_rows, batch_size=50
    )
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"⌛ Loading Anomalies: Circular Rings took {elapsed:.2f} seconds")


# 2.3 Bridges: employed by two companies
for cust in anoms[2*chunk : 3*chunk]:
    c1, c2 = random.sample(companies, 2)
    bridge_rows.append({"cust": cust, "co": c1})
    bridge_rows.append({"cust": cust, "co": c2})
with get_session() as sess:
    # 4.3 Bridges
    start_time = time.perf_counter()
    sess.run(
        """
        UNWIND $rows AS row
        MATCH (c:Customer {id: row.cust}), (co:Company {id: row.co})
        MERGE (c)-[:EMPLOYED_BY]->(co)
        """,
        rows=bridge_rows, batch_size=50
    )
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"⌛ Loading Anomalies: Bridges - Customers employeed by 2 companies took {elapsed:.2f} seconds")

#  Isolates: 5 device→IP pairs per customer, no link to customers
for cust in anoms[3*chunk : 4*chunk]:
    for _ in range(5):
        dev_counter += 1
        ip_counter  += 1
        isolate_rows.append({
            "dev": f"DEV_{dev_counter:05d}",
            "ip":  f"IP_{ip_counter:05d}"
        })

with get_session() as sess:
    # 4.4 Isolates
    start_time = time.perf_counter()
    sess.run(
        """
        UNWIND $rows AS row
        MERGE (d:Device {id: row.dev})
        SET d.os = 'Unknown'
        MERGE (i:IP_Address {id: row.ip})
        MERGE (d)-[:ASSOCIATED_WITH]->(i)
        """,
        rows=isolate_rows, batch_size=50
    )
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"⌛ Loading Anomalies: Isolated Devices and IP Addresses with no Customers took {elapsed:.2f} seconds")

# Dense watchlist cluster: shared address & payment method
shared_addr = f"ADDR_{addr_counter+1:05d}"
shared_pm   = f"PM_{pm_counter+1:05d}"
dense_addr_rows = [{"cust": cust, "addr": shared_addr}
                   for cust in anoms[4*chunk : ]]
dense_pm_rows   = [{"cust": cust, "pm":   shared_pm}
                   for cust in anoms[4*chunk : ]]

# 3. Create the two shared nodes up front
with get_session() as sess:
    sess.run(
        "MERGE (a:Address {id:$addr}) SET a.city='London', a.name=$addr",
        addr=shared_addr
    )
    sess.run(
        "MERGE (p:Payment_Method {id:$pm}) SET p.pm_type='Credit_Card', p.name=$pm",
        pm=shared_pm
    )
with get_session() as sess:
    start_time = time.perf_counter()
    # 4.5 Dense cluster – shared address
    sess.run(
        """
        UNWIND $rows AS row
        MATCH (c:Customer {id: row.cust}), (a:Address {id: row.addr})
        MERGE (c)-[:LIVES_AT]->(a)
        """,
        rows=dense_addr_rows, batch_size=50
    )
    # 4.5 Dense cluster – shared payment method + watchlist flag
    sess.run(
        """
        UNWIND $rows AS row
        MATCH (c:Customer {id: row.cust}), (p:Payment_Method {id: row.pm})
        MERGE (c)-[:HAS_METHOD]->(p)
        SET c.on_watchlist = true
        """,
        rows=dense_pm_rows, batch_size=50
    )
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"⌛ Loading Anomalies: Dense clusters - Around shared address & payment method took {elapsed:.2f} seconds")


# ———————————————
# 5.5 Generate Velocity Attack Data
# ———————————————
print("Generating velocity attack data...")
start_time = time.perf_counter()

# A single phone number used for all new accounts
shared_phone_number = "PHONE_VELOCITY_1"
# A single address for all new accounts
shared_address_velocity = "ADDR_VELOCITY_1"

# 100 new customers and accounts
n_velocity_customers = 100
velocity_customers_rows = []
velocity_accounts_rows = []
velocity_phone_rows = []
velocity_address_rows = []

for i in range(n_velocity_customers):
    cust_id = f"CUST_V_{i:03d}"
    acct_id = f"ACCT_V_{i:03d}"
    
    velocity_customers_rows.append({
        "id": cust_id,
        "pep": False,
        "wl": True, # on watchlist
        "name": cust_id
    })
    
    velocity_accounts_rows.append({
        "cust": cust_id,
        "acct": acct_id,
        "name": acct_id
    })
    
    velocity_phone_rows.append({
        "cust": cust_id,
        "phone": shared_phone_number
    })

    velocity_address_rows.append({
        "cust": cust_id,
        "addr": shared_address_velocity
    })

with get_session() as sess:
    # Create velocity customers
    sess.run(
        """
        UNWIND $rows AS row
        MERGE (c:Customer {id: row.id})
        SET c.is_pep       = row.pep,
              c.on_watchlist = row.wl,
              c.name = row.name
        """,
        rows=velocity_customers_rows,
    )

    # Create velocity accounts and link to customers
    sess.run(
        """
        UNWIND $rows AS row
        MERGE (a:Account {id: row.acct})
        SET a.name = row.name
        WITH a, row
        MATCH (c:Customer {id: row.cust})
        MERGE (c)-[:OWNS]->(a)
        """,
        rows=velocity_accounts_rows    )
    
    # Create phone number and link to customers
    sess.run("MERGE (p:PhoneNumber {id: $phone_id, name: $phone_id})", phone_id=shared_phone_number)
    
    sess.run(
        """
        UNWIND $rows AS row
        MATCH (c:Customer {id: row.cust})
        MATCH (p:PhoneNumber {id: row.phone})
        MERGE (c)-[:HAS_PHONE]->(p)
        """,
        rows=velocity_phone_rows    )
    
    # Create address and link to customers
    sess.run("MERGE (a:Address {id: $addr_id, name: $addr_id, city: 'London', country: 'UK'})", addr_id=shared_address_velocity)
    
    sess.run(
        """
        UNWIND $rows AS row
        MATCH (c:Customer {id: row.cust})
        MATCH (a:Address {id: row.addr})
        MERGE (c)-[:LIVES_AT]->(a)
        """,
        rows=velocity_address_rows
    )

end_time = time.perf_counter()
elapsed = end_time - start_time
print(f"⌛ Loading Velocity Attack Data took {elapsed:.2f} seconds")


# ———————————————
# 5.6 Generate Co-opted Clean Skin Velocity Data (Device)
# ———————————————
def generate_device_velocity_attack():
    print("Generating co-opted clean skin velocity data (Device)...")
    start_time = time.perf_counter()

    with get_session() as session:
        # 1. Select 50 random existing customers not on a watchlist
        result = session.run("""
            MATCH (c:Customer)
            WHERE c.on_watchlist = false AND NOT c.id STARTS WITH 'CUST_V_'
            RETURN c.id AS customerId
            LIMIT 50
        """)
        customer_ids = [record["customerId"] for record in result]

        if len(customer_ids) < 50:
            print("⚠️  Could not find enough non-watchlist customers to generate device velocity alert.")
            return

        # 2. Create a shared device and link it to these customers
        shared_device_id = f"DEVICE_VELOCITY_{uuid.uuid4().hex[:4].upper()}"
        session.run("MERGE (d:Device {id: $id})", id=shared_device_id)

        for cust_id in customer_ids:
            session.run("""
                MATCH (c:Customer {id: $cust_id})
                MATCH (d:Device {id: $dev_id})
                MERGE (c)-[:USES_DEVICE]->(d)
            """, cust_id=cust_id, dev_id=shared_device_id)

        # 3. Create an alert for this pattern and link to all involved customers
        alert_id = f"ALERT_DEV_VEL_{uuid.uuid4().hex[:4].upper()}"
        description = f"Device Velocity Detected: {len(customer_ids)} established customers suddenly began using a new shared device ({shared_device_id})."
        
        session.run("""
            CREATE (a:Alert {
                id: $alert_id,
                description: $description,
                timestamp: $timestamp,
                status: 'new',
                latitude: $latitude,
                longitude: $longitude,
                related_entity_id: $shared_device_id
            })
        """, 
        alert_id=alert_id, 
        description=description, 
        timestamp=datetime.now().isoformat(),
        latitude=51.5074 + random.uniform(-0.05, 0.05),
        longitude=-0.1278 + random.uniform(-0.05, 0.05),
        shared_device_id=shared_device_id
        )

        for cust_id in customer_ids:
            session.run("""
                MATCH (c:Customer {id: $cust_id})
                MATCH (a:Alert {id: $alert_id})
                MERGE (c)-[:HAS_ALERT]->(a)
            """, cust_id=cust_id, alert_id=alert_id)
        
        print(f"  ✅ Created Device Velocity Alert: {alert_id} involving {len(customer_ids)} customers.")

    elapsed = time.perf_counter() - start_time
    print(f"⌛ Loading Device Velocity Attack Data took {elapsed:.2f} seconds")


# ———————————————
# 5.7 Generate High-Risk Jurisdiction Velocity Data (IP Address)
# ———————————————
def generate_ip_velocity_attack():
    print("Generating high-risk jurisdiction velocity data (IP Address)...")
    start_time = time.perf_counter()
    
    with get_session() as session:
        # 1. Create a high-risk IP address
        high_risk_ip_id = f"IP_HIGH_RISK_{uuid.uuid4().hex[:4].upper()}"
        session.run("MERGE (ip:IP_Address {id: $id, country: 'High-Risk-Jurisdiction'})", id=high_risk_ip_id)

        # 2. Create 75 new synthetic customers
        n_ip_customers = 75
        ip_customer_ids = [f"CUST_IP_V_{i:03d}" for i in range(n_ip_customers)]
        
        for cust_id in ip_customer_ids:
            session.run("""
                MERGE (c:Customer {id: $cust_id, name: $cust_id, on_watchlist: true})
                WITH c
                MATCH (ip:IP_Address {id: $ip_id})
                MERGE (c)-[:HAS_IP]->(ip) 
            """, cust_id=cust_id, ip_id=high_risk_ip_id)
            
        # 3. Create an alert and link to all involved customers
        alert_id = f"ALERT_IP_VEL_{uuid.uuid4().hex[:4].upper()}"
        description = f"High-Risk IP Velocity Detected: {n_ip_customers} new customers appeared from a high-risk IP ({high_risk_ip_id})."
        
        session.run("""
            CREATE (a:Alert {
                id: $alert_id,
                description: $description,
                timestamp: $timestamp,
                status: 'new',
                latitude: $latitude,
                longitude: $longitude,
                related_entity_id: $high_risk_ip_id
            })
        """, 
        alert_id=alert_id, 
        description=description, 
        timestamp=datetime.now().isoformat(),
        latitude=51.5074 + random.uniform(-0.05, 0.05),
        longitude=-0.1278 + random.uniform(-0.05, 0.05),
        high_risk_ip_id=high_risk_ip_id
        )

        for cust_id in ip_customer_ids:
            session.run("""
                MATCH (c:Customer {id: $cust_id})
                MATCH (a:Alert {id: $alert_id})
                MERGE (c)-[:HAS_ALERT]->(a)
            """, cust_id=cust_id, alert_id=alert_id)
        
        print(f"  ✅ Created High-Risk IP Velocity Alert: {alert_id} involving {n_ip_customers} customers.")
        
    elapsed = time.perf_counter() - start_time
    print(f"⌛ Loading High-Risk IP Velocity Data took {elapsed:.2f} seconds")


# 6. Generate Alerts
def detect_and_create_velocity_alerts():
    print("Detecting velocity patterns and creating alerts...")
    with get_session() as session:
        result = session.run("""
          MATCH (p:PhoneNumber)<-[r:HAS_PHONE]-(c:Customer)
          WITH p, count(c) as customer_count, collect(c.id) as customer_ids
            WHERE customer_count > 50
            OPTIONAL MATCH (alert:Alert {related_entity_id: p.id})
            WHERE alert IS NULL
            RETURN p.id as phone_id, customer_count, customer_ids
        """)
        
        records = result.data()
        for record in records:
            phone_id = record['phone_id']
            customer_count = record['customer_count']
            customer_ids = record['customer_ids']
            
            alert_id = f"ALERT_VEL_{uuid.uuid4().hex[:4].upper()}"
            description = f"Velocity attack detected: phone number {phone_id} linked to {customer_count} customers."
            
            # Create the alert node once
            session.run("""
                CREATE (a:Alert {
                    id: $alert_id,
                    description: $description,
                    timestamp: $timestamp,
                    latitude: $latitude,
                    longitude: $longitude,
                    status: 'new',
                    related_entity_id: $phone_id
                })
            """, alert_id=alert_id, description=description, 
                 timestamp=datetime.now().isoformat(),
                 latitude=51.5074 + random.uniform(-0.05, 0.05),
                 longitude=-0.1278 + random.uniform(-0.05, 0.05),
                 phone_id=phone_id)

            # Link the alert to all customers involved
            for customer_id in customer_ids:
                session.run("""
                    MATCH (c:Customer {id: $customer_id})
                    MATCH (a:Alert {id: $alert_id})
                    MERGE (c)-[:HAS_ALERT]->(a)
                """, customer_id=customer_id, alert_id=alert_id)

            print(f"  Created alert {alert_id} for phone {phone_id} involving {len(customer_ids)} customers.")

n_alerts = 100
alert_rows = []
alert_counter = 0

# London coordinates
london_lat = 51.5074
london_lon = -0.1278

alertable_customers = random.sample(customer_rows, n_alerts)

for customer in alertable_customers:
    alert_counter += 1
    aid = f"ALERT_{alert_counter:04d}"
    alert_rows.append({
        "cust": customer["id"],
        "alert_id": aid,
        "description": f"Suspicious login detected for customer {customer['id']}",
        "timestamp": (datetime(2025,1,1) + timedelta(days=random.randint(0,120))).isoformat(),
        "latitude": london_lat + random.uniform(-0.05, 0.05),
        "longitude": london_lon + random.uniform(-0.05, 0.05),
        "status": "new"
    })

with get_session() as sess:
    start_time = time.perf_counter()
    sess.run(
        """
        UNWIND $rows AS row
        MERGE (a:Alert {id: row.alert_id})
        SET a.description = row.description,
              a.timestamp = row.timestamp,
              a.latitude = row.latitude,
              a.longitude = row.longitude,
              a.status = row.status
        WITH a, row
        MATCH (c:Customer {id: row.cust})
        MERGE (c)-[:HAS_ALERT]->(a)
        """,
        rows=alert_rows, batch_size=50
    )
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"⌛ Loading Alerts took {elapsed:.2f} seconds")

detect_and_create_velocity_alerts()
generate_device_velocity_attack()
generate_ip_velocity_attack()

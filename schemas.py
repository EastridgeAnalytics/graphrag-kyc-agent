# schemas.py
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any

# Tool 1: Get Customer and Accounts
class CustomerAccountsInput(BaseModel):
    customer_id: str
    model_config = ConfigDict(extra='forbid')

class TransactionModel(BaseModel):
    id: Optional[str] = None
    amount: Optional[float] = None
    timestamp: Optional[str] = None
    model_config = ConfigDict(extra='forbid')

class AccountModel(BaseModel):
    id: str = None
    name: str = None
    transactions: List[TransactionModel] = []
    model_config = ConfigDict(extra='forbid')

class CustomerModel(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    on_watchlist: Optional[bool] = False
    is_pep: Optional[bool] = False
    model_config = ConfigDict(extra='forbid')

class CustomerAccountsOutput(BaseModel):
    customer: CustomerModel
    accounts: List[AccountModel]
    model_config = ConfigDict(extra='forbid')

# Tool 2: Identify watchlisted customers in suspicious rings

class RingModel(BaseModel):
    ring_path: List[Dict[str, Any]]  # List of node dicts
    watched_customers: List[Dict[str, Any]]  # List of customer dicts
    watch_relationships: List[Dict[str, Any]]  # List of relationship dicts
    model_config = ConfigDict(extra='forbid')

class CustomerRingsInput(BaseModel):
    max_number_rings: int = 10
    customer_in_watchlist: Optional[bool] = True
    customer_is_pep: Optional[bool] = False
    model_config = ConfigDict(extra='forbid')

class CustomerRingsOutput(BaseModel):
    customer_rings: List[RingModel]
    model_config = ConfigDict(extra='forbid')

# New schemas for kyc_cypher_tools.py
class CustomerInfo(BaseModel):
    id: str
    name: str
    on_watchlist: bool
    is_pep: bool
    model_config = ConfigDict(extra='forbid')

class CustomerInfoOutput(BaseModel):
    customer: Optional[CustomerInfo] = None
    account_names: List[str]
    model_config = ConfigDict(extra='forbid')

class RingCustomer(BaseModel):
    customer_name: str
    customer_id: str
    customer_on_watchlist: bool
    customer_politically_exposed: bool
    customer_accounts_in_ring: List[str]
    model_config = ConfigDict(extra='forbid')

class CustomerBridgeOutput(BaseModel):
    customer_id: str
    customer_name: str
    on_watchlist: bool
    is_pep: bool
    employer_names: List[str]
    model_config = ConfigDict(extra='forbid')

class HotPropertyOutput(BaseModel):
    address: str
    city: str
    num_other_customers: int
    customer_name: str
    customer_on_watchlist: bool
    customer_is_pep: bool
    model_config = ConfigDict(extra='forbid')

class SharedPII(BaseModel):
    customer_id: str
    customer_name: str
    shared_pii_type: str
    shared_pii_value: str
    model_config = ConfigDict(extra='forbid')

class SharedPIIOutput(BaseModel):
    shared_pii: List[SharedPII]
    model_config = ConfigDict(extra='forbid')

class GenerateCypherRequest(BaseModel):
    question: str = Field(..., description="The natural language question to generate a Cypher query for")
    model_config = ConfigDict(extra='forbid')


# Alert Model
class AlertModel(BaseModel):
    id: str
    description: str
    timestamp: str
    latitude: float
    longitude: float
    status: str

# SAR Draft Model
class SARDraftModel(BaseModel):
    id: str
    narrative: str
    created_at: str
    status: str
    analyst_commentary: str

# Centralized Input Schemas for Tools
class UpdateAlertStatusInput(BaseModel):
    alert_id: str = Field(..., description="The ID of the alert to update.")
    status: str = Field(..., description="The new status for the alert (e.g., 'under_review', 'closed', 'forwarded').")
    model_config = ConfigDict(extra='forbid')

class LoadSqlCustomerToNeo4jInput(BaseModel):
    customer_name: str = Field(..., description="The name of the customer to load from the SQL database")
    model_config = ConfigDict(extra='forbid')

class GetTransactionsForAlertInput(BaseModel):
    alert_id: str = Field(..., description="The ID of the alert to retrieve transactions for.")
    model_config = ConfigDict(extra='forbid')

class SearchSqlCustomerInput(BaseModel):
    name: str = Field(..., description="The name of the customer to search for in the SQL database.")
    model_config = ConfigDict(extra='forbid')

class IngestSqlCustomerInput(BaseModel):
    name: str = Field(..., description="The name of the customer to search in the SQL DB and ingest into Neo4j.")
    model_config = ConfigDict(extra='forbid')

class SearchSqlCustomerByRiskInput(BaseModel):
    risk_score: int = Field(..., description="The minimum risk score to search for in the SQL database.")
    model_config = ConfigDict(extra='forbid')


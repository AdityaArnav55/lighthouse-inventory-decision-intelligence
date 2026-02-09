# Design Document: Lighthouse Inventory Intelligence

## Overview

Lighthouse is an AI-powered inventory decision intelligence platform built as a cloud-native, scalable MVP. The system uses machine learning for demand forecasting, statistical analysis for risk detection, and optimization algorithms for generating actionable recommendations. The architecture prioritizes decision intelligence over visualization, with an AI copilot interface for natural interaction.

### Key Design Principles

1. **Cloud-Native Architecture**: Stateless services, horizontal scalability, managed cloud services
2. **Event-Driven Processing**: Asynchronous data processing for predictions and risk detection
3. **API-First Design**: All functionality exposed through RESTful APIs
4. **Modular Components**: Independent services for prediction, risk detection, and recommendations
5. **Hackathon MVP Focus**: Core intelligence features with minimal UI, rapid deployment capability

### Technology Stack Recommendations

- **Cloud Platform**: AWS (Lambda, S3, RDS Aurora, SQS, API Gateway) or equivalent Azure/GCP services
- **ML Framework**: Python with scikit-learn, statsmodels, Prophet for time series forecasting
- **API Layer**: FastAPI or Flask for REST APIs
- **Database**: PostgreSQL (Aurora) for transactional data, TimescaleDB extension for time-series
- **Caching**: Redis for frequently accessed predictions and risk scores
- **AI Copilot**: OpenAI GPT-4 or Anthropic Claude with function calling for structured queries

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Gateway                              │
│                    (Authentication, Rate Limiting)               │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├──────────────┬──────────────┬──────────────┬────────
             │              │              │              │
        ┌────▼────┐    ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
        │   AI    │    │ Demand  │   │  Risk   │   │  Recom  │
        │ Copilot │    │Predictor│   │Detector │   │ Engine  │
        │ Service │    │ Service │   │ Service │   │ Service │
        └────┬────┘    └────┬────┘   └────┬────┘   └────┬────┘
             │              │              │              │
             └──────────────┴──────────────┴──────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
               ┌────▼────┐    ┌────▼────┐   ┌────▼────┐
               │PostgreSQL│    │  Redis  │   │   S3    │
               │   (RDS)  │    │ (Cache) │   │(Storage)│
               └──────────┘    └─────────┘   └─────────┘
                    │
               ┌────▼────┐
               │   SQS   │
               │ (Queue) │
               └─────────┘
```

### Component Interaction Flow

1. **Data Ingestion**: CSV/JSON uploaded → S3 → SQS event → Processing Lambda → PostgreSQL
2. **Demand Prediction**: Scheduled job → Demand Predictor → Cache results → PostgreSQL
3. **Risk Detection**: Scheduled job → Risk Detector → Cache results → PostgreSQL
4. **Recommendations**: On-demand or scheduled → Recommendation Engine → Cache → PostgreSQL
5. **AI Copilot**: User query → AI Copilot → Function calls to services → Structured response

## Components and Interfaces

### 1. Data Ingestion Service

**Responsibility**: Validate, transform, and store uploaded inventory and sales data.

**Interfaces**:
```python
POST /api/v1/data/inventory
POST /api/v1/data/sales
POST /api/v1/data/products
POST /api/v1/data/batches
```

**Key Functions**:
- `validate_inventory_data(data: dict) -> ValidationResult`
- `validate_sales_data(data: dict) -> ValidationResult`
- `transform_and_store(data: dict, data_type: str) -> StorageResult`
- `trigger_prediction_update(sku_list: List[str]) -> None`

**Data Validation Rules**:
- Inventory: Required fields (sku, location, quantity, timestamp)
- Sales: Required fields (sku, location, quantity, timestamp, price)
- Products: Required fields (sku, name, category, supplier, lead_time_days)
- Batches: Required fields (sku, batch_number, quantity, expiry_date) when expiry tracking enabled

### 2. Demand Predictor Service

**Responsibility**: Generate demand forecasts using time series analysis and ML models.

**Interfaces**:
```python
GET /api/v1/predictions/demand/{sku}
POST /api/v1/predictions/demand/batch
POST /api/v1/predictions/refresh
```

**Core Algorithm**:

The demand predictor uses a hybrid approach combining multiple forecasting methods:

1. **Seasonal Decomposition**:
   - Decompose historical sales into trend, seasonal, and residual components
   - Use STL (Seasonal and Trend decomposition using Loess) for robust decomposition
   - Extract seasonal patterns (weekly, monthly, yearly)

2. **Base Forecast Model**:
   - Use Prophet (Facebook's time series forecasting library) as primary model
   - Prophet handles seasonality, holidays, and trend changes automatically
   - Fallback to SARIMA (Seasonal ARIMA) for SKUs with insufficient data

3. **Feature Engineering**:
   - Seasonality indicators (day of week, month, quarter)
   - Regional demand multipliers (location-based adjustment factors)
   - Promotional flags (binary indicator for promotion periods)
   - Inventory velocity (rolling 7-day, 30-day average sales rate)
   - Lag features (sales from previous 7, 14, 30 days)

4. **Ensemble Prediction**:
   - Combine Prophet forecast with velocity-based forecast
   - Weight: 70% Prophet, 30% velocity-based for stable SKUs
   - Weight: 50% Prophet, 50% velocity-based for volatile SKUs
   - Volatility measured by coefficient of variation in historical sales

5. **Confidence Intervals**:
   - Generate 80% and 95% prediction intervals from Prophet
   - Adjust intervals based on historical forecast accuracy (MAPE)
   - Return lower bound, mean, upper bound for each forecast horizon

**Key Functions**:
```python
def predict_demand(
    sku: str,
    location: str,
    horizon_days: int = 30
) -> DemandForecast:
    """
    Generate demand forecast for a SKU at a location.
    
    Returns:
        DemandForecast with fields:
        - forecast_date: date
        - predicted_quantity: float
        - lower_bound: float (80% CI)
        - upper_bound: float (80% CI)
        - confidence_score: float (0-1)
        - factors: dict (seasonality, trend, velocity)
    """
    pass

def incorporate_seasonality(
    historical_sales: pd.DataFrame,
    sku: str
) -> SeasonalityFactors:
    """Extract and quantify seasonal patterns."""
    pass

def incorporate_regional_behavior(
    sku: str,
    location: str
) -> RegionalMultiplier:
    """Calculate location-specific demand adjustment."""
    pass

def incorporate_promotions(
    sku: str,
    forecast_period: DateRange
) -> PromotionalImpact:
    """Adjust forecast for known promotional periods."""
    pass

def calculate_velocity(
    sku: str,
    location: str,
    window_days: int = 30
) -> float:
    """Calculate current inventory velocity (units/day)."""
    pass
```

**Model Training and Updates**:
- Retrain models weekly using rolling 12-month window
- Incremental updates when new data ingested (within 1 hour)
- Store model artifacts in S3, metadata in PostgreSQL
- Track forecast accuracy (MAPE, RMSE) for model performance monitoring

### 3. Risk Detector Service

**Responsibility**: Identify stockout, overstock, and expiry risks using statistical analysis.

**Interfaces**:
```python
GET /api/v1/risks/stockout/{sku}
GET /api/v1/risks/overstock/{sku}
GET /api/v1/risks/expiry/{sku}
GET /api/v1/risks/summary
POST /api/v1/risks/refresh
```

**Risk Calculation Algorithms**:

#### Stockout Risk

```python
def calculate_stockout_risk(
    sku: str,
    location: str,
    horizon_days: int
) -> StockoutRisk:
    """
    Calculate probability of stockout within horizon.
    
    Algorithm:
    1. Get current stock level (Q_current)
    2. Get demand forecast for horizon (D_forecast, D_std)
    3. Get pending orders and delivery dates (Q_pending)
    4. Calculate available stock: Q_available = Q_current + Q_pending
    5. Calculate demand distribution: N(D_forecast, D_std)
    6. Calculate stockout probability: P(D > Q_available)
    7. Use normal CDF: P = 1 - Φ((Q_available - D_forecast) / D_std)
    
    Risk Levels:
    - High: P > 0.70
    - Medium: 0.40 < P <= 0.70
    - Low: P <= 0.40
    """
    pass
```

#### Overstock Risk

```python
def calculate_overstock_risk(
    sku: str,
    location: str
) -> OverstockRisk:
    """
    Calculate overstock risk based on days of inventory.
    
    Algorithm:
    1. Get current stock level (Q_current)
    2. Get 60-day demand forecast (D_60day)
    3. Calculate days of inventory: DOI = Q_current / (D_60day / 60)
    4. Adjust for seasonality (reduce threshold for seasonal items)
    5. Calculate storage cost impact: Cost = Q_current * unit_cost * holding_rate
    6. Calculate working capital impact: WC = Q_current * unit_cost
    
    Risk Levels:
    - High: DOI > 90 days
    - Medium: 60 < DOI <= 90 days
    - Low: DOI <= 60 days
    
    Seasonal Adjustment:
    - For seasonal items, reduce thresholds by 30% during off-season
    """
    pass
```

#### Expiry Risk

```python
def calculate_expiry_risk(
    sku: str,
    batch_number: str,
    location: str
) -> ExpiryRisk:
    """
    Calculate expiry risk for batches with expiry dates.
    
    Algorithm:
    1. Get batch expiry date and quantity (Q_batch, expiry_date)
    2. Calculate days until expiry: days_remaining
    3. Get demand forecast for days_remaining (D_forecast)
    4. Calculate consumption probability: P(consume) = D_forecast / Q_batch
    5. Calculate expiry probability: P(expiry) = 1 - P(consume)
    
    Risk Levels:
    - High: days_remaining < 30 OR P(expiry) > 0.50
    - Medium: 30 <= days_remaining < 60 OR 0.25 < P(expiry) <= 0.50
    - Low: days_remaining >= 60 AND P(expiry) <= 0.25
    
    Priority for rebalancing: High risk batches with earliest expiry dates
    """
    pass
```

**Key Functions**:
```python
def detect_all_risks(location: str = None) -> RiskSummary:
    """Run all risk detection algorithms for all SKUs."""
    pass

def get_high_risk_items(risk_type: str) -> List[RiskAlert]:
    """Get all high-risk items for a specific risk type."""
    pass

def calculate_risk_score(
    stockout_prob: float,
    overstock_prob: float,
    expiry_prob: float
) -> float:
    """Composite risk score (0-100) for prioritization."""
    pass
```

### 4. Recommendation Engine Service

**Responsibility**: Generate actionable recommendations for reorder points, order quantities, and stock rebalancing.

**Interfaces**:
```python
GET /api/v1/recommendations/reorder/{sku}
GET /api/v1/recommendations/rebalance
POST /api/v1/recommendations/refresh
```

**Recommendation Algorithms**:

#### Dynamic Reorder Point

```python
def calculate_reorder_point(
    sku: str,
    location: str,
    service_level: float = 0.95
) -> ReorderRecommendation:
    """
    Calculate optimal reorder point using demand forecast and lead time.
    
    Algorithm (Safety Stock Method):
    1. Get lead time (L) from product master
    2. Get demand forecast for lead time period (D_L, σ_L)
    3. Get service level target (SL) - default 95%
    4. Calculate safety stock: SS = Z_SL * σ_L * sqrt(L)
       where Z_SL is the z-score for service level
    5. Calculate reorder point: ROP = D_L + SS
    6. Calculate order quantity using EOQ:
       EOQ = sqrt((2 * D_annual * order_cost) / holding_cost)
    7. Adjust for MOQ (minimum order quantity) constraints
    
    Returns:
        - reorder_point: int
        - order_quantity: int
        - safety_stock: int
        - service_level: float
        - rationale: str
    """
    pass
```

#### Stock Rebalancing

```python
def generate_rebalancing_recommendations(
    sku: str = None
) -> List[RebalanceRecommendation]:
    """
    Identify transfer opportunities between locations.
    
    Algorithm:
    1. For each SKU, get inventory levels at all locations
    2. Get risk assessments (stockout, overstock, expiry) for each location
    3. Identify pairs: (source with overstock/expiry, destination with stockout)
    4. Calculate transfer benefit score:
       Benefit = (stockout_risk_reduction * 10) + 
                 (overstock_cost_reduction * 5) + 
                 (expiry_risk_reduction * 15)
    5. Calculate transfer cost: Cost = quantity * unit_cost * transfer_rate
    6. Calculate net benefit: Net = Benefit - Cost
    7. Recommend transfers with Net > 0, sorted by Net descending
    8. Ensure transfers don't create new risks at source location
    
    Returns list of:
        - sku: str
        - source_location: str
        - destination_location: str
        - quantity: int
        - benefit_score: float
        - transfer_cost: float
        - rationale: str
    """
    pass
```

**Key Functions**:
```python
def calculate_economic_order_quantity(
    annual_demand: float,
    order_cost: float,
    holding_cost: float,
    unit_cost: float
) -> int:
    """Calculate EOQ for optimal order quantity."""
    pass

def adjust_for_constraints(
    calculated_quantity: int,
    min_order_qty: int,
    max_order_qty: int,
    order_multiple: int
) -> int:
    """Adjust calculated quantity for real-world constraints."""
    pass

def prioritize_recommendations(
    recommendations: List[Recommendation]
) -> List[Recommendation]:
    """Sort recommendations by urgency and impact."""
    pass
```

### 5. AI Copilot Service

**Responsibility**: Provide natural language interface for inventory queries and explanations.

**Interfaces**:
```python
POST /api/v1/copilot/query
POST /api/v1/copilot/action
GET /api/v1/copilot/history
```

**Architecture**:

The AI Copilot uses a Large Language Model (LLM) with function calling to translate natural language queries into structured API calls.

**Function Calling Schema**:

```python
AVAILABLE_FUNCTIONS = [
    {
        "name": "get_inventory_status",
        "description": "Get current inventory level and status for a SKU",
        "parameters": {
            "sku": "string",
            "location": "string (optional)"
        }
    },
    {
        "name": "get_demand_forecast",
        "description": "Get demand prediction for a SKU",
        "parameters": {
            "sku": "string",
            "location": "string",
            "horizon_days": "int (default 30)"
        }
    },
    {
        "name": "get_risk_assessment",
        "description": "Get risk analysis for a SKU",
        "parameters": {
            "sku": "string",
            "location": "string",
            "risk_type": "string (stockout|overstock|expiry|all)"
        }
    },
    {
        "name": "get_recommendations",
        "description": "Get reorder or rebalancing recommendations",
        "parameters": {
            "sku": "string (optional)",
            "recommendation_type": "string (reorder|rebalance)"
        }
    },
    {
        "name": "explain_recommendation",
        "description": "Explain why a recommendation was made",
        "parameters": {
            "recommendation_id": "string"
        }
    }
]
```

**Query Processing Flow**:

1. **User Query**: "What's the stockout risk for SKU-12345 in Mumbai?"
2. **LLM Processing**: Parse intent and extract parameters
3. **Function Call**: `get_risk_assessment(sku="SKU-12345", location="Mumbai", risk_type="stockout")`
4. **API Execution**: Call Risk Detector Service
5. **Response Formatting**: LLM formats structured data into natural language
6. **User Response**: "SKU-12345 in Mumbai has a 75% stockout risk in the next 7 days. Current stock is 50 units, but predicted demand is 120 units. I recommend ordering 100 units immediately."

**Key Functions**:
```python
def process_query(
    user_query: str,
    user_context: UserContext
) -> CopilotResponse:
    """
    Process natural language query and return response.
    
    Steps:
    1. Send query to LLM with function calling schema
    2. Extract function calls from LLM response
    3. Execute function calls against appropriate services
    4. Send results back to LLM for natural language formatting
    5. Return formatted response to user
    """
    pass

def execute_function_call(
    function_name: str,
    parameters: dict
) -> dict:
    """Execute a function call against backend services."""
    pass

def format_response(
    raw_data: dict,
    query_context: str
) -> str:
    """Format structured data into natural language response."""
    pass

def handle_action_request(
    action: str,
    parameters: dict,
    user_context: UserContext
) -> ActionResult:
    """Handle user requests to take action on recommendations."""
    pass
```

## Data Models

### Core Entities

#### Product
```python
class Product:
    sku: str  # Primary key
    name: str
    category: str
    supplier: str
    lead_time_days: int
    unit_cost: float
    min_order_quantity: int
    order_multiple: int
    expiry_tracking_enabled: bool
    created_at: datetime
    updated_at: datetime
```

#### Inventory
```python
class Inventory:
    id: int  # Primary key
    sku: str  # Foreign key to Product
    location: str
    quantity: int
    last_updated: datetime
    pending_orders: List[PendingOrder]
```

#### PendingOrder
```python
class PendingOrder:
    id: int  # Primary key
    sku: str  # Foreign key to Product
    location: str
    quantity: int
    expected_delivery_date: date
    order_date: date
    status: str  # pending, in_transit, delivered
```

#### SalesTransaction
```python
class SalesTransaction:
    id: int  # Primary key
    sku: str  # Foreign key to Product
    location: str
    quantity: int
    price: float
    transaction_date: datetime
    is_promotional: bool
```

#### Batch
```python
class Batch:
    id: int  # Primary key
    sku: str  # Foreign key to Product
    batch_number: str
    location: str
    quantity: int
    expiry_date: date
    received_date: date
```

#### DemandForecast
```python
class DemandForecast:
    id: int  # Primary key
    sku: str  # Foreign key to Product
    location: str
    forecast_date: date
    predicted_quantity: float
    lower_bound_80: float
    upper_bound_80: float
    confidence_score: float
    seasonality_factor: float
    trend_factor: float
    velocity_factor: float
    created_at: datetime
```

#### RiskAssessment
```python
class RiskAssessment:
    id: int  # Primary key
    sku: str  # Foreign key to Product
    location: str
    risk_type: str  # stockout, overstock, expiry
    risk_level: str  # high, medium, low
    risk_probability: float
    horizon_days: int  # for stockout risk
    days_of_inventory: float  # for overstock risk
    days_until_expiry: int  # for expiry risk
    assessed_at: datetime
```

#### Recommendation
```python
class Recommendation:
    id: int  # Primary key
    recommendation_type: str  # reorder, rebalance
    sku: str  # Foreign key to Product
    location: str
    recommended_action: str
    quantity: int
    priority_score: float
    rationale: str
    status: str  # pending, accepted, rejected, executed
    created_at: datetime
    expires_at: datetime
```

#### RebalanceRecommendation
```python
class RebalanceRecommendation(Recommendation):
    source_location: str
    destination_location: str
    transfer_cost: float
    benefit_score: float
```

### Database Schema Design

**Indexes**:
- `inventory(sku, location)` - for fast inventory lookups
- `sales_transaction(sku, location, transaction_date)` - for demand analysis
- `demand_forecast(sku, location, forecast_date)` - for forecast retrieval
- `risk_assessment(sku, location, risk_type, assessed_at)` - for risk queries
- `batch(sku, location, expiry_date)` - for expiry risk detection

**Partitioning**:
- `sales_transaction` - partition by transaction_date (monthly partitions)
- `demand_forecast` - partition by forecast_date (monthly partitions)

**Time-Series Optimization**:
- Use TimescaleDB extension for PostgreSQL
- Convert `sales_transaction` and `demand_forecast` to hypertables
- Enable automatic chunk management and compression

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*


### Property Reflection

After analyzing all acceptance criteria, I've identified the following consolidations to eliminate redundancy:

**Consolidations:**
- **2.1, 2.2, 2.3**: All test stockout probability calculation for different horizons → Combine into single property about multi-horizon stockout calculation
- **2.4, 2.5**: Both test risk classification thresholds → Combine into single property about risk level classification
- **3.3, 3.4**: Both test overstock risk thresholds → Combine into single property about overstock classification
- **4.2, 4.3**: Both test expiry risk thresholds → Combine into single property about expiry risk classification
- **1.2, 1.3, 1.4, 1.5**: All test that specific factors are incorporated → Combine into property about forecast factor incorporation
- **8.6, 8.7**: Both test data round-trip persistence → Combine into single property about data persistence

**Properties to Keep Separate:**
- Each unique algorithm (demand prediction, risk detection, recommendations) needs its own properties
- Input validation and error handling are distinct from core logic
- Performance requirements are separate from functional correctness
- Security properties (authentication, authorization, tenant isolation) are distinct concerns

### Correctness Properties

Property 1: Demand forecast completeness
*For any* SKU with at least 12 months of historical data, when demand prediction is requested, the forecast should include daily predictions for 30 days with confidence intervals (lower bound, mean, upper bound) and a confidence score.
**Validates: Requirements 1.1, 1.6**

Property 2: Demand forecast factor incorporation
*For any* SKU, the demand forecast should reflect changes in seasonality patterns, regional behavior, promotional activity, and inventory velocity—meaning that modifying any of these factors should result in a different forecast.
**Validates: Requirements 1.2, 1.3, 1.4, 1.5**

Property 3: Multi-horizon stockout risk calculation
*For any* SKU at any location, the risk detector should calculate stockout probabilities for 7-day, 14-day, and 30-day horizons, and each probability should be between 0 and 1.
**Validates: Requirements 2.1, 2.2, 2.3**

Property 4: Stockout risk classification
*For any* SKU, when stockout probability exceeds 0.70, it should be flagged as high risk; when probability is between 0.40 and 0.70, it should be flagged as medium risk; when probability is below 0.40, it should be flagged as low risk.
**Validates: Requirements 2.4, 2.5**

Property 5: Stockout risk factors
*For any* SKU, the stockout risk calculation should decrease when current stock increases, decrease when predicted demand decreases, and decrease when pending orders are added.
**Validates: Requirements 2.6, 2.7**

Property 6: Overstock risk calculation
*For any* SKU at any location, the risk detector should calculate days of inventory (current stock divided by daily demand rate) and overstock probability.
**Validates: Requirements 3.1, 3.2**

Property 7: Overstock risk classification
*For any* SKU, when days of inventory exceeds 90 days, it should be flagged as high overstock risk; when days of inventory is between 60 and 90 days, it should be flagged as medium risk; when below 60 days, it should be flagged as low risk.
**Validates: Requirements 3.3, 3.4**

Property 8: Overstock seasonal adjustment
*For any* seasonal SKU during off-season periods, the overstock risk thresholds should be adjusted (reduced by at least 20%) compared to non-seasonal SKUs.
**Validates: Requirements 3.6**

Property 9: Expiry risk calculation
*For any* batch with expiry tracking enabled, the risk detector should calculate expiry risk based on days until expiry and predicted consumption rate.
**Validates: Requirements 4.1, 4.4**

Property 10: Expiry risk classification
*For any* batch with expiry tracking, when expiry is within 30 days, it should be flagged as high risk; when expiry is between 30 and 60 days, it should be flagged as medium risk; when expiry is beyond 60 days, it should be flagged as low risk.
**Validates: Requirements 4.2, 4.3**

Property 11: Expiry risk prioritization
*For any* set of batches with high expiry risk, when rebalancing recommendations are generated, these batches should have higher priority scores than batches without expiry risk.
**Validates: Requirements 4.5**

Property 12: Reorder point calculation factors
*For any* SKU, the calculated reorder point should increase when predicted demand increases, increase when lead time increases, increase when service level target increases, and increase when demand variability increases.
**Validates: Requirements 5.1, 5.2, 5.3, 5.4**

Property 13: Reorder recommendation completeness
*For any* SKU, when reorder recommendations are generated, they should include both a reorder point value and a recommended order quantity.
**Validates: Requirements 5.6**

Property 14: Minimum order quantity constraint
*For any* SKU with a minimum order quantity (MOQ) constraint, the recommended order quantity should be greater than or equal to the MOQ.
**Validates: Requirements 5.7**

Property 15: Rebalancing opportunity identification
*For any* SKU present at multiple locations, when one location has high overstock or expiry risk and another location has high stockout risk, the recommendation engine should generate at least one rebalancing recommendation.
**Validates: Requirements 6.1, 6.2**

Property 16: Rebalancing cost consideration
*For any* potential transfer, when transfer cost exceeds the benefit of risk reduction, no rebalancing recommendation should be generated.
**Validates: Requirements 6.3**

Property 17: Rebalancing recommendation completeness
*For any* rebalancing recommendation, it should include source location, destination location, SKU, quantity, and a rationale explaining the benefit.
**Validates: Requirements 6.5**

Property 18: AI copilot SKU query completeness
*For any* valid SKU query to the AI copilot, the response should include current stock levels, risk assessments (stockout, overstock, expiry), and any active recommendations.
**Validates: Requirements 7.2**

Property 19: AI copilot action execution
*For any* action request on a recommendation (accept, reject, execute), the system state should be updated to reflect the action, and the recommendation status should change accordingly.
**Validates: Requirements 7.4**

Property 20: AI copilot response time
*For any* query to the AI copilot, the response should be returned within 3 seconds.
**Validates: Requirements 7.5**

Property 21: Data validation rejection
*For any* inventory, sales, or product data upload with missing required fields or invalid data types, the system should reject the upload and return specific error messages identifying the invalid fields.
**Validates: Requirements 8.1, 8.4**

Property 22: Data persistence round-trip
*For any* valid product master data or batch data uploaded to the system, retrieving the data should return all originally uploaded attributes without loss or corruption.
**Validates: Requirements 8.6, 8.7**

Property 23: Data format support
*For any* valid inventory or sales data in CSV or JSON format, the system should successfully parse and store the data.
**Validates: Requirements 8.3**

Property 24: Sales data persistence
*For any* valid sales transaction data uploaded, the data should be stored and retrievable for demand analysis queries.
**Validates: Requirements 8.2**

Property 25: Batch processing performance
*For any* batch of up to 10,000 SKUs, demand prediction processing should complete within 15 minutes.
**Validates: Requirements 9.4**

Property 26: API statelessness
*For any* sequence of API calls, the response to each call should depend only on the request parameters and stored data, not on previous API calls in the session.
**Validates: Requirements 9.6**

Property 27: API authentication
*For any* API request without valid authentication credentials (API key or OAuth token), the system should return HTTP 401 Unauthorized.
**Validates: Requirements 10.2**

Property 28: API response format
*For any* successful API request, the response should be valid JSON that can be parsed without errors.
**Validates: Requirements 10.3**

Property 29: API error responses
*For any* API request that results in an error, the response should include a standard HTTP status code (4xx or 5xx) and a descriptive error message in the response body.
**Validates: Requirements 10.4**

Property 30: API rate limiting
*For any* API key, when the number of requests exceeds 1000 in a 1-hour window, subsequent requests should be rejected with HTTP 429 Too Many Requests.
**Validates: Requirements 10.6**

Property 31: Password complexity validation
*For any* password submitted during user registration or password change, passwords that don't meet complexity requirements (minimum 12 characters, mixed case, numbers, symbols) should be rejected.
**Validates: Requirements 11.3**

Property 32: Account lockout after failed attempts
*For any* user account, after 3 consecutive failed authentication attempts, the account should be temporarily locked and authentication should fail even with correct credentials for 15 minutes.
**Validates: Requirements 11.4**

Property 33: Role-based access control
*For any* user with a specific role (Admin, Inventory_Manager, Read_Only_User), API requests to resources outside their permitted scope should be rejected with HTTP 403 Forbidden.
**Validates: Requirements 11.5**

Property 34: Audit logging
*For any* data access operation (read, write, update, delete), an audit log entry should be created containing user identifier, timestamp, action type, and resource accessed.
**Validates: Requirements 11.6**

Property 35: Multi-tenant data isolation
*For any* user belonging to tenant A, queries for inventory, sales, or product data should never return data belonging to tenant B.
**Validates: Requirements 11.7**

Property 36: Inventory query response time
*For any* query for current inventory status of a single SKU, the system should respond within 2 seconds.
**Validates: Requirements 12.1**

Property 37: Demand prediction response time
*For any* batch request for demand predictions covering up to 100 SKUs, the system should return results within 5 seconds.
**Validates: Requirements 12.2**

Property 38: Risk detection batch performance
*For any* risk detection run covering all SKUs in the system, the analysis should complete within 10 minutes.
**Validates: Requirements 12.3**

Property 39: Cache invalidation
*For any* data update (inventory level change, new sales transaction), cached data should be invalidated and subsequent queries should reflect the update within 5 minutes.
**Validates: Requirements 12.7**

## Error Handling

### Error Categories

1. **Validation Errors**: Invalid input data, missing required fields, data type mismatches
2. **Business Logic Errors**: Insufficient data for predictions, conflicting recommendations
3. **System Errors**: Database connection failures, external service timeouts, resource exhaustion
4. **Authentication/Authorization Errors**: Invalid credentials, insufficient permissions, expired tokens

### Error Handling Strategy

**Validation Errors**:
- Return HTTP 400 Bad Request with detailed field-level error messages
- Include error codes for programmatic handling
- Example: `{"error": "VALIDATION_ERROR", "fields": {"quantity": "must be positive integer"}}`

**Business Logic Errors**:
- Return HTTP 422 Unprocessable Entity with explanation
- Provide actionable guidance for resolution
- Example: `{"error": "INSUFFICIENT_DATA", "message": "SKU requires 12 months of sales history", "available_months": 6}`

**System Errors**:
- Return HTTP 500 Internal Server Error for unexpected failures
- Return HTTP 503 Service Unavailable for temporary outages
- Log full error details internally, return generic message to client
- Implement automatic retry with exponential backoff for transient failures

**Authentication/Authorization Errors**:
- Return HTTP 401 Unauthorized for authentication failures
- Return HTTP 403 Forbidden for authorization failures
- Include WWW-Authenticate header for 401 responses
- Never expose sensitive information in error messages

### Graceful Degradation

**Insufficient Historical Data**:
- When SKU has < 12 months of data, use category-based forecasting
- Return forecast with low confidence score (< 0.5)
- Include warning in response: `{"warning": "INSUFFICIENT_DATA", "confidence": 0.3}`

**Missing Optional Data**:
- When promotional data unavailable, proceed without promotional adjustments
- When regional data unavailable, use global averages
- Document assumptions in forecast metadata

**Service Timeouts**:
- Implement circuit breaker pattern for external dependencies
- Return cached results when available during outages
- Include staleness indicator: `{"cached": true, "cache_age_minutes": 45}`

**Partial Failures**:
- For batch operations, process all valid items and report failures separately
- Return HTTP 207 Multi-Status with per-item results
- Example: `{"processed": 95, "failed": 5, "errors": [...]}`

## Testing Strategy

### Dual Testing Approach

The testing strategy employs both unit testing and property-based testing as complementary approaches:

**Unit Tests**: Focus on specific examples, edge cases, and integration points
- Specific scenarios with known inputs and expected outputs
- Edge cases (empty data, boundary values, null handling)
- Error conditions and exception handling
- Integration between components
- Mock external dependencies for isolated testing

**Property-Based Tests**: Focus on universal properties across all inputs
- Generate random valid inputs to test properties at scale
- Verify invariants hold across wide input ranges
- Catch unexpected edge cases through randomization
- Test algorithmic correctness without hardcoding examples
- Minimum 100 iterations per property test

### Property-Based Testing Configuration

**Framework Selection**:
- Python: Use Hypothesis library for property-based testing
- Each property test must run minimum 100 iterations
- Use `@given` decorators with appropriate strategies
- Configure deadline to 5000ms for complex properties

**Test Tagging**:
Each property test must include a comment tag referencing the design property:
```python
# Feature: lighthouse-inventory-intelligence, Property 1: Demand forecast completeness
@given(sku=skus(), historical_months=st.integers(min_value=12, max_value=36))
def test_demand_forecast_completeness(sku, historical_months):
    # Test implementation
    pass
```

**Generator Strategies**:
- SKU identifiers: alphanumeric strings 6-12 characters
- Quantities: positive integers 0-10000
- Dates: within last 3 years for historical, next 90 days for forecasts
- Probabilities: floats between 0.0 and 1.0
- Locations: from predefined list of valid location codes
- Prices: positive floats with 2 decimal places

### Test Coverage by Component

**Demand Predictor**:
- Unit tests: Specific seasonal patterns, promotional impacts, edge cases (single data point, all zeros)
- Property tests: Properties 1, 2 (forecast completeness and factor incorporation)
- Integration tests: End-to-end prediction with real data samples

**Risk Detector**:
- Unit tests: Boundary values for risk thresholds, specific risk scenarios
- Property tests: Properties 3-11 (stockout, overstock, expiry risk calculations and classifications)
- Integration tests: Multi-SKU risk detection with various inventory states

**Recommendation Engine**:
- Unit tests: MOQ constraints, specific rebalancing scenarios
- Property tests: Properties 12-17 (reorder points, order quantities, rebalancing)
- Integration tests: End-to-end recommendation generation with multiple locations

**AI Copilot**:
- Unit tests: Specific query patterns, function call routing
- Property tests: Properties 18-20 (query completeness, action execution, response time)
- Integration tests: Multi-turn conversations, complex queries

**Data Ingestion**:
- Unit tests: Specific validation rules, format parsing edge cases
- Property tests: Properties 21-24 (validation, persistence, format support)
- Integration tests: Large file uploads, concurrent uploads

**API Layer**:
- Unit tests: Specific authentication scenarios, error responses
- Property tests: Properties 26-30 (statelessness, authentication, format, rate limiting)
- Integration tests: Full API workflows, concurrent requests

**Security**:
- Unit tests: Specific RBAC scenarios, password patterns
- Property tests: Properties 31-35 (password validation, lockout, RBAC, audit logging, tenant isolation)
- Security tests: Penetration testing, vulnerability scanning

**Performance**:
- Unit tests: Not applicable
- Property tests: Properties 36-39 (response times, batch performance, cache invalidation)
- Load tests: Sustained load, spike testing, stress testing

### Test Data Management

**Synthetic Data Generation**:
- Generate realistic sales patterns with seasonality
- Create multi-location inventory scenarios
- Simulate promotional periods and regional variations
- Use seed values for reproducible test runs

**Test Fixtures**:
- Predefined SKU catalog with various characteristics
- Sample historical sales data (12-36 months)
- Multi-location inventory snapshots
- Known seasonal patterns for validation

**Test Database**:
- Use in-memory database (SQLite) for unit tests
- Use containerized PostgreSQL for integration tests
- Reset database state between test runs
- Seed with minimal required data for each test

### Continuous Integration

**CI Pipeline**:
1. Lint and format checks (flake8, black, mypy)
2. Unit tests (fast, run on every commit)
3. Property-based tests (100 iterations, run on every commit)
4. Integration tests (slower, run on pull requests)
5. Performance tests (run nightly)
6. Security scans (run weekly)

**Test Execution Time Targets**:
- Unit tests: < 2 minutes total
- Property tests: < 5 minutes total
- Integration tests: < 10 minutes total
- Full suite: < 20 minutes

**Coverage Requirements**:
- Minimum 80% code coverage for core logic
- 100% coverage for critical paths (risk detection, recommendations)
- Track coverage trends over time
- Fail builds if coverage decreases

### Testing Best Practices

1. **Test Independence**: Each test should be runnable in isolation
2. **Clear Assertions**: Use descriptive assertion messages
3. **Minimal Mocking**: Prefer real implementations over mocks when feasible
4. **Fast Feedback**: Optimize test execution time
5. **Deterministic Tests**: Avoid flaky tests, use fixed seeds for randomization
6. **Test Documentation**: Document complex test scenarios and edge cases
7. **Property Clarity**: Each property test should validate exactly one property
8. **Failure Diagnosis**: Property tests should shrink failing examples for easy debugging

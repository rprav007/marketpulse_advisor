# MarketPulse Advisor - Testing Scenarios

This document lists the scenarios and sample queries you can use to test the **MarketPulse Advisor** multi-agent system via the ADK Web UI or the CLI runner.

## Testing Setup
Ensure the ADK Web Server is running from the root workspace directory:
```bash
adk web --port 8080
```
Open **[http://127.0.0.1:8080](http://127.0.0.1:8080)** in your browser and select the `marketpulse_advisor` agent from the dropdown.

---

## Scenarios & Sample Queries

### Scenario 1: Happy Path (Standard Pre-Market Run)
*   **Subscriber ID**: `user_happy`
*   **Goal**: Verify that the technical scanner, backtester, and risk advisor coordinate successfully. AAPL and NVDA should be discarded due to Tech sector cap limits (active Tech exposure is 15%), and AMZN should be approved and sized (Risk-of-Ruin, capped at 20% equity).
*   **Sample Queries**:
    -   `Perform the pre-market scan for user_happy`
    -   `Show me the volatility compression recommendations for subscriber user_happy`
    -   `Run pre-market pipeline for user_happy`

### Scenario 2: Circuit Breaker (Daily Loss >= 2.5%)
*   **Subscriber ID**: `user_breached_loss`
*   **Goal**: Verify that the RiskAdvisorAgent raises a critical alert and aborts recommendations when the daily loss exceeds the 2.5% account threshold ($3,750 on a $150k account).
*   **Sample Queries**:
    -   `Perform the pre-market scan for user_breached_loss`
    -   `Are there any trade recommendations today for user_breached_loss?`

### Scenario 3: Position Cap Breached (3+ Positions Open)
*   **Subscriber ID**: `user_max_positions`
*   **Goal**: Verify that the RiskAdvisorAgent refuses to recommend new trades because the subscriber's account is already at the maximum limit of 3 concurrent positions.
*   **Sample Queries**:
    -   `Check for breakout recommendations for user_max_positions`
    -   `Can you size trades for user_max_positions?`

### Scenario 4: User Order Approval & Execution
*   **Subscriber ID**: `user_happy`
*   **Goal**: Approve a generated trade setup and trigger OCO bracket order submission to the brokerage API (logging to Firestore audit collections).
*   **Sample Queries** (run after Scenario 1 is complete):
    -   `I approve the AMZN recommendation. Please submit the bracket order for me.`
    -   `Execute the AMZN trade with the calculated targets.`

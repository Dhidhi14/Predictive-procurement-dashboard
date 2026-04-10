# Comprehensive Technical Project Documentation
## Predictive Procurement Analytics System

---

### Table of Contents
1. [Introduction and Project Scope](#1-introduction-and-project-scope)
2. [High-Level System Architecture](#2-high-level-system-architecture)
3. [Raw Data Dictionary (`new/Augmented/` & `new/master_data/`)](#3-raw-data-dictionary)
4. [ETL Pipeline Implementation (`etl_pipeline.py`)](#4-etl-pipeline-implementation)
5. [Feature Engineering & Behavioral Economics](#5-feature-engineering--behavioral-economics)
6. [Machine Learning Engine (`feature_engine.py`)](#6-machine-learning-engine)
7. [Decision Support Dashboard (`dashboard_app.py`)](#7-decision-support-dashboard)
8. [Target Database Schema (`schema.sql`)](#8-target-database-schema)
9. [Deployment and Execution](#9-deployment-and-execution)
10. [Code File Manifest & Detailed Explanations](#10-code-file-manifest--detailed-explanations)

---

## 1. Introduction and Project Scope

The **Predictive Procurement Analytics System** addresses a fundamental operational inefficiency in higher education: the over-ordering of physical textbooks and course materials. Historically, universities order textbooks assuming 100% of enrolled students will purchase them. In reality, students frequently opt-out of university-provided bundles due to high costs, utilizing secondary markets, renting, or pirating materials instead. This leads to massive sunk inventory costs.

This project implements a hybrid data architecture to process 7.56 million records while surfacing student buy-back risks, demographics, and product pricing. It predicts exactly which students are likely to opt-out, calculates the projected financial spend, and surfaces these insights to procurement executives via an interactive Streamlit dashboard. 

---

## 2. High-Level System Architecture

The application is structured into a modular pipeline, separating data ingestion, predictive modeling, and user interface rendering:

1. **Storage / File System**: Raw historical datasets containing millions of student interactions split by university (FIU, EKU, Mercer, etc.) exist in the `new/Augmented/` directory.
2. **ETL (Extract, Transform, Load)**: The `etl_pipeline.py` script dynamically crawls the file system, ingests the data into Pandas dataframes, handles memory-safe sampling, and engineers dozens of derived semantic features.
3. **Machine Learning**: The `feature_engine.py` script consumes the ETL dataframe, isolates independent variables, trains a `RandomForestClassifier` in real-time, and scores every student row with a definitive `Predicted_Purchase_Prob`.
4. **Presentation**: The `dashboard_app.py` script serves a modern, dark-themed Streamlit web interface. It caches the heavy operations and uses Plotly Express to render interactive data visualizations.

---

## 3. Raw Data Dictionary

The data ingested from the source directories contains granular transactional, demographic, and qualitative data.

### 3.1 External Data Sub-System: Sentiment Aggregation
The system utilizes a supplemental dataset, `Training_Data_Clean.csv` (located in `new/master_data/`), which contains hundreds of thousands of student feedback data points and survey responses on various textbooks. 
* **Usage (`enrich_sentiment.py`)**: This script acts as a specialized qualitative pre-processor. It reads `Training_Data_Clean.csv`, standardizes the book titles, and aggregates (averages) various qualitative metrics like "conceptual clarity" and "value for money." It outputs a smaller `book_sentiment.csv` file that the main `etl_pipeline.py` later left-joins against the bulk transaction data. This enables the ML model to factor in "student satisfaction" as a predictor for purchase behavior. 

### Identity and Academic Context
* `sis_user_id`: Unique surrogate key representing a specific student.
* `section_id`: The exact course section the student is enrolled in (e.g., `BN-8304-1-126-A-23-421`).
* `term_code`: The semester identifier (e.g., `F` for Fall, `W` for Winter/Spring).
* `term_year`: The 2-digit academic year.
* `student_full_part_time_status`: `F` (Full-Time) or `P` (Part-Time).

### Demographic and Financial Context
* `family_annual_income`: Numeric representation of household income, strongly correlating with price sensitivity.
* `has_loan`: Binary indicator (1/0) of whether the student relies on student loans.
* `has_scholarship`: Binary indicator (1/0) of whether the student possesses scholarship funding.
* `part_time_job`: Binary indicator outlining employment status.

### Product (Textbook) Context
* `title`: Title of the textbook.
* `author`: Author of the textbook.
* `isbn`: Unique product identifier.
* `ebook_ind`: 1.0 if the product is an explicit digital access code or eBook.
* `is_rental`: Binary flag indicating if the specific transaction target is a rental item.
* `retail_new`: The brand-new purchase price of the textbook.
* `retail_used`: Secondary market textbook price.
* `retail_new_rent`: The price to rent a brand-new copy.

### Derived Metrics (Pre-calculated in source DB)
* `price_affordability_score`: A proprietary metric representing the general financial strain of the purchase.
* `will_buy`: The **Target Variable**. 1 if the student actually purchased the bundle, 0 if they opted out.

---

## 4. ETL Pipeline Implementation (`etl_pipeline.py`)

The ETL module is solely responsible for turning the raw, noisy CSV files into a clean `mapped_df` that directly fuels the machine learning model and dashboard.

### 4.1 Hybrid Data Strategy (Option B)
The raw dataset totals 7.5 million records (~2.1 GB). Loading this entire corpus into a single dense matrix for ML encoding (dummies) requires over 30 GiB of RAM, which exceeded system limits. 

**Implementation detail:** The system now uses a **Hybrid Strategy**:
1. **Pre-computation (Global Accuracy)**: An offline script (`precompute_kpis.py`) processes the full 7.56M records in chunks, aggregating exact totals for `Projected_Spend` and `Book_Count`. This ensures the dashboard's high-level totals are **100% accurate** compared to the source data ($24.06M Global Spend).
2. **Dynamic Sampling (ML Interactivity)**: For predictive modeling and "What-If" simulations, the pipeline samples a representative **50,000-row** subset. This allows for real-time interactivity while maintaining a stable memory footprint.

### 4.2 Feature Engineering and Semantic Mapping
The raw columns are transformed to meet the semantic naming conventions utilized by the existing dashboard logic.

* **`Term`**: Synthesized by concatenating `term_code` and `term_year` (e.g., "F 21").
* **`Year`**: Extracted from `term_year`.
* **`Semester`**: Mapped from `term_code` (e.g., "F" to "Fall").
* **`College`**: Derived from the source CSV filename (e.g., "FIU", "EKU").
* **`Dept_Code`**: Extracted positionally from the `section_id`. 
* **`Student_Type`**: Safely mapped via `{"F": "Full-Time", "P": "Part-Time"}`.

### 4.3 Behavioral Economic Equations
The pipeline generates composite economic behavioral metrics:

1. **`Rental_to_Retail_Ratio`**:
   To prevent ZeroDivision errors when a book is inexplicably free, the denominator replaces `0.0` with `100.0`.
   ```python
   retail_new_safe = retail_new.replace(0.0, 100.0)
   ratio = retail_rent / retail_new_safe
   mapped_df["Rental_to_Retail_Ratio"] = ratio.clip(0.0, 1.5)
   ```

2. **`Arbitrage_Index`**:
   Defined as `1.0 - Rental_to_Retail_Ratio`. A ratio of `0.2` implies the student can rent the book for 20% of the cost. This creates an Arbitrage Index of `0.8` (Extremely High), signaling massive savings outside the standard bundle and predicting a high likelihood of opt-out.

3. **`Wallet_Pressure_Score`**:
   The raw `price_affordability_score` is normalized to a percentage via Min-Max scaling.
   ```python
   mapped_df["Wallet_Pressure_Score"] = (afford_score / afford_score.max()).clip(0, 1)
   ```

4. **`Digital_Lock_Flag`**:
   Derived directly from `ebook_ind`. Because digital access codes (like Pearson MyLab) cannot be bought on secondary markets, this is the strongest predictor enforcing an opt-in.

---

## 5. Feature Engineering & Behavioral Economics

This layer transforms raw transactional data into high-entropy signals that represent student psychology and economic pressure.

### 5.1 Economic Pressure Metrics

*   **`Rental_to_Retail_Ratio`**:
    *   **Logic**: `retail_new_rent / retail_new`
    *   **Significance**: Measures the "Rental Discount." If a book costs $100 to buy but only $20 to rent, the ratio is 0.2. A low ratio strongly predicts opt-out as the financial barrier to the university's "All-in" format is high compared to rental alternatives.
*   **`Arbitrage_Index`**:
    *   **Logic**: `1.0 - Rental_to_Retail_Ratio`
    *   **Significance**: Quantifies the "Savings Opportunity" on the secondary market. An index of 0.8 means students can save 80% by opting out and renting elsewhere. This is the #1 driver for price-sensitive students.
*   **`Wallet_Pressure_Score`**:
    *   **Logic**: Min-Max scaling of `price_affordability_score`.
    *   **Significance**: Normalizes the raw price into a 0-1 range. It signifies the relative "Pain Point" for the student's specific household income level.

### 5.2 Access & Governance Signals

*   **`Digital_Lock_Flag`**:
    *   **Logic**: Binary flag derived from `ebook_ind`.
    *   **Significance**: Represents "Non-Commodity" items. Unlike physical books, digital access codes (MasteringPhysics, MyLab) cannot be bought second-hand. High Digital Lock values **force** students to opt-in, effectively capping the opt-out risk.

### 5.3 Sentiment & Qualitative Enrichment

By integrating student feedback, the system now predicts based on "Product Quality":
*   **`Sentiment_Understandability`**: Books rated as "Hard to Understand" see higher opt-out rates because students are less willing to invest in a resource they find unhelpful.
*   **`Sentiment_Value_For_Money`**: Represents the percieved ROI. High price + Low value = Guaranteed Opt-Out.

---

## 6. Machine Learning Engine (`feature_engine.py`)

The analytics engine doesn't rely on synthetic probability; it spins up a real inference pipeline using Scikit-Learn.

### 5.1 Algorithm Selection
We employ a `RandomForestClassifier`. Decision Trees natively handle missing data effortlessly and are mathematically robust to outliers in variables like `family_annual_income`. Trees also do not require strict normalization unlike Logistic Regression.

**Configuration:**
```python
clf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=1)
```
* `n_estimators=50`: Calculates the ensemble consensus over 50 specific trees.
* `max_depth=6`: Truncates tree growth. Ensures the model generalizes behavior over subsets rather than overfitting entirely to the training data.
* `n_jobs=1`: Serial execution per instance to maintain memory stability in the Streamlit runtime.

### 5.2 The Training Paradigm & Accuracy
1. **Feature matrix (X)** focus on dynamically selected features including `Arbitrage_Index`, `Wallet_Pressure_Score`, `Digital_Lock_Flag`, `family_annual_income`, and encoded categorical terms.
2. **Target array (y)** is `Actual_Purchase_Flag` (the absolute truth of whether a student bought the bundle).
3. **Accuracy Monitoring**: The model typically achieves **~92.8% Accuracy** on unseen validation data. Since the 50k sample is drawn randomly from the 7.5M population, it captures the behavioral trends with high statistical confidence.

### 5.3 Forecast Output
After invoking `clf.fit(X, y)`, the pipeline calls `clf.predict_proba(X_all)`.
It specifically isolates the probability mapping to the positive class (`1`), appending this array to the global dataframe as `Predicted_Purchase_Prob`. 

**The Projected Spend Algorithm:**
To turn these behavioral probabilities into capital execution, the engine runs:
```python
Projected_Spend = Predicted_Demand_Units * Unit_Price * Predicted_Purchase_Prob
```
If a $100 book has a 30% probability of purchase, the projected spend allocation for that line item falls to $30.

### 5.4 Sentiment-Driven Enrichment
By joining the procurement data with the `Training_Data_Clean.csv` sentiment set, we introduced qualitative behavioral features.
- **Key Insight**: "Conceptual Clarity" and "Value for Money" scores became top-10 predictors of purchase behavior.
- **Accuracy Lift**: The model accuracy improved from 92.8% to **93.2%**, proving that student satisfaction is a secondary driver of bundle adoption.

### 6.5 Behavioral Signal Importance
The model typically identifies `Digital_Lock_Flag` and `Arbitrage_Index` as the most significant drivers of student behavior. While these are calculated internally, the dashboard focuses on the **financial outcome** of these behaviors through the Savings Potential ROI metrics.

---

## 7. Decision Support Dashboard (`dashboard_app.py`)

The presentation layer is powered entirely by Streamlit, utilizing custom CSS for a specialized "glassmorphism" aesthetic targeting a deep navy background.

### 7.1 ROI-First Metric Strategy
The dashboard prioritizes three core executive KPIs:
1. **Total Book Demand**: Absolute volume forecast for the upcoming semester.
2. **Total Projected Spend**: The baseline financial capital required for bulk procurement.
3. **Potential Savings (ROI)**: A predictive metric representing the dollar value of expected student opt-outs. This allows procurement teams to adjust orders **before** the semester begins.

### 7.2 Visualization Hierarchy
The dashboard is organized into logical analytical rows:
- **Price Sensitivity Context**: Correlating price categories with total spend.
- **Risk Segmentation**: Analyzing opt-out probability across student types (Part-time vs Full-time).
- **Time-Series Forecasting**: Visualizing spend trends across terms and semesters.
- **Departmental Savings Potential**: A high-impact Pareto chart identifying the top 10 departments where negotiation or inventory adjustments will yield the highest ROI.

### 7.3 Performance Optimization
To prevent OOM and maximize speed, the dashboard handles two data sources:
```python
@st.cache_data(show_spinner=False)
def get_summary_data() -> pd.DataFrame:
    # Loads 100% accurate totals from 7.5M record pre-computation
    return load_summary_kpis()

@st.cache_data(show_spinner=False)
def get_raw_data() -> pd.DataFrame:
    # Loads representative 50k sample for ML simulations
    return load_feature_table(sample_limit=50000)
```

### 7.4 Global Filtering & Accuracy
#### 7.4.1 Interactive Left Filter Menu
Users can dynamically constrain the entire dataset by College, Year, Department, Semester, and Student Type.

#### 7.4.2 Sidebar Accuracy Gauge
Positioned directly below the filters, this gauge provides real-time transparency into the machine learning model's performance (calculated via `clf.score`), ensuring users know the confidence level of the Savings ROI predictions.

### 7.5 Deep Mathematical Accuracy & Data Quality Checks
To ensure strictly accurate capital projections that can face procurement audits:
1. **Weighted ROI Calculation**: The Savings Potential KPI uses a direct weighted average (`sum(Gross_Spend * Opt_Out_Prob) / sum(Gross_Spend)`) across whatever sample is loaded. It bypasses simplistic median/mean reductions and correctly treats high-value items with more weight over overall savings totals.
2. **Gross Spend Harminization**: All categorical and time-series charts map "Projected Spend" against actual `Gross_Spend` (Predicted_Demand * Unit_Price). Previously, some underlying queries mixed post-opt-out revenue with gross spend, creating visual contradictions against the top KPI dashboard totals.
3. **Format Efficiency Fix**: When investigating financial exposure by Digital/Physical formats, the "Savings Potential" dialog strictly defines savings as `Gross * Opt_Out_Probability`, terminating compound multiplication errors (e.g., `Revenue * Opt_Out` which previously factored out the savings twice).
4. **Volume Integration**: Publisher-level deep-dives automatically factor in `Predicted_Demand_Units` when assessing limits on savings ROI. High-risk, low-volume vendors are therefore appropriately down-ranked compared to major publishers where minor opt-out probability improvements lead to massive bulk savings.
5. **Global Representation Scaling**: To address visually misaligned volume counts between the interactive sample and the offline summary, an automated *Scale Factor* maps the sampled units to match 100% of the `6.7M` global KPI totals natively before UI rendering. Therefore, the "Total Book Demand" KPI will now vertically align flawlessly with the counts aggregated inside the Price Category and Unit Demand charts.

---

## 8. Target Database Schema (`schema.sql`)
Documentation of the underlying PostgreSQL/Snowflake schema used for the global pre-computation.

---

## 9. Deployment and Execution
### Launch the Application
To run the server natively in development:
```bash
source .venv/bin/activate && streamlit run dashboard_app.py
```

---

## 10. Code File Manifest & Detailed Explanations
The project is built using a micro-script architecture, mapping different logical phases of the data pipeline to explicit Python scripts. Here is the detailed explanation for each active Python script in the system:

1. **`dashboard_app.py`** (The UI & Controller Layer)
   * **Functionality**: The core Streamlit web application. It houses the `get_raw_data()` and `get_summary_data()` cached data loaders to pull in the CSV outputs. It evaluates side-bar filters dynamically, calculates on-the-fly execution KPIs, and is responsible for rendering the complex Plotly glassmorphism charts and gauge metrics. It also interacts with `feature_engine.py` to train models strictly on the filtered subsets.

2. **`etl_pipeline.py`** (The Data Wrangler)
   * **Functionality**: Responsible for joining the raw master transaction files (`master_data_sampled.csv`) with the aggregated sentiment data. It evaluates memory-efficient transformations using fast C-engines, drops outdated formats (e.g., historical '21' semester data), creates vital ML economic features like `Arbitrage_Index` and `Wallet_Pressure_Score`, and enforces strict memory types (like `category` and `float32`). It feeds both the ML model and the Dashboard.

3. **`feature_engine.py`** (The Internal Machine Learning API)
   * **Functionality**: Defines the core Scikit-Learn logic. It contains `train_model(df)` which dynamically drops redundant identifiers, balances features, and trains a single fast `RandomForestClassifier`. Additionally, it exposes `apply_predictions(df, clf)` which is capable of applying simulated What-If modifications (like a 10% discount on prices) and re-predicting purchase behavior. 

4. **`precompute_kpis.py`** (The Global Offline Aggregator)
   * **Functionality**: A backend processing engine designed to bypass Streamlit's real-time memory constraints. This script directly utilizes `chunksize=750000` to iterate through the massive gigabytes-large `master_data.csv`. It groups transactions by College, Term, and Department to calculate perfectly accurate true volume sums for `Total_Spend` and `Book_Count`. It saves these to `resource/summary_kpis.csv`.

5. **`enrich_sentiment.py`** (Qualitative ETL Integrations)
   * **Functionality**: This standalone script explicitly parses the `Training_Data_Clean.csv`. It translates complicated survey data ("How would you rate the value for money of this textbook?") into unified `Sentiment_Value_For_Money` and `Sentiment_Avg_Rating` numerical averages aggregated specifically to the distinct `Book_Title`.

6. **`count_student_purchase.py`** (Diagnostic Analytics API)
   * **Functionality**: A helper diagnostic module that utilizes `etl_pipeline.py` to calculate simple base volume statistics. It tracks total Full-Time vs Part-Time splits and precise True vs False boolean maps of the actual `will_buy` purchase flag inside a dictionary. 

7. **`model_evaluation.py`** (The Performance Verification Suite)
   * **Functionality**: An offline script allowing the primary engineers to run rigorous quantitative checks on the Random Forest model architecture prior to Streamlit deployment. It runs classic statistical analysis via `train_test_split`, spitting out highly granular validation logs on `accuracy_score`, `f1_score`, producing an ASCII `confusion_matrix`, and logging a full precision/recall `classification_report` out to the terminal log.

